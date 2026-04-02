"""
LoRA Fine-tuning Script for Criminal Law Domain

This script fine-tunes an Ollama-compatible model using LoRA (Low-Rank Adaptation)
on criminal law examples to improve domain-specific legal reasoning.

Requirements:
  pip install transformers peft torch datasets bitsandbytes

Usage:
  python lora_finetune.py --data lora_training_data.jsonl --model mistral --output criminal_law_lora
  python lora_finetune.py --data lora_training_data.jsonl --ollama-model mistral --output criminal_law_lora
"""

import json
import torch
import argparse
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType


# Map Ollama model names to HuggingFace equivalents (all open-access)
OLLAMA_TO_HF_MODELS = {
    "mistral": "mistralai/Mistral-7B-v0.1",
    "llama3.1:8b": "NousResearch/Llama-2-7b-hf",  # Open-access alternative to meta-llama
    "llama2": "NousResearch/Llama-2-7b-hf",
    "neural-chat": "Intel/neural-chat-7b-v3",
    "orca-mini": "psmathur/orca_mini_v3_7B",
}


def load_training_data(jsonl_path: str) -> dict:
    """Load JSONL training data in format: {"prompt": "...", "response": "..."}"""
    examples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
    
    # Combine prompt and response
    texts = [f"{ex['prompt']}{ex['response']}" for ex in examples]
    return {"text": texts}


def setup_training(
    model_name: str,
    output_dir: str,
    use_8bit: bool = True,
    lora_r: int = 16,
) -> tuple:
    """Setup model, tokenizer, and LoRA config."""
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config (for memory efficiency)
    bnb_config = None
    device_map = "auto"
    
    if use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if use_8bit else None,
        device_map=device_map,
        torch_dtype=torch.float16 if not use_8bit else None,
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing (saves memory)
    model.gradient_checkpointing_enable()
    
    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer, lora_config


def train(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
) -> None:
    """Fine-tune the model with LoRA."""
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_32bit",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=None,  # will use default
    )
    
    trainer.train()
    
    # Save LoRA adapter
    model.save_pretrained(f"{output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{output_dir}/lora_adapter")
    
    print(f"✓ LoRA adapter saved to {output_dir}/lora_adapter")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune for criminal law domain")
    parser.add_argument(
        "--data",
        type=str,
        default="lora_training_data.jsonl",
        help="Path to JSONL training data (format: {\"prompt\": \"...\", \"response\": \"...\"})",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name (e.g., mistralai/Mistral-7B-v0.1)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        help="Ollama model name (e.g., mistral, llama3.1:8b) - will be converted to HuggingFace equivalent",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./lora_models/criminal_law",
        help="Output directory for LoRA adapter",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--no-8bit",
        action="store_true",
        help="Disable 8-bit quantization (use if you have plenty of VRAM)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (16-64 typical)",
    )
    
    args = parser.parse_args()
    
    # Convert Ollama model name to HuggingFace if needed
    model_name = args.model
    if args.ollama_model:
        if args.ollama_model in OLLAMA_TO_HF_MODELS:
            model_name = OLLAMA_TO_HF_MODELS[args.ollama_model]
            print(f"📍 Mapping Ollama model '{args.ollama_model}' → HuggingFace '{model_name}'")
        else:
            print(f"⚠️  Unknown Ollama model: {args.ollama_model}")
            print(f"   Known models: {', '.join(OLLAMA_TO_HF_MODELS.keys())}")
            return
    
    if not model_name:
        print("❌ Error: Must provide either --model (HuggingFace) or --ollama-model (Ollama)")
        return
    
    # Validate input
    if not Path(args.data).exists():
        print(f"❌ Training data not found: {args.data}")
        print("Create a JSONL file with format: {\"prompt\": \"...\", \"response\": \"...\"}")
        return
    
    print(f"📚 Loading training data from {args.data}...")
    data_dict = load_training_data(args.data)
    dataset = Dataset.from_dict(data_dict)
    print(f"✓ Loaded {len(dataset)} examples")
    
    print(f"\n🤖 Setting up model: {model_name}")
    model, tokenizer, lora_config = setup_training(
        model_name=model_name,
        output_dir=args.output,
        use_8bit=not args.no_8bit,
        lora_r=args.lora_r,
    )
    
    print(f"📊 LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"   Target modules: {lora_config.target_modules}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\n🚀 Starting training ({args.epochs} epochs, batch_size={args.batch_size})...")
    train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"   LoRA adapter saved to: {args.output}/lora_adapter")
    print(f"\n📝 Next steps:")
    print(f"  1. Install llama-cpp-python: pip install llama-cpp-python")
    print(f"  2. Merge and convert to GGUF format:")
    print(f"     python merge_lora_to_gguf.py --base-model {model_name} --lora {args.output}/lora_adapter --output final_model.gguf")
    print(f"  3. Create Ollama Modelfile and import:")
    print(f"     ollama create {args.ollama_model}-criminal --modelfile Modelfile")
    print(f"  4. Test in your Streamlit app!")


if __name__ == "__main__":
    main()
