"""
Merge LoRA adapter with base model and convert to GGUF format for Ollama

This script:
1. Loads the base model and LoRA adapter
2. Merges them into a single model
3. Converts to GGUF format (compatible with Ollama)

Requirements:
  pip install transformers peft torch llama-cpp-python

Usage:
  python merge_and_convert.py \
    --base-model mistralai/Mistral-7B-v0.1 \
    --lora ./lora_models/criminal_law/lora_adapter \
    --output ./final_model.gguf
"""

import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora(base_model_name: str, lora_path: str, output_path: str):
    """Merge LoRA adapter with base model and save as regular model."""
    
    print(f"📚 Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    
    print(f"🔗 Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print(f"🔄 Merging LoRA into base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    output_dir = Path(output_path).parent / f"{Path(output_path).stem}_merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving merged model to: {output_dir}")
    merged_model.save_pretrained(str(output_dir))
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"✓ Merged model saved to: {output_dir}")
    print(f"\nNext step: Convert to GGUF using llama.cpp")
    print(f"  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
    print(f"  2. Convert model:")
    print(f"     python llama.cpp/convert.py {output_dir}")
    print(f"  3. Import to Ollama:")
    print(f"     ollama create criminal-law-mistral -f Modelfile")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and convert to GGUF")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model name (e.g., mistralai/Mistral-7B-v0.1)",
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./final_model",
        help="Output path for GGUF model",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.lora).exists():
        print(f"❌ LoRA path not found: {args.lora}")
        return
    
    merge_lora(args.base_model, args.lora, args.output)


if __name__ == "__main__":
    main()
