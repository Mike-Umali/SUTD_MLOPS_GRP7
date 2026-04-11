"""
Merge LoRA adapters into base model and save as full HuggingFace model.
Run with: conda run -n base python3 merge_lora.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR  = "lora_adapters"
OUTPUT_DIR   = "sg-law-merged"

print(f"Loading base model: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(f"Loading LoRA adapters from: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to: {OUTPUT_DIR}/")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done. Merged model saved.")
