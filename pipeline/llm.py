"""
LLM backend abstraction — Claude (Anthropic), Ollama (local), HuggingFace Transformers (GPU), or llama-cpp (GGUF).
"""

# ── GGUF / llama-cpp backend ──────────────────────────────────────────────────

_LLAMA_CACHE: dict = {}  # model_path → Llama instance


def _load_llama(model_path: str):
    """Load (or return cached) a GGUF model via llama-cpp-python on GPU."""
    if model_path in _LLAMA_CACHE:
        return _LLAMA_CACHE[model_path]

    from llama_cpp import Llama
    print(f"[llama-cpp] Loading GGUF model: {model_path} ...")
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,   # offload all layers to GPU
        n_ctx=4096,
        verbose=False,
    )
    _LLAMA_CACHE[model_path] = model
    print(f"[llama-cpp] Model loaded.")
    return model


def llama_cpp_chat(
    model_path: str,
    system: str,
    messages: list,
    max_new_tokens: int = 512,
) -> str:
    """Run inference on a GGUF model using llama-cpp-python."""
    model = _load_llama(model_path)

    chat_messages = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    chat_messages.extend(messages)

    response = model.create_chat_completion(
        messages=chat_messages,
        max_tokens=max_new_tokens,
        temperature=0.3,
        repeat_penalty=1.5,
    )
    return response["choices"][0]["message"]["content"].strip()


def local_chat(
    model_path: str,
    system: str,
    messages: list,
    max_new_tokens: int = 512,
) -> str:
    """Auto-route to llama-cpp (GGUF) or transformers (HF) based on model path."""
    if model_path.endswith(".gguf"):
        return llama_cpp_chat(model_path, system, messages, max_new_tokens)
    return transformers_chat(model_path, system, messages, max_new_tokens)


# ── HuggingFace Transformers backend ─────────────────────────────────────────

_HF_CACHE: dict = {}  # model_path → (tokenizer, model)


def _load_hf_model(model_path: str):
    """Load (or return cached) a HuggingFace model + tokenizer on CUDA.
    Supports plain HF models and LoRA adapter repos (detected via adapter_config.json)."""
    if model_path in _HF_CACHE:
        return _HF_CACHE[model_path]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Check if this is a LoRA adapter repo
    try:
        from huggingface_hub import hf_hub_download
        import json
        cfg_path = hf_hub_download(repo_id=model_path, filename="adapter_config.json")
        with open(cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg["base_model_name_or_path"]
        is_lora = True
        print(f"[Transformers] LoRA adapter detected. Base model: {base_model}")
    except Exception:
        base_model = model_path
        is_lora = False

    print(f"[Transformers] Loading model: {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # Load in fp16 — torch_dtype=float16 overrides any quantization_config in the model config
    # (bitsandbytes is not supported on Blackwell/sm_120)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if is_lora:
        from peft import PeftModel
        print(f"[Transformers] Applying LoRA adapter: {model_path} ...")
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
        model = model.merge_and_unload()
        print(f"[Transformers] LoRA merged into base model (fp16).")

    model.eval()
    _HF_CACHE[model_path] = (tokenizer, model)
    print(f"[Transformers] Model loaded on {next(model.parameters()).device}")
    return tokenizer, model


def transformers_chat(
    model_path: str,
    system: str,
    messages: list,
    max_new_tokens: int = 1024,
) -> str:
    """Run inference on a local HuggingFace model using CUDA."""
    import torch

    tokenizer, model = _load_hf_model(model_path)

    chat_messages = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    chat_messages.extend(messages)

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    else:
        # Fallback for models without a chat template
        prompt = (f"System: {system}\n\n" if system else "")
        for msg in messages:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += "Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Ollama backend ────────────────────────────────────────────────────────────

def ollama_chat(model: str, system: str, messages: list, max_tokens: int = 4096) -> str:
    """Send a chat request to a local Ollama model and return response text."""
    import ollama as _ollama
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)
    response = _ollama.chat(
        model=model,
        messages=full_messages,
        options={
            "num_predict": max_tokens,
            "repeat_penalty": 1.6,
            "repeat_last_n": 512,
            "temperature": 0.3,
        },
    )
    return response.message.content


def ollama_available() -> bool:
    """Check if Ollama is reachable at localhost:11434."""
    try:
        import ollama as _ollama
        _ollama.list()
        return True
    except Exception:
        return False


def list_ollama_models() -> list:
    """Return locally available Ollama model names."""
    try:
        import ollama as _ollama
        result = _ollama.list()
        return [m.model for m in result.models]
    except Exception:
        return []
