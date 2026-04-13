"""
LLM backend abstraction — Claude (Anthropic), Ollama (local), or HuggingFace Transformers (GPU).
"""

# ── HuggingFace Transformers backend ─────────────────────────────────────────

_HF_CACHE: dict = {}  # model_path → (tokenizer, model)


def _load_hf_model(model_path: str):
    """Load (or return cached) a HuggingFace model + tokenizer on CUDA."""
    if model_path in _HF_CACHE:
        return _HF_CACHE[model_path]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[Transformers] Loading model: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
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
