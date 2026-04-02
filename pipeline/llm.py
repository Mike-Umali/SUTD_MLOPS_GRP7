"""
LLM backend abstraction — Claude (Anthropic) or Ollama (local/offline).
"""


def ollama_chat(
    model: str,
    system: str,
    messages: list,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """
    Send a chat request to a local Ollama model and return response text.
    
    Temperature: 0.1-0.3 for structured/legal outputs, 0.7-0.9 for creative
    """
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
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
        },
    )
    return response["message"]["content"]


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
        return [m["name"] for m in result.get("models", [])]
    except Exception:
        return []
