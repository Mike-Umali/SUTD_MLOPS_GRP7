"""
Manager Agent — routes the query to the relevant expert agents using Ollama JSON routing.
Classifies the query, selects relevant experts, and collects findings.
"""

import json
import re

from pipeline.agents.experts import EXPERT_PROFILES, run_expert_agent

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


def run_manager_agent(
    user_query: str,
    backend: str = "ollama",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> dict:
    """
    Manager agent: routes the query to relevant expert agents.
    Ollama mode: JSON routing prompt, then runs experts sequentially.
    """
    if backend != "ollama":
        raise ValueError("This manager.py is configured for Ollama only.")

    return _run_manager_ollama(user_query, ollama_model)


def _run_manager_ollama(user_query: str, ollama_model: str) -> dict:
    from pipeline.llm import ollama_chat

    domain_list = list(EXPERT_PROFILES.keys())

    system = (
        "You are a Singapore criminal law routing agent. "
        "Your task is to choose the most relevant expert domains for the user's query. "
        "Return ONLY a valid JSON array of strings. "
        "Do not include explanations, markdown, headings, or code fences. "
        "Always include 'sentencing' and 'criminal_procedure'. "
        f"Valid domains are: {domain_list}."
    )

    raw = ollama_chat(
        model=ollama_model,
        system=system,
        messages=[{"role": "user", "content": f"Query: {user_query}"}],
        max_tokens=150,
    )

    domains = _parse_domain_array(raw)

    for required in ("sentencing", "criminal_procedure"):
        if required not in domains:
            domains.append(required)

    domains = [d for d in domains if d in EXPERT_PROFILES]

    if not domains:
        domains = ["sentencing", "criminal_procedure", "general"]

    expert_results = []
    for domain in domains:
        print(f"  Manager → consulting {domain} expert (Ollama)...")
        result = run_expert_agent(
            domain=domain,
            query=user_query,
            backend="ollama",
            ollama_model=ollama_model,
        )
        expert_results.append(result)

    return {
        "manager_summary": f"Routed to: {', '.join(domains)}",
        "expert_results": expert_results,
        "experts_consulted": [r["expert_name"] for r in expert_results],
    }


def _parse_domain_array(raw: str) -> list[str]:
    """
    Extract and parse a JSON array of domain names from the model response.
    """
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return [item for item in parsed if isinstance(item, str)]