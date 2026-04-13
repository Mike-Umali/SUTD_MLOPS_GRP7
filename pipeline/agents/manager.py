"""
Manager Agent — orchestrates expert agents using Claude tool_use (or Ollama JSON routing).
Classifies the query, selects relevant experts, collects findings.
"""

import json
import re
import anthropic
from pipeline.agents.experts import run_expert_agent, EXPERT_PROFILES

MODEL = "claude-sonnet-4-6"

# Tool definitions — one per expert domain (Claude mode only)
EXPERT_TOOLS = [
    {
        "name": f"consult_{domain}",
        "description": f"Consult the {profile['name']}. {profile['expertise'][:120]}...",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific legal question to ask this expert.",
                }
            },
            "required": ["query"],
        },
    }
    for domain, profile in EXPERT_PROFILES.items()
]


def run_manager_agent(
    user_query: str,
    client: anthropic.Anthropic = None,
    backend: str = "claude",
    ollama_model: str = "qwen2.5:7b",
) -> dict:
    """
    Manager agent: routes the query to relevant expert agents.
    Claude mode: agentic tool_use loop.
    Ollama / Transformers mode: JSON routing prompt, then runs experts sequentially.
    """
    if backend in ("ollama", "transformers"):
        return _run_manager_local(user_query, ollama_model, backend)
    return _run_manager_claude(user_query, client)


# ── Claude mode ───────────────────────────────────────────────────────────────

def _run_manager_claude(user_query: str, client: anthropic.Anthropic) -> dict:
    system_prompt = """You are the Lead Criminal Law Manager Agent for a Singapore criminal law advisory system.

Your role is to:
1. Analyse the user's legal query
2. Identify which expert agents are relevant (may be multiple)
3. Consult each relevant expert using the provided tools
4. You may consult multiple experts if the case involves overlapping areas (e.g. drug trafficking + sentencing)

Always consult the Sentencing Expert for any query that involves penalties or what happens next.
Always consult the Criminal Procedure Expert if the query involves court process, appeals, or procedure.
Be selective — only consult experts relevant to the query."""

    messages = [{"role": "user", "content": user_query}]
    expert_results = []

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system_prompt,
            tools=EXPERT_TOOLS,
            messages=messages,
        )

        manager_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                manager_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(block)

        if response.stop_reason == "end_turn" or not tool_calls:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_call in tool_calls:
            domain = tool_call.name.replace("consult_", "")
            query = tool_call.input.get("query", user_query)

            print(f"  Manager → consulting {domain} expert...")
            result = run_expert_agent(domain, query, client=client, backend="claude")
            expert_results.append(result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": json.dumps({
                    "expert": result["expert_name"],
                    "findings": result["findings"],
                    "citations": result["citations"],
                }),
            })

        messages.append({"role": "user", "content": tool_results})

    return {
        "manager_summary": manager_text,
        "expert_results": expert_results,
        "experts_consulted": [r["expert_name"] for r in expert_results],
    }


# ── Local model mode (Ollama or Transformers) ─────────────────────────────────

_KEYWORD_DOMAINS = {
    "drug_offences": ["drug", "heroin", "cannabis", "meth", "trafficking", "mda", "diamorphine", "opium", "cocaine", "possession"],
    "sexual_offences": ["rape", "sexual", "molest", "outrage", "voyeur", "intimate", "minor", "exploitation"],
    "violent_crimes": ["murder", "homicide", "hurt", "assault", "grievous", "robbery", "kidnap", "stab", "kill", "death"],
    "property_financial": ["theft", "cheat", "cbt", "fraud", "money laundering", "corruption", "bribery", "forgery", "extortion"],
    "regulatory": ["road", "traffic", "immigration", "company", "workplace", "computer misuse", "customs", "drink driving"],
}


def _keyword_domains(query: str) -> list:
    """Return domains triggered by keywords in the query."""
    q = query.lower()
    return [d for d, kws in _KEYWORD_DOMAINS.items() if any(kw in q for kw in kws)]


def _run_manager_local(user_query: str, model_name: str, backend: str) -> dict:
    """JSON routing for local backends (Ollama or Transformers)."""
    if backend == "transformers":
        from pipeline.llm import transformers_chat
        chat_fn = lambda sys, msgs, max_tok: transformers_chat(
            model_path=model_name, system=sys, messages=msgs, max_new_tokens=max_tok
        )
    else:
        from pipeline.llm import ollama_chat
        chat_fn = lambda sys, msgs, max_tok: ollama_chat(
            model=model_name, system=sys, messages=msgs, max_tokens=max_tok
        )

    domain_list = list(EXPERT_PROFILES.keys())

    system = (
        "You are a Singapore criminal law routing agent. "
        "Given a legal query, decide which expert domains to consult. "
        "Always include sentencing and criminal_procedure. "
        f"Choose only from this list: {domain_list}. "
        "Reply with ONLY a JSON array, e.g. [\"drug_offences\", \"sentencing\", \"criminal_procedure\"]"
    )

    raw = chat_fn(
        system,
        [{"role": "user", "content": f"Query: {user_query}"}],
        150,
    )

    # Extract the JSON array from the response
    domains = []
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            domains = json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Keyword fallback — ensure domain is included even if model missed it
    for d in _keyword_domains(user_query):
        if d not in domains:
            domains.append(d)

    # Ensure required domains and strip any invalid ones
    for required in ("sentencing", "criminal_procedure"):
        if required not in domains:
            domains.append(required)
    domains = [d for d in domains if d in EXPERT_PROFILES]

    # Run each expert sequentially
    expert_results = []
    for domain in domains:
        print(f"  Manager → consulting {domain} expert ({backend})...")
        result = run_expert_agent(
            domain, user_query,
            backend=backend,
            ollama_model=model_name,
        )
        expert_results.append(result)

    return {
        "manager_summary": f"Routed to: {', '.join(domains)}",
        "expert_results": expert_results,
        "experts_consulted": [r["expert_name"] for r in expert_results],
    }
