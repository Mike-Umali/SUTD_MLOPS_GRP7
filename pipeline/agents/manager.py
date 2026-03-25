"""
Manager Agent — orchestrates expert agents using Claude tool_use.
Classifies the query, selects relevant experts, collects findings.
"""

import json
import anthropic
from pipeline.agents.experts import run_expert_agent, EXPERT_PROFILES

MODEL = "claude-sonnet-4-6"

# Tool definitions — one per expert domain
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


def run_manager_agent(user_query: str, client: anthropic.Anthropic) -> dict:
    """
    Manager agent: uses tool_use to consult relevant expert agents.
    Returns aggregated findings from all consulted experts.
    """
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

    # Agentic loop — keep going until Claude stops calling tools
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system_prompt,
            tools=EXPERT_TOOLS,
            messages=messages,
        )

        # Collect any text from manager
        manager_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                manager_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(block)

        # No more tool calls — done
        if response.stop_reason == "end_turn" or not tool_calls:
            break

        # Append assistant response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool call (expert agent)
        tool_results = []
        for tool_call in tool_calls:
            domain = tool_call.name.replace("consult_", "")
            query = tool_call.input.get("query", user_query)

            print(f"  Manager → consulting {domain} expert...")
            result = run_expert_agent(domain, query, client)
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

        # Feed tool results back to manager
        messages.append({"role": "user", "content": tool_results})

    return {
        "manager_summary": manager_text,
        "expert_results": expert_results,
        "experts_consulted": [r["expert_name"] for r in expert_results],
    }
