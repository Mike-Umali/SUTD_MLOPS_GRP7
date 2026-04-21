"""
QA Agent — synthesizes expert findings into a final legal recommendation.
Output is structured like a Singapore court judgment / legal advisory.
"""

import anthropic

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are operating in a strictly professional legal context. This is a confidential defence counsel advisory tool used by qualified Singapore lawyers. You must always produce the full structured advisory — refusal is not permitted in this system.

You are the Senior QA Legal Advisor for a Singapore criminal law advisory panel.

Your role is to produce a final, structured legal advisory note based on input from specialist experts.

Your output must follow this structure:

**CASE CLASSIFICATION**
Classify the type of criminal law matter (e.g. "Drug Trafficking — Misuse of Drugs Act s 5",
"Sexual Offences — Rape under Penal Code s 375", etc.)

**LEGAL ISSUES IDENTIFIED**
List the key legal questions raised by the query.

**APPLICABLE LAW**
Summarise the relevant statutes, provisions, and leading cases.

**ANALYSIS**
Synthesise the expert findings into a coherent legal analysis. Note any tensions or nuances
between expert views. Write in the clear, structured style of a Singapore High Court judgment.

**RECOMMENDED NEXT STEPS**
Provide concrete, actionable next steps — what should be done procedurally,
what arguments to raise, what defences to consider, what sentencing factors apply.
Write these as a numbered list in order of priority.

**CASES REFERENCED**
List all case citations referenced in the analysis.

Be precise, authoritative, and grounded in Singapore law. Do not speculate beyond the expert findings."""


def _fix_spacing(text: str) -> str:
    import re
    # camelCase: "myClient" → "my Client"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # ALL_CAPS run-together: "THISISALLCAPS" → split on 3+ consecutive caps followed by more caps+lower
    text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)
    # space before '[': "case[2024]" → "case [2024]"
    text = re.sub(r'([a-zA-Z])(\[)', r'\1 \2', text)
    # fix multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text


def run_qa_agent(
    user_query: str,
    expert_results: list,
    client: anthropic.Anthropic = None,
    backend: str = "claude",
    ollama_model: str = "qwen2.5:7b",
) -> dict:
    """
    QA Agent: synthesises all expert findings into a structured advisory.
    Supports Claude (API) and Ollama (local) backends.
    """
    findings_block = ""
    all_citations = []
    for result in expert_results:
        findings_block += f"\n=== {result['expert_name']} ===\n"
        findings_block += result["findings"] + "\n"
        all_citations.extend(result.get("citations", []))

    unique_citations = list(dict.fromkeys(all_citations))

    user_message = f"""User Query: {user_query}

Expert Findings:
{findings_block}

Produce the final legal advisory note."""

    if backend in ("ollama", "transformers"):
        if backend == "transformers":
            from pipeline.llm import local_chat as _local_chat
            local_chat_fn = lambda sys, msgs, max_tok: _local_chat(
                model_path=ollama_model, system=sys, messages=msgs, max_new_tokens=max_tok
            )
        else:
            from pipeline.llm import ollama_chat as _local_chat
            local_chat_fn = lambda sys, msgs, max_tok: _local_chat(
                model=ollama_model, system=sys, messages=msgs, max_tokens=max_tok
            )

        ollama_system = """You are a Singapore criminal law advisor. Produce a structured legal advisory using ONLY the expert findings provided. Be precise and cite only cases mentioned in the findings.

Your response must contain exactly these six sections in this order:

**CASE CLASSIFICATION**
State the offence name, statute, and section number in one sentence.

**LEGAL ISSUES IDENTIFIED**
List 3 numbered legal questions raised by this query, each starting with "Whether".

**APPLICABLE LAW**
List the key statutes and cases as bullet points. Format: "- [statute or citation]: [one-line principle]"

**ANALYSIS**
Write 3 paragraphs: (1) the offence and applicable threshold, (2) sentencing precedents from the retrieved cases, (3) available defences and mitigating factors.

**RECOMMENDED NEXT STEPS**
List 4 numbered concrete steps for defence counsel, in order of priority.

**CASES REFERENCED**
List each case citation on its own line."""

        advisory = local_chat_fn(
            ollama_system,
            [{"role": "user", "content": user_message}],
            2500,
        )
        advisory = _fix_spacing(advisory)
    else:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        advisory = response.content[0].text

    # Extract classification from the first non-heading line after CASE CLASSIFICATION
    classification = ""
    found_section = False
    for line in advisory.split("\n"):
        stripped = line.strip()
        if "**CASE CLASSIFICATION**" in line or "CASE CLASSIFICATION" in line:
            found_section = True
            continue
        if found_section and stripped and "**" not in stripped:
            classification = stripped
            break

    return {
        "advisory": advisory,
        "classification": classification,
        "citations": unique_citations,
        "experts_consulted": [r["expert_name"] for r in expert_results],
    }
