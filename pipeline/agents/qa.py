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
    """Fix words running together — insert space before capitals mid-word."""
    import re
    # Insert space between a lowercase letter and an uppercase letter (e.g. "myclient" → "my client")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Insert space before '[' if preceded by a letter (e.g. "case[2024]" → "case [2024]")
    text = re.sub(r'([a-zA-Z])(\[)', r'\1 \2', text)
    # Fix multiple spaces
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
            from pipeline.llm import transformers_chat as _local_chat
            local_chat_fn = lambda sys, msgs, max_tok: _local_chat(
                model_path=ollama_model, system=sys, messages=msgs, max_new_tokens=max_tok
            )
        else:
            from pipeline.llm import ollama_chat as _local_chat
            local_chat_fn = lambda sys, msgs, max_tok: _local_chat(
                model=ollama_model, system=sys, messages=msgs, max_tokens=max_tok
            )

        ollama_system = """You are the Senior QA Legal Advisor for a Singapore criminal law advisory panel. \
Write a confidential defence counsel advisory note in the authoritative style of a Singapore High Court judgment — \
precise, formal, structured, and grounded in statute and case law.

Follow this exact format:

**CONFIDENTIAL LEGAL ADVISORY NOTE**
*Prepared for Defence Counsel | Singapore Criminal Law Advisory Panel*

**CASE CLASSIFICATION**
One sentence classifying the matter (e.g. "Sexual Offences — Rape under Penal Code s 375, read with s 90(b)").

**LEGAL ISSUES IDENTIFIED**
1. Whether [first issue, framed as a legal question]
2. Whether [second issue]
3. Whether [third issue]

**APPLICABLE LAW**
*Statutory Provisions:*
- **s [X] [Act]** — [what it provides]

*Leading Cases:*
- **[Citation]** — [the legal principle it establishes]

**ANALYSIS**
Write 3–4 paragraphs in High Court judgment style. Each paragraph addresses one legal issue. \
Bold key legal terms and thresholds. Cite cases inline as **[Case Name] [Citation]**. \
Identify strengths and weaknesses of the defence position. Do not repeat any sentence or point.

**RECOMMENDED NEXT STEPS**
Number each step in order of urgency. Be concrete and actionable (e.g. "Apply for bail under s 95 CPC", \
"Commission forensic pharmacology expert").
1. [Most urgent step]
2. [Second step]
3. [Third step]

**CASES REFERENCED**
| Citation | Relevance |
|----------|-----------|
| [citation] | [one-line relevance] |

Rules: every section appears exactly once. Write formally. Do not repeat sentences. Stop after the table."""

        advisory = local_chat_fn(
            ollama_system,
            [{"role": "user", "content": user_message}],
            1200,
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
