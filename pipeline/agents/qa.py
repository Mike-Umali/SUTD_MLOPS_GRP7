"""
QA Agent — synthesizes expert findings into a final legal recommendation.
Output is structured like a Singapore court judgment / legal advisory.
"""

SYSTEM_PROMPT = """You are the Senior QA Legal Advisor for a Singapore criminal law advisory panel.

You will be given:
1. A user query
2. Findings from specialist experts

Your task is to produce a final structured legal advisory based only on the supplied expert findings.

Requirements:
- Follow the exact section headings below
- Be precise, professional, and grounded in Singapore criminal law
- Use only the supplied expert findings
- Do not invent case citations, statutes, or legal principles
- If the expert findings are insufficient on a point, say so clearly
- Do not speculate beyond the supplied material

Your output must follow this structure exactly:

**CASE CLASSIFICATION**
Classify the type of criminal law matter (e.g. "Drug Trafficking — Misuse of Drugs Act s 5",
"Sexual Offences — Rape under Penal Code s 375", etc.)

**LEGAL ISSUES IDENTIFIED**
List the key legal questions raised by the query.

**APPLICABLE LAW**
Summarise the relevant statutes, provisions, and leading cases.

**ANALYSIS**
Synthesise the expert findings into a coherent legal analysis. Note any tensions or nuances
between expert views. Write in a clear, structured style.

**RECOMMENDED NEXT STEPS**
Provide concrete, actionable next steps. Write these as a numbered list in order of priority.

**CASES REFERENCED**
List all case citations referenced in the analysis.
"""


def run_qa_agent(
    user_query: str,
    expert_results: list,
    backend: str = "ollama",
    ollama_model: str = "llama3.1:8b",
) -> dict:
    """
    QA Agent: synthesises all expert findings into a structured advisory.
    Ollama-only version.
    """
    from pipeline.llm import ollama_chat

    findings_block = ""
    all_citations = []

    for result in expert_results:
        expert_name = result.get("expert_name", "Unknown Expert")
        findings = result.get("findings", "")
        findings_block += f"\n=== {expert_name} ===\n"
        findings_block += findings + "\n"
        all_citations.extend(result.get("citations", []))

    unique_citations = list(dict.fromkeys(all_citations))

    user_message = f"""User Query: {user_query}

Expert Findings:
{findings_block}

Produce the final legal advisory note using only the expert findings above.
If the findings are incomplete, state that clearly instead of guessing."""

    advisory = ollama_chat(
        model=ollama_model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=1200,
    )

    classification = _extract_classification(advisory)

    return {
        "advisory": advisory,
        "classification": classification,
        "citations": unique_citations,
        "experts_consulted": [r.get("expert_name", "Unknown Expert") for r in expert_results],
    }


def _extract_classification(advisory: str) -> str:
    """
    Extract classification from the first non-heading line after CASE CLASSIFICATION.
    """
    classification = ""
    found_section = False

    for line in advisory.split("\n"):
        stripped = line.strip()

        if "**CASE CLASSIFICATION**" in stripped or stripped.upper() == "CASE CLASSIFICATION":
            found_section = True
            continue

        if found_section:
            if not stripped:
                continue
            if stripped.startswith("**") and stripped.endswith("**"):
                break
            classification = stripped
            break

    return classification