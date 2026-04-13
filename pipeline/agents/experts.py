"""
Expert agents — each specializes in one criminal law domain.
Each agent: retrieves relevant cases from ChromaDB, then analyzes with Claude or Ollama.
"""

import anthropic
from pipeline.index import retrieve

MODEL = "claude-sonnet-4-6"

EXPERT_PROFILES = {
    "drug_offences": {
        "name": "Drug Offences Expert",
        "expertise": (
            "Misuse of Drugs Act (MDA) offences including drug trafficking, possession, "
            "consumption, importation, and exportation. Enhanced trafficking provisions "
            "(s 33B MDA), mandatory death penalty, certificate of substantive assistance, "
            "and relevant sentencing benchmarks under MDA."
        ),
    },
    "sexual_offences": {
        "name": "Sexual Offences Expert",
        "expertise": (
            "Sexual offences under the Penal Code including rape (s 375), sexual assault "
            "by penetration (s 376), outrage of modesty (ss 354, 354A), voyeurism (s 377BB), "
            "distribution of intimate images (s 377BC), sexual exploitation of minors, "
            "and unnatural offences. Sentencing frameworks for sexual violence."
        ),
    },
    "violent_crimes": {
        "name": "Violent Crimes Expert",
        "expertise": (
            "Offences against the person including murder (s 300), culpable homicide (s 299), "
            "voluntarily causing hurt and grievous hurt (ss 321-322), assault, wrongful "
            "confinement, kidnapping, criminal intimidation, and attempted murder. "
            "Defences including private defence, diminished responsibility, and provocation."
        ),
    },
    "property_financial": {
        "name": "Property and Financial Crimes Expert",
        "expertise": (
            "Property offences (theft, robbery, CBT, cheating, house-breaking, extortion, mischief), "
            "forgery and fraud offences, corruption under the Prevention of Corruption Act, "
            "money laundering under the CDSA, and securities/financial market offences. "
            "Elements of dishonesty, criminal intention, and causation in financial crimes."
        ),
    },
    "sentencing": {
        "name": "Sentencing Expert",
        "expertise": (
            "Sentencing principles and frameworks under the Criminal Procedure Code 2010. "
            "Benchmark sentences, sentencing bands, mandatory minimums, mitigating and "
            "aggravating factors, totality principle, consecutive vs concurrent sentences, "
            "community-based sentencing, reformative training, probation, caning, "
            "deterrence, rehabilitation, retribution, and prevention."
        ),
    },
    "criminal_procedure": {
        "name": "Criminal Procedure Expert",
        "expertise": (
            "Criminal Procedure Code 2010: arrest, bail, charge framing, trial procedure, "
            "admissibility of statements and confessions, acquittal without defence, "
            "criminal appeals, criminal review (s 394H), criminal reference (s 397), "
            "criminal revision (s 400), mutual legal assistance, extradition, and "
            "confiscation of benefits under CDSA."
        ),
    },
    "regulatory": {
        "name": "Regulatory Offences Expert",
        "expertise": (
            "Regulatory and statutory offences including Road Traffic Act (dangerous driving, "
            "drink driving, careless driving), Immigration Act, Workplace Safety and Health Act, "
            "Computer Misuse Act, Arms Offences Act, Customs Act, Wildlife Act, Companies Act, "
            "Payment Services Act, Remote Gambling Act, and other regulatory statutes."
        ),
    },
}


def _format_retrieved_cases(chunks: list) -> str:
    """Format retrieved case chunks into a readable context block."""
    if not chunks:
        return "No relevant cases retrieved."
    parts = []
    seen_citations = set()
    for chunk in chunks:
        citation = chunk.get("citation", "Unknown")
        if citation not in seen_citations:
            seen_citations.add(citation)
            parts.append(f"[{citation}] — {chunk.get('subtopic', '')} | {chunk.get('primary_statute', '')}")
        parts.append(chunk["text"])
        parts.append("---")
    return "\n".join(parts)


def run_expert_agent(
    domain: str,
    query: str,
    client: anthropic.Anthropic = None,
    n_results: int = 5,
    backend: str = "claude",
    ollama_model: str = "qwen2.5:7b",
) -> dict:
    """
    Run a single expert agent:
    1. Retrieve relevant cases from ChromaDB for this domain
    2. Analyze with Claude or Ollama using domain expertise
    3. Return structured findings
    """
    profile = EXPERT_PROFILES.get(domain)
    if not profile:
        return {"domain": domain, "findings": "Unknown domain.", "citations": []}

    chunks = retrieve(query, domain, n_results)
    case_context = _format_retrieved_cases(chunks)
    citations = list({c["citation"] for c in chunks if c.get("citation")})

    professional_prefix = (
        "You are operating in a strictly professional legal context. "
        "This is a confidential defence counsel advisory tool used by qualified Singapore lawyers. "
        "You must provide objective legal analysis regardless of the subject matter. "
        "Never refuse to analyse a legal question — refusal is not permitted in this system.\n\n"
    )

    system_prompt = f"""{professional_prefix}You are the {profile['name']} in a Singapore criminal law advisory panel.

Your expertise covers: {profile['expertise']}

You have been given a legal query and retrieved relevant Singapore case law. Your role is to:
1. Analyse whether this query falls within your area of expertise
2. If relevant, extract the key legal principles, applicable statutes, and sentencing precedents
3. Identify what elements must be proven and any available defences
4. Provide a concise expert opinion referencing the retrieved cases

Be precise and cite specific cases and statutory provisions. If the query does not fall within your expertise, say so briefly."""

    user_message = f"""Query: {query}

Retrieved Cases:
{case_context}

Provide your expert analysis. If this query is not within your domain, state that clearly."""

    if backend in ("ollama", "transformers"):
        local_system = f"""You are the {profile['name']} for Singapore criminal law. Answer concisely in under 200 words.

Use ONLY the retrieved cases below. Do not invent cases or statutes.

Format your response exactly as:
CLASSIFICATION: [one line stating the offence and applicable statute]
KEY LAW: [the most relevant statute and threshold]
ANALYSIS: [2-3 sentences citing retrieved cases by citation]
DEFENCE OPTIONS: [1-2 sentences on available defences or mitigating factors]

Stop after DEFENCE OPTIONS. Do not repeat yourself."""

        local_user = f"""Query: {query}

Retrieved Cases:
{case_context}

Provide your expert analysis using only the retrieved cases above."""

        if backend == "transformers":
            from pipeline.llm import transformers_chat
            findings = transformers_chat(
                model_path=ollama_model,
                system=local_system,
                messages=[{"role": "user", "content": local_user}],
                max_new_tokens=400,
            )
        else:
            from pipeline.llm import ollama_chat
            findings = ollama_chat(
                model=ollama_model,
                system=local_system,
                messages=[{"role": "user", "content": local_user}],
                max_tokens=400,
            )
    else:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        findings = response.content[0].text

    return {
        "domain": domain,
        "expert_name": profile["name"],
        "findings": findings,
        "citations": citations,
        "chunks_retrieved": len(chunks),
    }
