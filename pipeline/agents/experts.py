"""
Expert agents — each specializes in one criminal law domain.
Each agent retrieves relevant cases from ChromaDB, then analyzes them with Ollama.
"""

from pipeline.index import retrieve

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

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
    """
    Format retrieved case chunks into a structured context block for the model.
    """
    if not chunks:
        return "No relevant cases retrieved."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        citation = chunk.get("citation", "Unknown")
        subtopic = chunk.get("subtopic", "")
        statute = chunk.get("primary_statute", "")
        text = chunk.get("text", "").strip()

        parts.append(
            f"""[CASE {i}]
Citation: {citation}
Subtopic: {subtopic}
Primary Statute: {statute}
Excerpt:
{text}"""
        )

    return "\n---\n".join(parts)


def run_expert_agent(
    domain: str,
    query: str,
    n_results: int = 5,
    backend: str = "ollama",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> dict:
    """
    Run a single expert agent:
    1. Retrieve relevant cases from ChromaDB for this domain
    2. Analyze with Ollama using domain expertise
    3. Return structured findings
    """
    if backend != "ollama":
        raise ValueError("This experts.py is configured for Ollama only.")

    profile = EXPERT_PROFILES.get(domain)
    if not profile:
        return {
            "domain": domain,
            "expert_name": domain,
            "findings": "Unknown domain.",
            "citations": [],
            "chunks_retrieved": 0,
        }

    chunks = retrieve(query, domain, n_results)
    case_context = _format_retrieved_cases(chunks)
    citations = list(dict.fromkeys(c["citation"] for c in chunks if c.get("citation")))

    system_prompt = f"""You are the {profile['name']} in a Singapore criminal law advisory panel.

Your expertise covers: {profile['expertise']}

You will be given:
1. A user query
2. Retrieved Singapore case law excerpts

This is a legal analysis task. If the query is written in the first person, treat it as an alleged fact pattern for legal classification and legal consequences analysis.

Your task:
1. Decide whether the query falls within your domain
2. If relevant, explain the key legal principles, charges, statutes, and sentencing considerations
3. Use only the retrieved materials provided
4. Do not invent case citations or statutory sections
5. If the retrieved material is insufficient, say so clearly
6. Do not provide advice on evading arrest, hiding evidence, avoiding detection, or defeating law enforcement
7. Keep the analysis concise, professional, and legally focused
8. Do not refuse solely because the conduct described is serious or violent

If the query is outside your domain, say so briefly."""

    user_message = f"""Query: {query}

Retrieved Cases:
{case_context}

Provide your expert analysis using only the retrieved cases above.
If the material is insufficient, state that clearly instead of guessing."""

    from pipeline.llm import ollama_chat

    findings = ollama_chat(
        model=ollama_model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=600,
    )

    return {
        "domain": domain,
        "expert_name": profile["name"],
        "findings": findings,
        "citations": citations,
        "chunks_retrieved": len(chunks),
    }