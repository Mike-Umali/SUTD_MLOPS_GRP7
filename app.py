"""
Streamlit app — Singapore Criminal Law Advisory System.
Multi-agent RAG pipeline: Manager → Expert Agents → QA Agent.
Ollama-only local/offline backend.
"""

import streamlit as st

from pipeline.agents.manager import run_manager_agent
from pipeline.agents.qa import run_qa_agent
from pipeline.llm import ollama_available, list_ollama_models

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

st.set_page_config(
    page_title="SG Criminal Law Advisory",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")

    ollama_ok = ollama_available()

    if ollama_ok:
        st.success("Ollama is running")
        available_models = list_ollama_models()

        if available_models:
            default_index = (
                available_models.index(DEFAULT_OLLAMA_MODEL)
                if DEFAULT_OLLAMA_MODEL in available_models
                else 0
            )
            ollama_model = st.selectbox(
                "Model",
                options=available_models,
                index=default_index,
                help="Select a locally available Ollama model.",
            )
        else:
            st.warning("No local models found. Run: `ollama pull llama3.1:8b`")
            ollama_model = st.text_input("Model name", value=DEFAULT_OLLAMA_MODEL)
    else:
        st.error("Ollama not reachable. Start it with: `ollama serve`")
        ollama_model = st.text_input("Model name", value=DEFAULT_OLLAMA_MODEL)

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Multi-agent RAG pipeline over **876 Singapore criminal judgments** (2015–2026) "
        "from the Supreme Court of Singapore."
    )
    st.markdown("**Pipeline**")
    st.markdown(
        "1. **Manager Agent** — routes your query to relevant expert domains\n"
        "2. **Expert Agents** — retrieve case law from ChromaDB and analyse\n"
        "3. **QA Agent** — synthesises findings into a structured advisory"
    )
    st.markdown("**Expert Domains**")
    st.markdown(
        "- Drug Offences (MDA)\n"
        "- Sexual Offences\n"
        "- Violent Crimes\n"
        "- Property & Financial Crimes\n"
        "- Sentencing\n"
        "- Criminal Procedure\n"
        "- Regulatory Offences"
    )
    st.divider()
    st.caption("SUTD MLOPS Group 7 · Ollama + ChromaDB")

# ── Main ─────────────────────────────────────────────────────────────────────

st.title("Singapore Criminal Law Advisory System")
st.caption(
    "Enter a criminal law query to receive a structured legal advisory backed by Singapore case law."
)

query = st.text_area(
    "Legal Query",
    height=100,
    placeholder=(
        "e.g. What is the mandatory minimum sentence for trafficking 15g of heroin? "
        "What defences are available?\n\n"
        "e.g. My client is charged with rape under s 375 Penal Code. "
        "What sentencing framework applies?"
    ),
    label_visibility="collapsed",
)

disable_run = (not query.strip()) or (not ollama_model.strip())

run_btn = st.button(
    "Get Legal Advisory",
    type="primary",
    disabled=disable_run,
    use_container_width=True,
)

# ── Pipeline execution ────────────────────────────────────────────────────────

if run_btn:
    if not ollama_available():
        st.error("Ollama is not running. Start it with `ollama serve` and try again.")
        st.stop()

    # Clear any previous results
    for key in ("manager_output", "qa_output"):
        st.session_state.pop(key, None)

    label_backend = f"Ollama ({ollama_model})"

    with st.status(f"Manager Agent routing query to experts ({label_backend})...", expanded=True) as status:
        try:
            manager_output = run_manager_agent(
                user_query=query,
                backend="ollama",
                ollama_model=ollama_model,
            )
        except Exception as e:
            status.update(label="Manager Agent failed", state="error")
            st.error(f"Manager Agent error: {e}")
            st.stop()

        experts = manager_output.get("experts_consulted", [])
        st.write(f"Experts consulted: {', '.join(experts) if experts else 'None'}")

        status.update(label=f"QA Agent synthesising findings ({label_backend})...")
        try:
            qa_output = run_qa_agent(
                user_query=query,
                expert_results=manager_output.get("expert_results", []),
                backend="ollama",
                ollama_model=ollama_model,
            )
        except Exception as e:
            status.update(label="QA Agent failed", state="error")
            st.error(f"QA Agent error: {e}")
            st.stop()

        status.update(label="Advisory ready", state="complete", expanded=False)

    st.session_state["manager_output"] = manager_output
    st.session_state["qa_output"] = qa_output

# ── Results ───────────────────────────────────────────────────────────────────

if "qa_output" in st.session_state and "manager_output" in st.session_state:
    qa_output = st.session_state["qa_output"]
    manager_output = st.session_state["manager_output"]
    experts = manager_output.get("experts_consulted", [])

    st.divider()

    col_class, col_experts = st.columns([3, 2])
    with col_class:
        st.subheader("Case Classification")
        st.info(qa_output.get("classification") or "See advisory for classification.")
    with col_experts:
        st.subheader("Experts Consulted")
        for name in experts:
            st.markdown(f"- {name}")

    st.divider()

    tab_advisory, tab_experts = st.tabs(["Final Advisory", "Expert Findings"])

    with tab_advisory:
        st.markdown(qa_output.get("advisory", ""))

        if qa_output.get("citations"):
            st.divider()
            st.subheader("Cases Referenced")
            cols = st.columns(2)
            for i, citation in enumerate(qa_output["citations"]):
                cols[i % 2].markdown(f"`{citation}`")

    with tab_experts:
        for result in manager_output.get("expert_results", []):
            expert_name = result.get("expert_name", "Unknown Expert")
            chunks_retrieved = result.get("chunks_retrieved", "?")
            with st.expander(f"{expert_name} ({chunks_retrieved} chunks retrieved)"):
                st.markdown(result.get("findings", ""))
                if result.get("citations"):
                    st.divider()
                    st.caption("Cases retrieved: " + "  |  ".join(result["citations"]))