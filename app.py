"""
Streamlit app — Singapore Criminal Law Advisory System.
Multi-agent RAG pipeline: Manager → Expert Agents → QA Agent.
Supports Claude (online) and Ollama (local/offline) backends.
"""

import os
import traceback
import streamlit as st

from pipeline.agents.manager import run_manager_agent
from pipeline.agents.qa import run_qa_agent

st.set_page_config(
    page_title="SG Criminal Law Advisory",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")

    backend = st.radio(
        "Backend",
        options=["GPU (Local Model)", "Ollama (local)", "Claude (online)"],
        index=0,
        help=(
            "GPU: runs a HuggingFace model directly on CUDA (recommended on SUTD cluster). "
            "Ollama: local CPU/GPU inference via Ollama. "
            "Claude: Anthropic API (requires key)."
        ),
    )
    use_ollama = backend == "Ollama (local)"
    use_transformers = backend == "GPU (Local Model)"

    st.divider()

    if use_transformers:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    st.success(f"GPU detected: {gpu_name}")
                except Exception:
                    st.success("GPU detected (CUDA available)")
            else:
                st.warning("No CUDA GPU detected — model will run on CPU (slow).")
        except ImportError:
            st.warning("PyTorch not installed. Install with: pip install torch")

        ollama_model = st.text_input(
            "Model path or HuggingFace ID",
            value="Qwen/Qwen2.5-3B-Instruct",
            help="HuggingFace model ID (e.g. Qwen/Qwen2.5-3B-Instruct) or path to a local checkpoint.",
        )
        api_key = None
        client = None

    elif use_ollama:
        from pipeline.llm import ollama_available, list_ollama_models
        ollama_ok = ollama_available()

        if ollama_ok:
            st.success("Ollama is running")
            available_models = list_ollama_models()
            if available_models:
                ollama_model = st.selectbox(
                    "Model",
                    options=available_models,
                    help="Select a locally available Ollama model.",
                )
            else:
                st.warning("No models found. Run: `ollama pull qwen2.5:7b`")
                ollama_model = st.text_input("Model name", value="qwen2.5:7b")
        else:
            st.error("Ollama not reachable. Start it with: `ollama serve`")
            ollama_model = st.text_input("Model name", value="qwen2.5:7b")

        api_key = None
        client = None
    else:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="Your key is used only for this session and never stored.",
        )
        ollama_model = None
        client = None

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
    if use_transformers:
        st.caption("SUTD MLOPS Group 7 · GPU (Transformers) + ChromaDB")
    elif use_ollama:
        st.caption("SUTD MLOPS Group 7 · Ollama + ChromaDB")
    else:
        st.caption("SUTD MLOPS Group 7 · Claude + ChromaDB")

# ── Main ─────────────────────────────────────────────────────────────────────

st.title("Singapore Criminal Law Advisory System")
st.caption("Enter a criminal law query to receive a structured legal advisory backed by Singapore case law.")

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

run_btn = st.button(
    "Get Legal Advisory",
    type="primary",
    disabled=not query.strip(),
    use_container_width=True,
)

# ── Pipeline execution ────────────────────────────────────────────────────────

if run_btn:
    if use_transformers:
        active_backend = "transformers"
        label_backend = f"GPU ({ollama_model})"
    elif use_ollama:
        active_backend = "ollama"
        label_backend = f"Ollama ({ollama_model})"
    else:
        active_backend = "claude"
        label_backend = "Claude"
        if not api_key:
            st.error("Please enter your Anthropic API Key in the sidebar.")
            st.stop()
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    # Clear any previous results
    for key in ("manager_output", "qa_output"):
        st.session_state.pop(key, None)

    with st.status(f"Manager Agent routing query to experts ({label_backend})...", expanded=True) as status:
        try:
            manager_output = run_manager_agent(
                user_query=query,
                client=client,
                backend=active_backend,
                ollama_model=ollama_model or "qwen2.5:7b",
            )
        except Exception as e:
            status.update(label="Manager Agent failed", state="error")
            st.error(f"Manager Agent error: {e}")
            st.code(traceback.format_exc())
            st.stop()

        experts = manager_output["experts_consulted"]
        st.write(f"Experts consulted: {', '.join(experts)}")

        status.update(label=f"QA Agent synthesising findings ({label_backend})...")
        try:
            qa_output = run_qa_agent(
                user_query=query,
                expert_results=manager_output["expert_results"],
                client=client,
                backend=active_backend,
                ollama_model=ollama_model or "qwen2.5:7b",
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
    experts = manager_output["experts_consulted"]

    st.divider()

    col_class, col_experts = st.columns([3, 2])
    with col_class:
        st.subheader("Case Classification")
        st.info(qa_output["classification"] or "See advisory for classification.")
    with col_experts:
        st.subheader("Experts Consulted")
        for name in experts:
            st.markdown(f"- {name}")

    st.divider()

    tab_advisory, tab_experts = st.tabs(["Final Advisory", "Expert Findings"])

    with tab_advisory:
        st.markdown(qa_output["advisory"])

        if qa_output.get("citations"):
            st.divider()
            st.subheader("Cases Referenced")
            cols = st.columns(2)
            for i, citation in enumerate(qa_output["citations"]):
                cols[i % 2].markdown(f"`{citation}`")

    with tab_experts:
        for result in manager_output["expert_results"]:
            with st.expander(f"{result['expert_name']}  ({result.get('chunks_retrieved', '?')} chunks retrieved)"):
                st.markdown(result["findings"])
                if result.get("citations"):
                    st.divider()
                    st.caption("Cases retrieved: " + "  |  ".join(result["citations"]))
