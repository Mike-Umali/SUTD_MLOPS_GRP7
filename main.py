"""
Main entry point for the Singapore Criminal Law Agentic RAG Pipeline.

Usage:
  python main.py --index                                  # Build ChromaDB index from PDFs (run once)
  python main.py --query "..."                           # Run agentic pipeline on a query
  python main.py --query "..." --model llama3.1:8b       # Run with a specific Ollama model
  python main.py                                         # Interactive mode
"""

import argparse

from pipeline.index import build_index
from pipeline.agents.manager import run_manager_agent
from pipeline.agents.qa import run_qa_agent

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


def run_pipeline(query: str, model: str = DEFAULT_OLLAMA_MODEL) -> dict:
    """Full agentic pipeline: manager → experts → QA using Ollama."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"Model: {model}")
    print(f"{'=' * 60}\n")

    print("[1/2] Manager Agent routing query to experts...")
    manager_output = run_manager_agent(
        user_query=query,
        backend="ollama",
        ollama_model=model,
    )

    experts_consulted = manager_output.get("experts_consulted", [])
    print(f"\nExperts consulted: {', '.join(experts_consulted)}")

    print("\n[2/2] QA Agent synthesizing findings...")
    qa_output = run_qa_agent(
        user_query=query,
        expert_results=manager_output["expert_results"],
        backend="ollama",
        ollama_model=model,
    )

    print(f"\n{'=' * 60}")
    print("FINAL ADVISORY")
    print(f"{'=' * 60}\n")
    print(qa_output["advisory"])
    print(f"\n{'=' * 60}")
    print(f"Case Classification: {qa_output.get('classification', '')}")
    print(f"Citations: {', '.join(qa_output.get('citations', []))}")
    print(f"{'=' * 60}\n")

    return qa_output


def main():
    parser = argparse.ArgumentParser(description="Singapore Criminal Law Agentic RAG Pipeline")
    parser.add_argument("--index", action="store_true", help="Build ChromaDB index from PDFs")
    parser.add_argument("--query", type=str, help="Legal query to run through the pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    args = parser.parse_args()

    if args.index:
        print("Building ChromaDB index from case PDFs...")
        build_index()
        return

    if args.query:
        run_pipeline(args.query, model=args.model)
    else:
        # Interactive mode
        print("Singapore Criminal Law Advisory System (Ollama)")
        print("Type 'exit' to quit.\n")
        while True:
            query = input("Enter your legal query: ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if query:
                run_pipeline(query, model=args.model)


if __name__ == "__main__":
    main()