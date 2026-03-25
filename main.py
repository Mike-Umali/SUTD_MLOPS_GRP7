"""
Main entry point for the Singapore Criminal Law Agentic RAG Pipeline.

Usage:
  python main.py --index          # Build ChromaDB index from PDFs (run once)
  python main.py --query "..."    # Run agentic pipeline on a query
  python main.py                  # Interactive mode
"""

import argparse
import os
import anthropic

from pipeline.index import build_index
from pipeline.agents.manager import run_manager_agent
from pipeline.agents.qa import run_qa_agent


def run_pipeline(query: str, client: anthropic.Anthropic) -> dict:
    """Full agentic pipeline: manager → experts → QA."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    print("[1/2] Manager Agent routing query to experts...")
    manager_output = run_manager_agent(query, client)

    experts_consulted = manager_output["experts_consulted"]
    print(f"\nExperts consulted: {', '.join(experts_consulted)}")

    print("\n[2/2] QA Agent synthesizing findings...")
    qa_output = run_qa_agent(
        user_query=query,
        expert_results=manager_output["expert_results"],
        client=client,
    )

    print(f"\n{'='*60}")
    print("FINAL ADVISORY")
    print(f"{'='*60}\n")
    print(qa_output["advisory"])
    print(f"\n{'='*60}")
    print(f"Case Classification: {qa_output['classification']}")
    print(f"Citations: {', '.join(qa_output['citations'])}")
    print(f"{'='*60}\n")

    return qa_output


def main():
    parser = argparse.ArgumentParser(description="Singapore Criminal Law Agentic RAG Pipeline")
    parser.add_argument("--index", action="store_true", help="Build ChromaDB index from PDFs")
    parser.add_argument("--query", type=str, help="Legal query to run through the pipeline")
    args = parser.parse_args()

    if args.index:
        print("Building ChromaDB index from case PDFs...")
        build_index()
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    if args.query:
        run_pipeline(args.query, client)
    else:
        # Interactive mode
        print("Singapore Criminal Law Advisory System")
        print("Type 'exit' to quit.\n")
        while True:
            query = input("Enter your legal query: ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if query:
                run_pipeline(query, client)


if __name__ == "__main__":
    main()
