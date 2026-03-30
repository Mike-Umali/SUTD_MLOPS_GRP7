"""
Master evaluation runner.

Usage:
  python eval/run_eval.py --retrieval
  python eval/run_eval.py --routing
  python eval/run_eval.py --advisory
  python eval/run_eval.py --all
"""

from __future__ import annotations
import argparse
import os
import sys
from datetime import date

# Ensure project root is on path and CWD is project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from eval.test_set import TEST_CASES


def _get_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def run_retrieval():
    from eval.retrieval_eval import evaluate_retrieval, compute_retrieval_report, print_retrieval_report
    print("Running retrieval evaluation...")
    results = evaluate_retrieval(TEST_CASES)
    report = compute_retrieval_report(results)
    print_retrieval_report(report)
    return report


def run_routing(client):
    from eval.routing_eval import evaluate_routing, compute_routing_report, print_routing_report
    print("Running routing evaluation (this will make API calls)...")
    results = evaluate_routing(client, TEST_CASES)
    report = compute_routing_report(results)
    print_routing_report(report)
    return report


def run_advisory(client):
    from eval.advisory_eval import evaluate_advisory, compute_advisory_report, print_advisory_report
    print("Running advisory evaluation (full pipeline + judge, ~10 queries)...")
    results = evaluate_advisory(client)
    report = compute_advisory_report(results)
    print_advisory_report(report)
    return report


def print_header():
    print("=" * 60)
    print("  SINGAPORE CRIMINAL LAW RAG — EVALUATION REPORT")
    print(f"  Run date: {date.today()}")
    print("=" * 60)


def print_summary(reports: dict):
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if "retrieval" in reports:
        r = reports["retrieval"]
        print(f"  Retrieval hit rate   : {r['overall_hit_rate']:.3f}  ({r['n_hits']}/{r['n_evaluated']})")
        print(f"  Retrieval avg score  : {r['overall_avg_score']:.3f}")
    if "routing" in reports:
        r = reports["routing"]
        print(f"  Routing macro-F1     : {r['macro_f1']:.3f}")
        print(f"  Routing exact match  : {r['exact_match_rate']:.3f}  ({r['n_exact']}/{r['n_valid']})")
    if "advisory" in reports:
        r = reports["advisory"]
        print(f"  Advisory avg score   : {r['overall_avg_score']:.2f} / 5.00")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Singapore Criminal Law RAG — Evaluation Runner")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval evaluation (no API)")
    parser.add_argument("--routing",   action="store_true", help="Run routing evaluation (requires API)")
    parser.add_argument("--advisory",  action="store_true", help="Run advisory evaluation (requires API)")
    parser.add_argument("--all",       action="store_true", help="Run all evaluations")
    args = parser.parse_args()

    if not any([args.retrieval, args.routing, args.advisory, args.all]):
        parser.print_help()
        sys.exit(0)

    print_header()
    reports = {}
    client = None

    needs_api = args.routing or args.advisory or args.all
    if needs_api:
        client = _get_client()

    if args.retrieval or args.all:
        reports["retrieval"] = run_retrieval()

    if args.routing or args.all:
        reports["routing"] = run_routing(client)

    if args.advisory or args.all:
        reports["advisory"] = run_advisory(client)

    if len(reports) > 1:
        print_summary(reports)


if __name__ == "__main__":
    main()
