"""
Retrieval evaluation — no Anthropic API required.

Measures:
  - Hit rate: does a relevant subtopic appear in the top-5 retrieved chunks?
  - Avg relevance score: mean cosine similarity of the top-5 results
  - Per-domain breakdown
"""

from __future__ import annotations
import os
import sys
import statistics
from dataclasses import dataclass
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.index import get_client, get_embedding_fn, get_collection
from eval.test_set import TEST_CASES, TestCase


@dataclass
class RetrievalResult:
    tc_id: str
    query: str
    domain: str
    hit: bool
    top_subtopics: List[str]
    scores: List[float]
    avg_score: float
    n_retrieved: int


def evaluate_retrieval(test_cases: List[TestCase] = None, n_results: int = 5) -> List[RetrievalResult]:
    if test_cases is None:
        test_cases = TEST_CASES

    # Initialise once — avoids re-loading ONNX model for each query
    client = get_client()
    embedding_fn = get_embedding_fn()
    collection_cache: Dict[str, object] = {}

    results = []
    for tc in test_cases:
        domain = tc.domain_for_retrieval

        if domain not in collection_cache:
            collection_cache[domain] = get_collection(domain, client, embedding_fn)
        col = collection_cache[domain]

        count = col.count()
        if count == 0:
            print(f"  [WARN] {tc.id}: collection '{domain}' is empty — skipping")
            continue

        raw = col.query(
            query_texts=[tc.query],
            n_results=min(n_results, count),
            include=["metadatas", "distances"],
        )

        if not raw["metadatas"] or not raw["metadatas"][0]:
            chunks = []
        else:
            chunks = list(zip(raw["metadatas"][0], raw["distances"][0]))

        top_subtopics = [m.get("subtopic", "") for m, _ in chunks]
        scores = [round(1 - d, 3) for _, d in chunks]
        # Use substring matching — ChromaDB metadata sometimes stores compound
        # subtopic strings (e.g. "General; General sexual offences"), so exact
        # set membership fails. A hit = any relevant subtopic appears within
        # any returned subtopic string.
        hit = any(
            rel in st
            for st in top_subtopics
            for rel in tc.relevant_subtopics
        )
        avg_score = round(statistics.mean(scores), 3) if scores else 0.0

        results.append(RetrievalResult(
            tc_id=tc.id,
            query=tc.query,
            domain=domain,
            hit=hit,
            top_subtopics=top_subtopics,
            scores=scores,
            avg_score=avg_score,
            n_retrieved=len(chunks),
        ))

    return results


def compute_retrieval_report(results: List[RetrievalResult]) -> dict:
    if not results:
        return {}

    overall_hit_rate = round(sum(r.hit for r in results) / len(results), 3)
    overall_avg_score = round(statistics.mean(r.avg_score for r in results), 3)

    by_domain: Dict[str, dict] = {}
    for domain in sorted(set(r.domain for r in results)):
        dr = [r for r in results if r.domain == domain]
        by_domain[domain] = {
            "hit_rate": round(sum(r.hit for r in dr) / len(dr), 3),
            "avg_score": round(statistics.mean(r.avg_score for r in dr), 3),
            "n_cases": len(dr),
        }

    return {
        "overall_hit_rate": overall_hit_rate,
        "overall_avg_score": overall_avg_score,
        "n_evaluated": len(results),
        "n_hits": sum(r.hit for r in results),
        "by_domain": by_domain,
        "results": results,
    }


def print_retrieval_report(report: dict) -> None:
    print("\nRETRIEVAL EVALUATION  (no API required)")
    print("-" * 60)
    print(f"  Test cases evaluated : {report['n_evaluated']}")
    print(f"  Overall hit rate     : {report['overall_hit_rate']:.3f}  ({report['n_hits']}/{report['n_evaluated']})")
    print(f"  Overall avg score    : {report['overall_avg_score']:.3f}  (mean cosine similarity, top-5)")
    print()
    print(f"  {'Domain':<25} {'Cases':>5}   {'Hit Rate':>8}   {'Avg Score':>9}")
    print("  " + "─" * 55)
    for domain, stats in report["by_domain"].items():
        print(f"  {domain:<25} {stats['n_cases']:>5}   {stats['hit_rate']:>8.3f}   {stats['avg_score']:>9.3f}")

    print()
    print(f"  {'ID':<8} {'Domain':<22} {'Hit':<5} {'Avg Score':<10} Top Subtopics Retrieved")
    print("  " + "─" * 80)
    for r in report["results"]:
        hit_str = "YES" if r.hit else "NO "
        subtopics_str = " | ".join(s for s in r.top_subtopics if s)[:45]
        print(f"  {r.tc_id:<8} {r.domain:<22} {hit_str:<5} {r.avg_score:<10.3f} {subtopics_str}")
