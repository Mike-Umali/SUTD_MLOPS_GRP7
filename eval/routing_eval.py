"""
Routing evaluation — requires Anthropic API.

Measures how accurately the Manager Agent selects the correct expert domains.
Metrics: precision, recall, F1 (set-based per query), macro-averaged overall.
"""

from __future__ import annotations
import os
import sys
import statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import anthropic
from pipeline.agents.manager import run_manager_agent
from eval.test_set import TEST_CASES, TestCase

# Maps expert_name strings (returned by run_manager_agent) → domain keys
EXPERT_NAME_TO_DOMAIN: Dict[str, str] = {
    "Drug Offences Expert":                 "drug_offences",
    "Sexual Offences Expert":               "sexual_offences",
    "Violent Crimes Expert":                "violent_crimes",
    "Property and Financial Crimes Expert": "property_financial",
    "Sentencing Expert":                    "sentencing",
    "Criminal Procedure Expert":            "criminal_procedure",
    "Regulatory Offences Expert":           "regulatory",
}


@dataclass
class RoutingResult:
    tc_id: str
    query: str
    expected_domains: List[str]
    actual_domains: List[str]
    precision: float
    recall: float
    f1: float
    exact_match: bool
    error: str = ""


def _prf(expected: List[str], actual: List[str]) -> Tuple[float, float, float]:
    exp_set = set(expected)
    act_set = set(actual)
    if not act_set:
        return 0.0, 0.0, 0.0
    tp = len(exp_set & act_set)
    precision = tp / len(act_set)
    recall = tp / len(exp_set) if exp_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def evaluate_routing(
    client: anthropic.Anthropic,
    test_cases: List[TestCase] = None,
) -> List[RoutingResult]:
    if test_cases is None:
        test_cases = TEST_CASES

    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {tc.id}: {tc.query[:60]}...")
        try:
            manager_output = run_manager_agent(tc.query, client)
            actual_names = manager_output["experts_consulted"]
            actual_domains = [
                EXPERT_NAME_TO_DOMAIN.get(name, name) for name in actual_names
            ]
            p, r, f1 = _prf(tc.expected_domains, actual_domains)
            results.append(RoutingResult(
                tc_id=tc.id,
                query=tc.query,
                expected_domains=tc.expected_domains,
                actual_domains=actual_domains,
                precision=p,
                recall=r,
                f1=f1,
                exact_match=(set(tc.expected_domains) == set(actual_domains)),
            ))
        except Exception as e:
            print(f"    [ERROR] {tc.id}: {e}")
            results.append(RoutingResult(
                tc_id=tc.id,
                query=tc.query,
                expected_domains=tc.expected_domains,
                actual_domains=[],
                precision=0.0,
                recall=0.0,
                f1=0.0,
                exact_match=False,
                error=str(e),
            ))

    return results


def compute_routing_report(results: List[RoutingResult]) -> dict:
    valid = [r for r in results if not r.error]
    if not valid:
        return {"error": "No valid results"}

    macro_p = round(statistics.mean(r.precision for r in valid), 3)
    macro_r = round(statistics.mean(r.recall for r in valid), 3)
    macro_f1 = round(statistics.mean(r.f1 for r in valid), 3)
    exact_rate = round(sum(r.exact_match for r in valid) / len(valid), 3)

    return {
        "n_evaluated": len(results),
        "n_valid": len(valid),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "exact_match_rate": exact_rate,
        "n_exact": sum(r.exact_match for r in valid),
        "results": results,
    }


def print_routing_report(report: dict) -> None:
    print("\nROUTING EVALUATION  (requires Anthropic API)")
    print("-" * 60)
    print(f"  Test cases evaluated : {report['n_evaluated']}")
    print(f"  Macro Precision      : {report['macro_precision']:.3f}")
    print(f"  Macro Recall         : {report['macro_recall']:.3f}")
    print(f"  Macro F1             : {report['macro_f1']:.3f}")
    print(f"  Exact match rate     : {report['exact_match_rate']:.3f}  ({report['n_exact']}/{report['n_valid']})")
    print()
    print(f"  {'ID':<8} {'P':>5} {'R':>5} {'F1':>5} {'EM':>4}  Expected → Actual")
    print("  " + "─" * 85)
    for r in report["results"]:
        if r.error:
            print(f"  {r.tc_id:<8} {'ERR':>5}  {r.error[:50]}")
        else:
            em = "Y" if r.exact_match else "N"
            exp = ",".join(r.expected_domains)
            act = ",".join(r.actual_domains)
            print(f"  {r.tc_id:<8} {r.precision:>5.2f} {r.recall:>5.2f} {r.f1:>5.2f} {em:>4}  {exp} → {act}")
