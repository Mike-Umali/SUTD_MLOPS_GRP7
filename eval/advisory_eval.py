"""
Advisory evaluation.

Runs the full pipeline on a 10-query subset and uses Claude as judge
to score each advisory on 5 dimensions (each 1-5).

The pipeline backend (generation) is configurable: claude, ollama, or transformers.
The judge always uses Claude (Anthropic API) for consistent scoring.

Dimensions:
  1. legal_accuracy    — correct statutes, sections, case law
  2. completeness      — all key legal issues addressed
  3. citation_quality  — specific, on-point, properly formatted citations
  4. format_compliance — all 5 advisory sections present and structured
  5. actionability     — RECOMMENDED NEXT STEPS are concrete and practical
"""

from __future__ import annotations
import os
import sys
import json
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import anthropic
from pipeline.agents.manager import run_manager_agent
from pipeline.agents.qa import run_qa_agent
from eval.test_set import TEST_CASES, TestCase

MODEL = "claude-sonnet-4-6"

# 10-query subset — one per domain + two cross-domain cases
ADVISORY_SUBSET_IDS = [
    "TC-01",  # drug_offences (death penalty / s 33B)
    "TC-04",  # sexual_offences (rape sentencing bands)
    "TC-06",  # sexual_offences + criminal_procedure (consent / nonverbal)
    "TC-07",  # violent_crimes (murder vs culpable homicide)
    "TC-10",  # property_financial (CBT s 409)
    "TC-12",  # sentencing (totality principle)
    "TC-14",  # criminal_procedure (bail pending appeal)
    "TC-15",  # criminal_procedure (admissibility / cautioned statement)
    "TC-18",  # regulatory (drink driving)
    "TC-03",  # drug_offences + sentencing (cross-domain diamorphine trafficking)
]

JUDGE_SYSTEM = """You are an expert evaluator of Singapore criminal law legal advisories. You have deep knowledge of Singapore criminal law, the Penal Code, Misuse of Drugs Act, Criminal Procedure Code 2010, Evidence Act, and leading Singapore case law from the High Court and Court of Appeal.

Your task is to score an AI-generated legal advisory on five dimensions using a 1-5 scale."""

JUDGE_USER_TEMPLATE = """Evaluate the following AI-generated legal advisory for a Singapore criminal law query.

ORIGINAL QUERY:
{query}

AI ADVISORY:
{advisory}

Score this advisory on EACH of the following five dimensions (1-5):

1 = Very poor / largely incorrect or missing
2 = Below expectations / significant gaps
3 = Adequate / meets basic requirements
4 = Good / minor gaps only
5 = Excellent / no significant issues

DIMENSIONS:

1. LEGAL_ACCURACY (1-5)
   Are statutes, section numbers, and legal principles correctly stated?
   Are case citations attributed to the correct legal propositions?
   Deduct for wrong section numbers, misattributed cases, or legal errors.

2. COMPLETENESS (1-5)
   Are all key legal issues raised by the query addressed?
   Are elements of offence, available defences, and sentencing factors covered?
   Deduct for significant omissions a Singapore practitioner would expect.

3. CITATION_QUALITY (1-5)
   Are citations in proper Singapore neutral citation format (e.g. [2024] SGHC 123)?
   Are cited cases relevant and on-point for the propositions they support?
   Deduct for made-up citations, irrelevant cases, or no citations at all.

4. FORMAT_COMPLIANCE (1-5)
   Does the advisory include all required sections:
   CASE CLASSIFICATION, LEGAL ISSUES IDENTIFIED, APPLICABLE LAW,
   ANALYSIS, RECOMMENDED NEXT STEPS, CASES REFERENCED?
   Deduct for missing or mislabelled sections.

5. ACTIONABILITY (1-5)
   Are RECOMMENDED NEXT STEPS numbered, prioritised, and immediately actionable?
   Do they specify what to do procedurally, what arguments to raise, what defences to consider?
   Deduct for vague generalities ("consult a lawyer", "consider the facts").

Output ONLY a JSON object with no other text:
{{
  "legal_accuracy": <int 1-5>,
  "completeness": <int 1-5>,
  "citation_quality": <int 1-5>,
  "format_compliance": <int 1-5>,
  "actionability": <int 1-5>,
  "reasoning": {{
    "legal_accuracy": "<one sentence>",
    "completeness": "<one sentence>",
    "citation_quality": "<one sentence>",
    "format_compliance": "<one sentence>",
    "actionability": "<one sentence>"
  }}
}}"""

DIMENSIONS = ["legal_accuracy", "completeness", "citation_quality", "format_compliance", "actionability"]


@dataclass
class AdvisoryResult:
    tc_id: str
    query: str
    advisory: str
    classification: str
    citations: List[str]
    experts_consulted: List[str]
    scores: Dict[str, int] = field(default_factory=dict)
    reasoning: Dict[str, str] = field(default_factory=dict)
    total_score: float = 0.0
    avg_score: float = 0.0
    error: str = ""


def _parse_judge(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def evaluate_advisory(
    client: anthropic.Anthropic,
    test_cases: List[TestCase] = None,
    subset_ids: List[str] = None,
    backend: str = "claude",
    model: str = "qwen2.5:7b",
) -> List[AdvisoryResult]:
    if test_cases is None:
        test_cases = TEST_CASES
    if subset_ids is None:
        subset_ids = ADVISORY_SUBSET_IDS

    subset = [tc for tc in test_cases if tc.id in subset_ids]
    results = []

    for i, tc in enumerate(subset, 1):
        print(f"  [{i}/{len(subset)}] {tc.id}: running pipeline ({backend})...")
        try:
            pipeline_client = client if backend == "claude" else None
            manager_output = run_manager_agent(
                tc.query, pipeline_client, backend=backend, ollama_model=model
            )
            qa_output = run_qa_agent(
                user_query=tc.query,
                expert_results=manager_output["expert_results"],
                client=pipeline_client,
                backend=backend,
                ollama_model=model,
            )

            advisory = qa_output["advisory"]
            print(f"    Advisory generated ({len(advisory)} chars). Running judge...")

            judge_response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=JUDGE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        query=tc.query,
                        advisory=advisory,
                    ),
                }],
            )
            judge_text = judge_response.content[0].text
            parsed = _parse_judge(judge_text)

            if parsed:
                scores = {d: int(parsed.get(d, 0)) for d in DIMENSIONS}
                reasoning = parsed.get("reasoning", {})
                total = sum(scores.values())
                avg = round(total / len(DIMENSIONS), 2)
            else:
                print(f"    [WARN] {tc.id}: could not parse judge response")
                scores = {}
                reasoning = {}
                total = 0.0
                avg = 0.0

            results.append(AdvisoryResult(
                tc_id=tc.id,
                query=tc.query,
                advisory=advisory,
                classification=qa_output.get("classification", ""),
                citations=qa_output.get("citations", []),
                experts_consulted=manager_output["experts_consulted"],
                scores=scores,
                reasoning=reasoning,
                total_score=total,
                avg_score=avg,
            ))

        except Exception as e:
            print(f"    [ERROR] {tc.id}: {e}")
            results.append(AdvisoryResult(
                tc_id=tc.id,
                query=tc.query,
                advisory="",
                classification="",
                citations=[],
                experts_consulted=[],
                error=str(e),
            ))

    return results


def compute_advisory_report(results: List[AdvisoryResult]) -> dict:
    valid = [r for r in results if r.scores]
    if not valid:
        return {"error": "No valid results"}

    dim_avgs = {}
    dim_mins = {}
    dim_maxs = {}
    for d in DIMENSIONS:
        vals = [r.scores[d] for r in valid]
        dim_avgs[d] = round(statistics.mean(vals), 2)
        dim_mins[d] = min(vals)
        dim_maxs[d] = max(vals)

    overall_avg = round(statistics.mean(r.avg_score for r in valid), 2)

    return {
        "n_evaluated": len(results),
        "n_valid": len(valid),
        "overall_avg_score": overall_avg,
        "dimension_avgs": dim_avgs,
        "dimension_mins": dim_mins,
        "dimension_maxs": dim_maxs,
        "results": results,
    }


def print_advisory_report(report: dict) -> None:
    print("\nADVISORY EVALUATION  (requires Anthropic API, LLM-as-judge)")
    print("-" * 60)
    print(f"  Queries evaluated    : {report['n_evaluated']}")
    print(f"  Overall avg score    : {report['overall_avg_score']:.2f} / 5.00")
    print()
    print(f"  {'Dimension':<22} {'Avg':>5}   {'Min':>4}   {'Max':>4}")
    print("  " + "─" * 42)
    for d in DIMENSIONS:
        print(f"  {d:<22} {report['dimension_avgs'][d]:>5.2f}   {report['dimension_mins'][d]:>4}   {report['dimension_maxs'][d]:>4}")

    print()
    print(f"  {'ID':<8} {'Acc':>4} {'Com':>4} {'Cit':>4} {'Fmt':>4} {'Act':>4}  {'Avg':>5}  Classification")
    print("  " + "─" * 80)
    for r in report["results"]:
        if r.error:
            print(f"  {r.tc_id:<8} ERROR: {r.error[:50]}")
        elif not r.scores:
            print(f"  {r.tc_id:<8} (judge parse failed)")
        else:
            s = r.scores
            clf = r.classification[:35] if r.classification else ""
            print(
                f"  {r.tc_id:<8} {s['legal_accuracy']:>4} {s['completeness']:>4} "
                f"{s['citation_quality']:>4} {s['format_compliance']:>4} "
                f"{s['actionability']:>4}  {r.avg_score:>5.2f}  {clf}"
            )
