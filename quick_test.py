#!/usr/bin/env python
"""
Quick Test Script — Run this NOW to measure current Ollama performance

This generates test results that prove whether your optimization worked.

Usage:
    python quick_test.py --model mistral
    python quick_test.py --model llama3.1:8b
    python quick_test.py --models mistral llama3.1:8b (compare)
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path


# Test queries covering all domains
TEST_QUERIES = [
    {
        "name": "Drug Trafficking Sentencing",
        "domain": "drug_offences",
        "query": "What is the maximum sentence for drug trafficking of 15 grams of heroin under Singapore law?",
        "must_have": ["15", "heroin", "sentence", "MDA", "Misuse"]
    },
    {
        "name": "Rape Definition & Elements",
        "domain": "sexual_offences",
        "query": "What are the key elements that must be proven to convict someone of rape under Penal Code Section 375?",
        "must_have": ["consent", "375", "Penal Code", "penetration"]
    },
    {
        "name": "Criminal Appeal Process",
        "domain": "criminal_procedure",
        "query": "How do I appeal a criminal conviction to the High Court and what are the time limits?",
        "must_have": ["appeal", "14 days", "High Court", "grounds"]
    },
    {
        "name": "Cheating Defense",
        "domain": "property_financial",
        "query": "What is cheating under Singapore criminal law and what elements must be proven?",
        "must_have": ["cheating", "417", "deception", "intent"]
    },
    {
        "name": "Culpable Homicide Sentencing",
        "domain": "sentencing",
        "query": "What is the sentencing framework for culpable homicide (not amounting to murder)?",
        "must_have": ["sentencing", "rash", "negligent", "culpable"]
    },
    {
        "name": "Grievous Hurt Defense",
        "domain": "violent_crimes",
        "query": "What defences are available for voluntarily causing grievous hurt (Penal Code Section 325)?",
        "must_have": ["defence", "325", "grievous", "hurt"]
    },
]


class OllamaTestRunner:
    def __init__(self, model: str):
        self.model = model
        self.results = []
        self.timestamps = []
        
    def test_query(self, test_query: dict) -> dict:
        """Run a single query and measure effectiveness"""
        try:
            import ollama
            
            system_prompt = (
                f"You are a {test_query['domain']} expert in Singapore criminal law. "
                f"Answer concisely but comprehensively, citing relevant statutes and cases."
            )
            
            start_time = time.time()
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test_query["query"]}
                ],
                stream=False,
            )
            elapsed = time.time() - start_time
            
            answer = response["message"]["content"]
            
            # Score the response
            score = self._score_response(answer, test_query)
            
            return {
                "name": test_query["name"],
                "domain": test_query["domain"],
                "status": "success",
                "answer": answer,
                "time_seconds": round(elapsed, 2),
                "word_count": len(answer.split()),
                "score": score,
                "has_citations": bool("(" in answer and ")" in answer),
                "has_statute": any(x in answer.lower() for x in ["s ", "section", "code"]),
                "contains_required": all(x.lower() in answer.lower() for x in test_query["must_have"]),
            }
        except Exception as e:
            return {
                "name": test_query["name"],
                "domain": test_query["domain"],
                "status": "failed",
                "error": str(e),
                "time_seconds": 0,
                "word_count": 0,
                "score": 0,
            }
    
    def _score_response(self, answer: str, test_query: dict) -> int:
        """Score response 0-10"""
        score = 0
        answer_lower = answer.lower()
        
        # Length (1-2 points)
        if len(answer.split()) > 100:
            score += 2
        elif len(answer.split()) > 50:
            score += 1
        
        # Has citations (2 points)
        if "(" in answer and ")" in answer:
            score += 2
        
        # Has statutes (2 points)
        if any(x in answer_lower for x in ["section", "s ", "act", "code"]):
            score += 2
        
        # Structure (1 point)
        if any(x in answer for x in ["\n", "[", "•", "-", "1.", "2.", "3."]):
            score += 1
        
        # Contains required terms (2 points)
        required_count = sum(1 for x in test_query["must_have"] if x.lower() in answer_lower)
        if required_count == len(test_query["must_have"]):
            score += 2
        elif required_count >= len(test_query["must_have"]) - 1:
            score += 1
        
        return min(10, score)
    
    def run_all_tests(self) -> None:
        """Run all test queries"""
        print(f"\n{'='*70}")
        print(f"Model: {self.model}")
        print(f"{'='*70}\n")
        
        for i, test_q in enumerate(TEST_QUERIES, 1):
            print(f"[{i}/{len(TEST_QUERIES)}] {test_q['name']}...", end=" ", flush=True)
            
            result = self.test_query(test_q)
            self.results.append(result)
            
            if result["status"] == "success":
                print(f"✓ Score: {result['score']}/10")
            else:
                print(f"✗ Error: {result['error'][:40]}")
        
        self.print_results()
    
    def print_results(self) -> None:
        """Print detailed results"""
        print(f"\n{'='*70}")
        print(f"DETAILED RESULTS — {self.model}")
        print(f"{'='*70}\n")
        
        successful = [r for r in self.results if r["status"] == "success"]
        
        if not successful:
            print("❌ All tests failed!")
            return
        
        # Summary statistics
        avg_score = sum(r["score"] for r in successful) / len(successful)
        avg_time = sum(r["time_seconds"] for r in successful) / len(successful)
        avg_words = sum(r["word_count"] for r in successful) / len(successful)
        has_citations = sum(1 for r in successful if r["has_citations"]) / len(successful) * 100
        has_statute = sum(1 for r in successful if r["has_statute"]) / len(successful) * 100
        has_required = sum(1 for r in successful if r["contains_required"]) / len(successful) * 100
        
        print(f"📊 SUMMARY STATISTICS")
        print(f"{'-'*70}")
        print(f"Average Score:        {avg_score:>6.1f}/10.0")
        print(f"Average Response Time: {avg_time:>6.2f}s")
        print(f"Average Word Count:    {avg_words:>6.0f} words")
        print(f"Contains Citations:    {has_citations:>6.0f}%")
        print(f"Has Statute/Code Ref:  {has_statute:>6.0f}%")
        print(f"Has All Required Terms:{has_required:>6.0f}%")
        print(f"{'-'*70}\n")
        
        # Detailed per-query
        print(f"📋 PER-QUERY BREAKDOWN\n")
        for r in successful:
            print(f"  {r['name']}")
            print(f"    Score: {r['score']}/10 | Time: {r['time_seconds']:.1f}s | Words: {r['word_count']}")
            print(f"    Citations: {'✓' if r['has_citations'] else '✗'} | Statute: {'✓' if r['has_statute'] else '✗'} | Complete: {'✓' if r['contains_required'] else '✗'}")
            print()
        
        # Rating
        print(f"{'='*70}")
        if avg_score >= 8:
            rating = "✅ EXCELLENT — Ready for production"
        elif avg_score >= 7:
            rating = "✅ GOOD — Usable for most queries"
        elif avg_score >= 5:
            rating = "⚠️  ACCEPTABLE — Works but needs improvement"
        else:
            rating = "❌ POOR — Upgrade model or increase tokens"
        print(f"OVERALL RATING: {rating}")
        print(f"{'='*70}\n")
        
        # Save results
        output_file = f"test_results_{self.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump({
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "avg_score": avg_score,
                    "avg_time": avg_time,
                    "avg_words": avg_words,
                    "has_citations_pct": has_citations,
                    "has_statute_pct": has_statute,
                    "has_required_pct": has_required,
                },
                "results": successful,
            }, f, indent=2)
        print(f"✓ Results saved to: {output_file}\n")


def compare_models(models: list) -> None:
    """Compare multiple models side-by-side"""
    all_results = {}
    
    for model in models:
        runner = OllamaTestRunner(model)
        runner.run_all_tests()
        all_results[model] = runner.results
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<20} {'Avg Score':>12} {'Avg Time':>12} {'Citations':>12}")
    print(f"{'-'*70}")
    
    for model in models:
        successful = [r for r in all_results[model] if r["status"] == "success"]
        if successful:
            avg_score = sum(r["score"] for r in successful) / len(successful)
            avg_time = sum(r["time_seconds"] for r in successful) / len(successful)
            citations_pct = sum(1 for r in successful if r["has_citations"]) / len(successful) * 100
            
            rating = "✅" if avg_score >= 7 else "⚠️ " if avg_score >= 5 else "❌"
            print(f"{model:<20} {avg_score:>11.1f} {rating:>1} {avg_time:>11.1f}s {citations_pct:>10.0f}%")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quick test of Ollama model effectiveness"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Single model to test",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Multiple models to compare",
    )
    
    args = parser.parse_args()
    
    print("\n🚀 Ollama Effectiveness Test")
    print("════════════════════════════════════════════════════════════")
    
    if args.models:
        compare_models(args.models)
    elif args.model:
        runner = OllamaTestRunner(args.model)
        runner.run_all_tests()
    else:
        print("Usage:")
        print("  Single model:   python quick_test.py --model mistral")
        print("  Compare models: python quick_test.py --models mistral neural-chat llama3.1:8b")


if __name__ == "__main__":
    main()
