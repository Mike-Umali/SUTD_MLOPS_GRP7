"""
Model Comparison Tool — Test different Ollama models on the same queries

This script helps you objectively compare output quality between different models.
Use it to decide which model is best for your criminal law advisory system.

Usage:
    python compare_models.py --models mistral neural-chat orca-mini
"""

import json
import time
import argparse
from pathlib import Path


def load_test_queries(filepath: str = None) -> list:
    """Load test queries from file or use defaults."""
    if filepath and Path(filepath).exists():
        with open(filepath, "r") as f:
            return json.load(f)
    
    # Default test queries
    return [
        {
            "domain": "drug_offences",
            "query": "What is the maximum sentence for trafficking 20 grams of heroin?",
        },
        {
            "domain": "sexual_offences",
            "query": "What are the key elements that must be proven to convict someone of rape?",
        },
        {
            "domain": "violent_crimes",
            "query": "What defences are available for a charge of voluntarily causing grievous hurt?",
        },
        {
            "domain": "property_financial",
            "query": "What constitutes cheating under Singapore law and what are the penalties?",
        },
        {
            "domain": "sentencing",
            "query": "What factors does the court consider when sentencing for drug trafficking?",
        },
        {
            "domain": "criminal_procedure",
            "query": "What is the difference between criminal appeal and criminal revision?",
        },
    ]


def test_model(model_name: str, query: str, domain: str, max_tokens: int = 1000) -> dict:
    """Test a model on a single query."""
    try:
        import ollama
        
        system_prompt = (
            f"You are a {domain} expert in Singapore criminal law. "
            f"Answer the following query concisely and cite relevant statutes and cases."
        )
        
        start_time = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            stream=False,
        )
        elapsed = time.time() - start_time
        
        answer = response["message"]["content"]
        word_count = len(answer.split())
        
        return {
            "model": model_name,
            "domain": domain,
            "query": query,
            "answer": answer,
            "response_time": round(elapsed, 2),
            "word_count": word_count,
            "status": "success",
        }
    except Exception as e:
        return {
            "model": model_name,
            "domain": domain,
            "query": query,
            "answer": None,
            "error": str(e),
            "status": "failed",
        }


def score_response(response: dict) -> dict:
    """
    Score a response based on quality metrics.
    Scores are out of 10.
    """
    if response["status"] == "failed":
        return {
            "completeness": 0,
            "structure": 0,
            "citations": 0,
            "length": 0,
            "overall": 0,
        }
    
    answer = response["answer"].lower()
    word_count = response["word_count"]
    
    # Completeness (does it attempt to answer?)
    completeness = 5 if word_count > 50 else 0
    completeness += 5 if any(x in answer for x in ["statute", "section", "act"]) else 0
    
    # Structure (does it have markers of structure?)
    structure = 5 if any(x in answer for x in ["[", ":", "•", "-"]) else 3
    
    # Citations (does it cite cases and statutes?)
    cite_score = 0
    if answer.count("(") > 0:
        cite_score += 3
    if "v" in answer or "public prosecutor" in answer or "penal code" in answer:
        cite_score += 4
    if "section" in answer or "s " in answer:
        cite_score += 3
    citations = min(10, cite_score)
    
    # Length (is it substantial but not excessive?)
    if 100 < word_count < 400:
        length = 10
    elif 50 < word_count < 500:
        length = 8
    elif word_count > 30:
        length = 5
    else:
        length = 2
    
    overall = (completeness + structure + citations + length) / 4
    
    return {
        "completeness": round(completeness, 1),
        "structure": round(structure, 1),
        "citations": round(citations, 1),
        "length": round(length, 1),
        "overall": round(overall, 1),
    }


def print_result(result: dict, scores: dict) -> None:
    """Pretty print comparison result."""
    print(f"\n{'='*80}")
    print(f"Model: {result['model']}")
    print(f"Domain: {result['domain']}")
    print(f"Query: {result['query'][:60]}...")
    print(f"{'='*80}")
    
    if result['status'] == 'failed':
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        return
    
    print(f"\n📊 SCORES (out of 10):")
    print(f"  Completeness:  {scores['completeness']:>5.1f} (does it try to answer?)")
    print(f"  Structure:     {scores['structure']:>5.1f} (is it organized?)")
    print(f"  Citations:     {scores['citations']:>5.1f} (cases/statutes cited?)")
    print(f"  Length:        {scores['length']:>5.1f} (substantive enough?)")
    print(f"  {'─'*30}")
    print(f"  OVERALL:       {scores['overall']:>5.1f} ⭐")
    
    print(f"\n⏱️  Response time: {result['response_time']}s")
    print(f"📝 Word count: {result['word_count']}")
    
    print(f"\n💬 RESPONSE:\n")
    print(result['answer'][:500])
    if len(result['answer']) > 500:
        print(f"\n... ({result['word_count']} words total)")


def compare_models(models: list, queries: list = None):
    """Compare multiple models on the same queries."""
    if not queries:
        queries = load_test_queries()
    
    results = []
    
    print(f"🚀 Testing {len(models)} models on {len(queries)} queries...\n")
    
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Testing model: {model}")
        model_results = []
        
        for j, test_q in enumerate(queries):
            print(f"  [{j+1}/{len(queries)}] {test_q['domain']}...", end=" ", flush=True)
            result = test_model(
                model_name=model,
                query=test_q["query"],
                domain=test_q["domain"],
            )
            results.append(result)
            model_results.append(result)
            
            if result['status'] == 'success':
                print(f"✓ ({result['word_count']} words, {result['response_time']}s)")
            else:
                print(f"✗ {result.get('error', 'Error')}")
    
    # Print detailed comparison
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON")
    print(f"{'='*80}")
    
    for result in results:
        scores = score_response(result)
        print_result(result, scores)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    summary_data = {}
    for result in results:
        model = result['model']
        if model not in summary_data:
            summary_data[model] = []
        
        scores = score_response(result)
        summary_data[model].append(scores['overall'])
    
    print(f"{'Model':<20} {'Avg Score':>12} {'Status':>15}")
    print(f"{'-'*48}")
    for model, scores in sorted(summary_data.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
        avg = sum(scores) / len(scores) if scores else 0
        status = "✅" if avg > 7 else "⚠️ " if avg > 5 else "❌"
        print(f"{model:<20} {avg:>12.1f} {status:>15}")
    
    print(f"\n✅ = Excellent (7+), ⚠️  = Good (5-7), ❌ = Poor (<5)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Ollama models on criminal law queries"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mistral", "neural-chat", "llama3.1:8b"],
        help="Models to compare (space-separated)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to JSON file with test queries",
    )
    
    args = parser.parse_args()
    
    print("🔍 Ollama Model Comparison Tool")
    print("════════════════════════════════════════════════════════════\n")
    
    compare_models(args.models, queries=load_test_queries(args.queries))
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
