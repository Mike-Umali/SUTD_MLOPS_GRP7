"""
Testing Guide: Measure Ollama Effectiveness

This guide shows you how to objectively test and compare improvements across:
1. Model quality (accuracy, completeness)
2. Response quality (structure, citations)
3. Performance (speed, memory)
4. Cost/benefit analysis
"""

import json
from pathlib import Path


# ==============================================================================
# APPROACH 1: AUTOMATED MODEL COMPARISON
# ==============================================================================

"""
FASTEST WAY TO TEST (5 minutes)

1. Open PowerShell and run:
   
   cd c:\Users\jesly\OneDrive\SUTD_MLOPS_GRP7
   python compare_models.py --models mistral neural-chat llama3.1:8b
   
2. This tests all 3 models on 6 legal queries
3. Scores each response (0-10) on: completeness, structure, citations, length
4. Outputs a summary table showing which model is best

EXAMPLE OUTPUT:
   Model                  Avg Score         Status
   ─────────────────────────────────────────
   orca-mini:13b              8.5            ✅
   mistral:7b                 7.9            ✅
   neural-chat:7b             7.2            ⚠️
   llama3.1:8b                5.1            ❌

This tells you: orca-mini is best, but mistral is acceptable and faster.
"""


# ==============================================================================
# APPROACH 2: BEFORE/AFTER COMPARISON
# ==============================================================================

"""
COMPARE CURRENT SETUP TO OLD SETUP (10 minutes)

Step 1: Create a test query log
─────────────────────────────────

Create test_queries.json:

[
  {
    "query": "What is the maximum sentence for drug trafficking of 15g of heroin?",
    "domain": "drug_offences",
    "expected_elements": ["15g", "max sentence", "sentencing", "MDA s 5"]
  },
  {
    "query": "What are defences to rape under Penal Code s 375?",
    "domain": "sexual_offences",
    "expected_elements": ["consent", "defences", "s 375", "Penal Code"]
  },
  {
    "query": "How do I appeal a criminal conviction?",
    "domain": "criminal_procedure",
    "expected_elements": ["appeal", "CPC", "High Court", "14 days"]
  }
]

Step 2: Run test and save results
──────────────────────────────────

python test_effectiveness.py \
    --model mistral \
    --queries test_queries.json \
    --output results_mistral.json

python test_effectiveness.py \
    --model llama3.1:8b \
    --queries test_queries.json \
    --output results_llama.json

Step 3: Compare
───────────────

python analyze_results.py \
    --baseline results_llama.json \
    --improved results_mistral.json

OUTPUT:
   Metric          llama3.1:8b    mistral:7b    Improvement
   ─────────────────────────────────────────────────────
   Avg Response    189 words       287 words     +52%
   Contains Cases  40%             85%           +45%
   Contains Stats  60%             94%           +34%
   Structure Score 3.2/10          7.8/10        +143%
   Speed (avg)     8.2s            4.1s          2x faster
   Overall Score   4.1/10          7.7/10        +87%

This shows concrete improvements across all metrics.
"""


# ==============================================================================
# APPROACH 3: QUALITY EVALUATION CHECKLIST
# ==============================================================================

"""
MANUAL TESTING (evaluate each response) — 5-10 minutes per query

For each query, score the response on:

COMPLETENESS (Does it fully answer the question?)
  ☐ Addresses main question
  ☐ Mentions relevant statutes/sections
  ☐ References case law/precedents
  ☐ Explains practical application
  
STRUCTURE (Is it well-organized?)
  ☐ Clear sections/headings
  ☐ Logical flow
  ☐ Easy to follow
  ☐ Professional tone
  
ACCURACY (Is the law correct?)
  ☐ Statutes cited are correct
  ☐ Cases are real/relevant
  ☐ No hallucinated cases/laws
  ☐ Reasoning is sound
  
CITATIONS (Are sources properly cited?)
  ☐ Case names + year (e.g., "ABC v Public Prosecutor [2020]")
  ☐ Statute sections (e.g., "Penal Code s 300")
  ☐ Correct legal framework
  ☐ Cross-references other relevant law

SCORING:
  Each checkbox = 1 point
  Total: 12 points
  
  Score 10-12: ⭐⭐⭐⭐⭐ Excellent
  Score 8-9:   ⭐⭐⭐⭐ Good
  Score 6-7:   ⭐⭐⭐ Acceptable
  Score 4-5:   ⭐⭐ Poor
  Score <4:    ⭐ Very Poor

TEST QUERIES (use these):
  1. "What is the maximum sentence for drug trafficking of 15g heroin?"
  2. "What are the key elements of rape under Penal Code s 375?"
  3. "What defences are available for a sexual assault charge?"
  4. "What is the sentencing framework for culpable homicide?"
  5. "How do I appeal a criminal conviction?"
  6. "What is criminal revision under CPC s 400?"
"""


# ==============================================================================
# APPROACH 4: PERFORMANCE METRICS
# ==============================================================================

"""
MEASURE SPEED & RESOURCE USAGE

1. RESPONSE TIME
   ──────────────
   Run in PowerShell:
   
   $start = Get-Date
   streamlit run app.py
   # Wait for load, then query a legal question
   $end = Get-Date
   ($end - $start).TotalSeconds
   
   Expected times:
   - mistral:7b:      2-5 seconds
   - orca-mini:13b:   4-8 seconds
   - llama3.1:8b:     3-6 seconds
   
   If significantly slower, you might have memory issues.

2. MEMORY USAGE
   ────────────
   Open Task Manager while Ollama is running:
   
   Process        Memory (MB)    Notes
   ──────────────────────────────────────
   ollama.exe     6000-8000      mistral:7b
   ollama.exe     8000-12000     orca-mini:13b
   python.exe     1000-2000      Streamlit
   
   Total: Should be <16GB for most systems.
   If exceeds your system RAM, use smaller model.

3. TOKEN THROUGHPUT
   ────────────────
   For a 2000-token expert response that took 5 seconds:
   
   Throughput = 2000 tokens / 5 sec = 400 tokens/sec
   
   Good range: 300-600 tokens/sec
   If <200 tokens/sec, system is struggling (check VRAM)

4. COST ANALYSIS
   ──────────────
   Ollama (free, local):
   - Electricity: ~0.5 kWh per query = $0.05 per query
   - Hardware: One-time cost ($500-$2000)
   
   Claude API (paid):
   - ~$0.05-0.10 per query
   - No hardware cost
   
   BREAKEVEN: Ollama breaks even after ~100-200 queries
"""


# ==============================================================================
# APPROACH 5: ACCURACY VERIFICATION
# ==============================================================================

"""
FACT-CHECK THE RESPONSES (Critical!)

For each response, verify:

1. STATUTE REFERENCES
   Query: "Is Penal Code s 375 for rape?"
   ✓ Correct: Yes, rape is defined in PC s 375
   ✗ Hallucination: Model cites "Penal Code s 999" (doesn't exist)

2. CASE CITATIONS
   Query: Does "Suvendu Prasad Chatterjee v Public Prosecutor [2020]" exist?
   ✓ Correct: Yes, drug trafficking sentencing case
   ✗ Hallucination: Model cites "Raj v Public Prosecutor [2025]" (fake/future)

3. SENTENCING BENCHMARKS
   Query: "Is 15g of heroin a death penalty case?"
   Your knowledge: No, death penalty is >15g in some circumstances
   ✓ Correct response
   ✗ Wrong response: Model says all heroin trafficking = death

4. LEGAL PRINCIPLES
   Query: "Can consent be a defence to rape?"
   Your knowledge: Yes, lack of consent is an element
   ✓ Correct: "Consent is the key defence to rape"
   ✗ Wrong: "No defences exist for rape"

HALLUCINATION CHECK:
If you catch the model citing fake cases/laws, that's a MAJOR issue:
→ Return to Claude only (Ollama hallucinating)
→ Or wait for fine-tuning to fix domain knowledge
"""


# ==============================================================================
# APPROACH 6: COMPARISON TABLE TEMPLATE
# ==============================================================================

"""
CREATE A COMPARISON SPREADSHEET

Model            | Speed | Memory | Quality | Accuracy | Cost  | Verdict
─────────────────┴───────┴────────┴─────────┴──────────┴───────┴────────
Claude (online)  | ⚡⚡⚡ | N/A    | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$    | BEST
                 | Fast  | Cloud  | Excellent| Excellent| Costly|
─────────────────┼───────┼────────┼─────────┼──────────┼───────┼────────
orca-mini:13b    | 🔥    | 10GB   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐  | FREE  | IDEAL
                 | Slow  | High   | Excellent| Very Good| Local |
─────────────────┼───────┼────────┼─────────┼──────────┼───────┼────────
mistral:7b       | 🔥🔥  | 6GB    | ⭐⭐⭐⭐  | ⭐⭐⭐   | FREE  | GOOD
                 | Medium| Normal | Good    | Good     | Local |
─────────────────┼───────┼────────┼─────────┼──────────┼───────┼────────
llama3.1:8b      | 🔥🔥  | 7GB    | ⭐⭐⭐   | ⭐⭐    | FREE  | OK
                 | Medium| Normal | Acceptable|Fair   | Local |

FILL THIS IN with YOUR actual test results.
"""


# ==============================================================================
# APPROACH 7: THRESHOLD TESTING
# ==============================================================================

"""
TEST DIFFERENT TOKEN LIMITS

Goal: Find the sweet spot between quality and speed

Test these on the same query:

Token Limit | Speed | Completeness | Score
────────────┼───────┼──────────────┼──────
500         | ⚡⚡⚡ | ⭐⭐        | 4/10
1000        | ⚡⚡  | ⭐⭐⭐      | 6/10
2000        | 🔥   | ⭐⭐⭐⭐    | 8/10  ← Current setting
4000        | 🔥🔥  | ⭐⭐⭐⭐⭐  | 9/10
8000        | 🔥🔥🔥 | ⭐⭐⭐⭐⭐  | 9.5/10 (diminishing returns)

At 2000 tokens, you get 80% of the quality with 50% of the time.
At 4000 tokens, you get 90% of the quality with 75% of the time.

Decision: 2000 is "good enough", 4000 is "optimal"
"""


# ==============================================================================
# APPROACH 8: A/B TESTING WITH USERS
# ==============================================================================

"""
GET FEEDBACK FROM ACTUAL USERS

Create a quick survey after each query:

"How helpful was this response?"
☐ Very helpful (saved time)
☐ Helpful (got the answer)
☐ Somewhat helpful (needed Claude to verify)
☐ Not helpful (inaccurate)

"Would you use Ollama for this type of query?"
☐ Always (good enough for drafts)
☐ Sometimes (depends on complexity)
☐ Never (need Claude API)

"What was missing from the response?"
☐ Case citations
☐ Statutory references
☐ Practical advice
☐ Structure/clarity
☐ Nothing - it was complete

After 20-30 queries, you'll have clear data on effectiveness.
"""

print(__doc__)
