"""
generate_qa.py — Generate Q&A fine-tuning pairs from Singapore criminal law cases.

Reads each case PDF + dataset.csv metadata, calls Claude Haiku to produce
instruction-following Q&A pairs grounded in real case facts and holdings.

Output: data/qa_pairs.jsonl   (~500-800 pairs, chat format for Qwen/Unsloth)

Usage:
    python generate_qa.py                  # 150 cases, 4 pairs each (~600 pairs)
    python generate_qa.py --cases 200      # more cases
    python generate_qa.py --resume         # skip already-processed citations
"""

import argparse
import json
import os
import re
import time

import anthropic
import pandas as pd
import pdfplumber

OUTPUT_PATH = "data/qa_pairs.jsonl"
CASES_DIR = "cases"
MODEL = "claude-haiku-4-5-20251001"  # cheapest, fast enough for generation

SYSTEM_PROMPT = (
    "You are an expert Singapore criminal law advisor with deep knowledge of "
    "the Penal Code, Misuse of Drugs Act, Criminal Procedure Code 2010, "
    "and Singapore appellate case law."
)

GENERATION_PROMPT = """\
You are producing training data for a fine-tuned Singapore criminal law AI assistant.

Given the court judgment below and its metadata, generate exactly {n_pairs} \
diverse Q&A pairs that a defence lawyer or law student might ask.

METADATA:
  Citation      : {citation}
  Area of law   : {area_of_law}
  Topic         : {topic}
  Subtopic      : {subtopic}
  Primary statute: {primary_statute}
  Catchwords    : {catchwords}

JUDGMENT EXCERPT:
{case_text}

Coverage requirements — spread your {n_pairs} pairs across different aspects:
  • The specific charge(s) and statutory provisions invoked
  • Key legal test or principle applied by the court
  • Outcome / sentence and the factors that drove it
  • Any defence raised, procedural point, or evidential issue

Rules:
  - Ground every answer strictly in the excerpt above; do not fabricate facts.
  - Cite the case ({citation}) and specific section numbers where relevant.
  - Each answer should be 120–280 words — thorough but not padded.
  - Questions must be self-contained (no "in this case" without naming the case).

Respond with ONLY a valid JSON array — no commentary before or after:
[
  {{"question": "...", "answer": "..."}},
  ...
]"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str, max_chars: int = 5000) -> str:
    """Extract the first ~5000 chars from a case PDF (covers intro + key analysis)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages[:7]:
                t = page.extract_text()
                if t:
                    text += t + "\n"
                if len(text) >= max_chars:
                    break
        return text[:max_chars].strip()
    except Exception:
        return ""


def sample_cases(df: pd.DataFrame, n_cases: int) -> pd.DataFrame:
    """
    One row per unique case, sampled proportionally across area_of_law buckets.
    Uses the first catchword row for each case (richest metadata).
    """
    unique = df.drop_duplicates(subset="filename").reset_index(drop=True)
    total = len(unique)

    sampled_parts = []
    for area, group in unique.groupby("area_of_law"):
        quota = max(1, round(n_cases * len(group) / total))
        sampled_parts.append(group.sample(min(quota, len(group)), random_state=42))

    sampled = pd.concat(sampled_parts).drop_duplicates(subset="filename")

    # top-up or trim to exact n_cases
    if len(sampled) < n_cases:
        remaining = unique[~unique["filename"].isin(sampled["filename"])]
        extra = remaining.sample(min(n_cases - len(sampled), len(remaining)), random_state=42)
        sampled = pd.concat([sampled, extra])

    return sampled.head(n_cases).reset_index(drop=True)


def build_catchwords(df: pd.DataFrame, filename: str) -> str:
    """Collect all catchwords for a case (multiple rows in CSV)."""
    rows = df[df["filename"] == filename]["catchword"].dropna().tolist()
    return " | ".join(rows[:5])  # cap at 5 to avoid prompt bloat


def call_claude(client: anthropic.Anthropic, prompt: str) -> list:
    """Call Claude Haiku and parse the JSON array response."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Be lenient: extract first [...] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []

    try:
        pairs = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    return [p for p in pairs if isinstance(p, dict) and "question" in p and "answer" in p]


def format_entry(pair: dict, row: pd.Series) -> dict:
    """Wrap a Q&A pair in chat format + metadata."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ],
        "metadata": {
            "citation":    row["citation"],
            "area_of_law": row["area_of_law"],
            "topic":       row["topic"],
            "subtopic":    row["subtopic"],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Q&A fine-tuning data from SG case law")
    parser.add_argument("--cases",  type=int, default=150, help="Number of cases to process (default 150)")
    parser.add_argument("--pairs",  type=int, default=4,   help="Q&A pairs per case (default 4)")
    parser.add_argument("--resume", action="store_true",   help="Append to existing file, skip done citations")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY environment variable not set.")
    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv("dataset.csv")
    sampled = sample_cases(df, args.cases)

    print(f"Sampled {len(sampled)} cases across {sampled['area_of_law'].nunique()} domains:")
    print(sampled["area_of_law"].value_counts().to_string())
    print(f"\nTarget: ~{len(sampled) * args.pairs} Q&A pairs  →  {OUTPUT_PATH}\n")

    os.makedirs("data", exist_ok=True)

    # Load already-done citations when resuming
    done = set()
    if args.resume and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done.add(entry["metadata"]["citation"])
                except Exception:
                    pass
        print(f"Resuming — {len(done)} citations already processed.\n")

    total_pairs = 0
    errors = 0

    mode = "a" if args.resume else "w"
    with open(OUTPUT_PATH, mode) as out:
        for i, row in sampled.iterrows():
            citation = row["citation"]
            filename = row["filename"]

            if citation in done:
                continue

            pdf_path = os.path.join(CASES_DIR, filename)
            if not os.path.exists(pdf_path):
                print(f"  [{i+1}/{len(sampled)}] SKIP (no PDF): {citation}")
                continue

            case_text = extract_text(pdf_path)
            if len(case_text) < 300:
                print(f"  [{i+1}/{len(sampled)}] SKIP (too short): {citation}")
                continue

            catchwords = build_catchwords(df, filename)
            prompt = GENERATION_PROMPT.format(
                citation=citation,
                area_of_law=row["area_of_law"],
                topic=row["topic"],
                subtopic=row["subtopic"],
                primary_statute=row["primary_statute"],
                catchwords=catchwords,
                case_text=case_text,
                n_pairs=args.pairs,
            )

            print(f"  [{i+1}/{len(sampled)}] {citation}  ({row['area_of_law']} / {row['subtopic']})")

            try:
                pairs = call_claude(client, prompt)
                for pair in pairs:
                    out.write(json.dumps(format_entry(pair, row)) + "\n")
                total_pairs += len(pairs)
                print(f"             → {len(pairs)} pairs written  (running total: {total_pairs})")
            except Exception as e:
                print(f"             → ERROR: {e}")
                errors += 1

            time.sleep(0.3)  # polite rate limiting

    print(f"\n{'='*50}")
    print(f"Done.  {total_pairs} pairs written to {OUTPUT_PATH}")
    if errors:
        print(f"Errors: {errors} cases failed — run with --resume to retry.")

    # Quick stats on output
    if os.path.exists(OUTPUT_PATH):
        from collections import Counter
        area_counts = Counter()
        total = 0
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    area_counts[e["metadata"]["area_of_law"]] += 1
                    total += 1
                except Exception:
                    pass
        print(f"\nFinal dataset: {total} pairs")
        for area, count in area_counts.most_common():
            print(f"  {area:<30} {count}")


if __name__ == "__main__":
    main()
