# SUTD MLOPS Group 7 — Singapore Criminal Law Agentic RAG Pipeline

An end-to-end MLOps pipeline that scrapes Singapore criminal law judgments, builds a labeled dataset, indexes them into a vector store, and answers legal queries through a multi-agent RAG system powered by Claude.

---

## System Architecture

```
                        ┌─────────────────────────────────────┐
                        │           PART 1: DATA PIPELINE     │
                        └─────────────────────────────────────┘

  eLitigation (SUPCT)
         │
         ▼
  collect_links()  ──── 3,500 URLs (7x buffer for ~85% civil cases)
         │
         ▼
  scrape_case()  ──── title, citation, catchwords per case
         │
         ▼
  is_criminal_case()  ──── filter: only criminal cases pass
         │
         ▼
  download_pdf()  ──── saves to cases/ (gitignored)
         │
         ▼
  classify_catchword()  ──── taxonomy longest-prefix match
         │
         ▼
  dataset.csv  ──── 500 cases, 1,029 rows, 11 columns


                        ┌─────────────────────────────────────┐
                        │       PART 2: AGENTIC RAG PIPELINE  │
                        └─────────────────────────────────────┘

  dataset.csv + cases/
         │
         ▼
  extract.py  ──── pdfplumber extraction + 800-char chunks (150 overlap)
         │
         ▼
  index.py  ──── ONNXMiniLM_L6_V2 embeddings → ChromaDB (8 collections)
         │
         ▼
  User Query (CLI or Streamlit)
         │
         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    MANAGER AGENT (Claude)                    │
  │  Analyses query, selects relevant expert domains via tool_use│
  │  Always routes to: Sentencing + Criminal Procedure           │
  └──────────────┬───────────────────────────────────────────────┘
                 │  dispatches to (in parallel)
        ┌────────┼──────────────────────────────┐
        ▼        ▼        ▼        ▼            ▼
  [Drug]  [Sexual]  [Violent]  [Property]  [Regulatory]
  Expert   Expert    Expert     Expert      Expert
        \       \      |       /           /
         \       \     |      /           /
          ▼       ▼    ▼     ▼           ▼
         ChromaDB retrieval per domain (cosine similarity)
                        │
                        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                      QA AGENT (Claude)                       │
  │  Synthesises all expert findings into a structured advisory  │
  │  Written in Singapore High Court judgment style              │
  └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
  Final Advisory: Classification | Legal Issues | Applicable Law |
                  Analysis | Next Steps | Cases Referenced
                        │
                        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                   STREAMLIT WEB APP (app.py)                 │
  │  Interactive UI — live pipeline status, tabbed results       │
  └──────────────────────────────────────────────────────────────┘
```

---

## Part 1: Data Pipeline

### 1.1 Scraper (`scraper.py`)

Scrapes Singapore Supreme Court criminal judgments from eLitigation.

**Key design decisions:**
- Collects 3,500 candidate URLs (7x buffer) — ~85% of SUPCT cases are civil
- Checks `is_criminal_case()` **before** downloading PDFs (saves bandwidth and storage)
- Stops exactly at 500 confirmed criminal cases
- Each row includes: `filename`, `case_name`, `citation`, `catchword`, `area_of_law`, `topic`, `subtopic`, `primary_statute`, `is_criminal`, `taxonomy_key`, `pdf_url`

### 1.2 Taxonomy (`taxonomy.py`)

A handcrafted Singapore criminal law taxonomy with **~280 entries**.

**Classification approach:**
- `classify_catchword(cw)` — longest-prefix match → `(area, topic, subtopic, statute, is_criminal, taxonomy_key)`
- Matching is case-insensitive and dash-normalized (handles em-dash, en-dash, double en-dash, spaced hyphens)
- `split_catchword(raw)` — fallback for unmatched catchwords
- `is_criminal_case(area)` — keyword-based criminal detection for pre-download filtering

**Coverage:**

| Area | Categories |
|---|---|
| Offences against person | Murder, Culpable homicide, Hurt, Assault, Kidnapping, Stalking |
| Sexual offences | Rape, SAP, Outrage of modesty, Voyeurism, Exploitation of minor |
| Property offences | Theft, Robbery, CBT, Cheating, House-breaking, Extortion |
| Forgery and fraud | Forgery, Using forged document |
| Public order | Rioting, Unlawful assembly, Affray |
| Offences against administration | Obstructing public servant, False evidence, Fabricating evidence |
| Statutory offences (MDA) | Drug trafficking, possession, consumption, importation |
| Statutory offences (other) | Arms, Corruption, Money laundering, Computer Misuse, Immigration, Road Traffic, Companies Act, WSH Act, Women's Charter |
| Elements of crime | Mens rea, Actus reus, Intention, Knowledge, Recklessness, Causation |
| Participation | Common intention, Abetment, Criminal conspiracy, Attempt |
| General exceptions | Private defence, Intoxication, Unsoundness of mind, Diminished responsibility, Provocation |
| Sentencing | Benchmarks, Bands, Mandatory minimum, Probation, CBS, Caning, Totality, Deterrence |
| Criminal Procedure | Arrest, Bail, Charge, Trial, Statements, Appeals, Review, Reference |
| Evidence | Hearsay, Expert evidence, Admissibility, Similar fact, Corroboration, Presumptions |
| Constitutional Law | Equal protection, Fundamental liberties, Judicial review, Accused person rights |

### 1.3 Dataset (`dataset.csv`)

**Shape:** 1,029 rows × 11 columns | 500 unique cases | Years: 2020–2026

| Column | Description |
|---|---|
| `filename` | PDF filename (case ID) |
| `case_name` | Case name from page title |
| `citation` | Neutral citation (e.g. `[2024] SGCA 13`) |
| `catchword` | Raw catchword string from eLitigation |
| `area_of_law` | Classified area (e.g. `Criminal Law`) |
| `topic` | Classified topic (e.g. `Statutory offences`) |
| `subtopic` | Classified subtopic (e.g. `Misuse of Drugs Act`) |
| `primary_statute` | Primary statute (e.g. `Misuse of Drugs Act s 5`) |
| `is_criminal` | Boolean — whether row was classified as criminal |
| `taxonomy_key` | Matched taxonomy key (blank if unmatched) |
| `pdf_url` | URL to the judgment PDF |

**Data quality fixes applied:**

| Issue | Fix |
|---|---|
| Dash variants (em, en, double en-dash) | Normalized in `_normalize_dashes()` |
| Case mismatch (`Criminal law` vs `Criminal Law`) | Case-insensitive prefix matching |
| 1 corrupt row (case text leaked into catchword) | Dropped |
| 38 null citations | Filled from filename |
| 21.5% taxonomy mismatch → 1.7% | Added ~100 missing taxonomy entries |

---

### 1.4 EDA — Label Distribution & Initial Analysis

#### Dataset overview

| Metric | Value |
|---|---|
| Total rows | 1,029 |
| Unique cases | 500 |
| Avg catchwords per case | 2.06 |
| Max catchwords per case | 11 |
| Taxonomy match rate | 98.3% |
| Null subtopic | 5 rows |
| Null primary_statute | 17 rows |

#### Court breakdown (unique cases)

| Court | Cases | Description |
|---|---|---|
| SGHC | 331 (66.2%) | High Court (General Division) |
| SGCA | 143 (28.6%) | Court of Appeal |
| SGHCF | 4 (0.8%) | High Court (Family Division) |
| Unknown | 22 (4.4%) | Citation format varies |

#### Cases by year

| Year | Unique Cases | Catchword Rows |
|---|---|---|
| 2020 | 21 | 44 |
| 2021 | 99 | 185 |
| 2022 | 91 | 163 |
| 2023 | 84 | 188 |
| 2024 | 85 | 176 |
| 2025 | 77 | 168 |
| 2026 | 21 | 64 |

> 2020 and 2026 are partial years in the dataset (scraper captured up to March 2026).

#### Area of law distribution (rows)

| Area of Law | Rows | % |
|---|---|---|
| Criminal Law | 648 | 63.1% |
| Criminal Procedure | 279 | 27.1% |
| Constitutional Law | 40 | 3.9% |
| Evidence | 23 | 2.2% |
| Statutory Interpretation | 12 | 1.2% |
| Administrative Law | 5 | 0.5% |
| Civil Procedure | 5 | 0.5% |
| Other | 17 | 1.7% |

#### Topic distribution (top 10)

| Topic | Rows | % |
|---|---|---|
| Sentencing | 253 | 24.6% |
| Statutory offences | 202 | 19.6% |
| General | 121 | 11.8% |
| Sexual offences | 83 | 8.1% |
| Offences against person | 39 | 3.8% |
| Review | 33 | 3.2% |
| Appeals | 30 | 2.9% |
| Statements | 24 | 2.3% |
| General exceptions / Defences | 21 | 2.0% |
| Property offences | 17 | 1.7% |

> **Sentencing** is the single largest topic (24.6%), reflecting that a large proportion of criminal appeals in Singapore turn on sentencing rather than conviction.

#### Subtopic distribution (top 15)

| Subtopic | Rows | % |
|---|---|---|
| General | 306 | 29.7% |
| Misuse of Drugs Act | 102 | 9.9% |
| Sentencing principles | 56 | 5.4% |
| Appeals against sentence | 47 | 4.6% |
| General sexual offences | 38 | 3.7% |
| Criminal review | 33 | 3.2% |
| Criminal appeal | 29 | 2.8% |
| Rape | 24 | 2.3% |
| Murder | 17 | 1.7% |
| Criminal reference | 17 | 1.7% |
| Property offence | 15 | 1.5% |
| Penal Code | 14 | 1.4% |
| Stay of execution | 14 | 1.4% |
| Prevention of Corruption Act | 13 | 1.3% |
| Outrage of modesty | 12 | 1.2% |

#### Primary statute distribution (top 10)

| Primary Statute | Rows | % |
|---|---|---|
| Criminal Procedure Code 2010 | 424 | 41.2% |
| Misuse of Drugs Act | 102 | 9.9% |
| Penal Code | 54 | 5.2% |
| Penal Code Pt XI (Sexual offences) | 38 | 3.7% |
| Criminal Procedure Code 2010 s 394H (Review) | 33 | 3.2% |
| Penal Code s 375 (Rape) | 24 | 2.3% |
| Evidence Act | 22 | 2.1% |
| Criminal Procedure Code 2010 s 397 (Reference) | 17 | 1.7% |
| Penal Code s 300 (Murder) | 17 | 1.7% |
| Prevention of Corruption Act | 14 | 1.4% |

#### is_criminal label distribution

| Label | Rows | Unique Cases |
|---|---|---|
| True (criminal) | 927 (90.1%) | 479 (95.8%) |
| False (non-criminal catchword) | 102 (9.9%) | 21 (4.2%) |

> Cases with `is_criminal=False` rows are mixed criminal/civil matters (e.g. constitutional challenges arising out of criminal proceedings, civil procedure questions in criminal appeals). The case itself is criminal — the individual catchword touches a non-criminal area.

#### Key observations

1. **Sentencing dominates** — 24.6% of all catchwords relate to sentencing. Singapore's criminal appellate practice heavily focuses on sentence calibration rather than conviction challenges.

2. **MDA is the top specific statute** — Misuse of Drugs Act cases make up ~10% of all catchwords, driven by Singapore's mandatory death penalty jurisprudence and the large volume of trafficking cases.

3. **Court of Appeal vs High Court split** — 28.6% of cases are Court of Appeal decisions, which tend to be high-value precedent-setting judgments on sentencing frameworks and legal principles.

4. **Sexual offences are well-represented** — 8.1% of rows, driven by evolving sentencing band frameworks (Pram Nair, GBR) that generate significant appellate activity.

5. **Taxonomy coverage** — After expanding the taxonomy from ~180 to ~280 entries and fixing case-insensitive matching, unmatched catchwords dropped from 21.5% to **1.7%** (17 rows out of 1,029).

---

## Part 2: Agentic RAG Pipeline

### 2.1 Text Extraction (`pipeline/extract.py`)

Extracts and chunks text from case PDFs for indexing.

- `extract_text(pdf_path)` — extracts full text using `pdfplumber`
- `chunk_text(text, chunk_size=800, overlap=150)` — sliding window chunking
- `iter_case_chunks(csv_path)` — yields one dict per chunk with full metadata
- `assign_domain(row)` — maps each case to one of 8 expert domains based on area/topic/subtopic

**8 Expert Domains:**

| Domain | Coverage |
|---|---|
| `drug_offences` | MDA trafficking, possession, consumption, importation |
| `sexual_offences` | Rape, SAP, outrage of modesty, voyeurism, exploitation |
| `violent_crimes` | Murder, culpable homicide, hurt, robbery, assault |
| `property_financial` | Theft, CBT, cheating, money laundering, corruption |
| `sentencing` | Sentencing principles, benchmarks, appeals, probation |
| `criminal_procedure` | Arrest, bail, charge, trial, appeals, evidence |
| `regulatory` | Road traffic, immigration, Companies Act, WSH Act, Women's Charter |
| `general` | Constitutional law, judicial review, elements of crime |

### 2.2 Vector Index (`pipeline/index.py`)

Builds and queries the ChromaDB vector store.

- **Embedding model:** `ONNXMiniLM_L6_V2` (ChromaDB built-in — no PyTorch dependency)
- **Storage:** Persistent ChromaDB at `chroma_db/` (gitignored)
- **Collections:** One per domain (8 total), cosine similarity metric
- **Indexing:** Batched upserts of 100 chunks, skips already-indexed IDs (safe to re-run)
- `retrieve(query, domain, n_results=5)` — returns top-n chunks with relevance scores
- `retrieve_multi_domain(query, domains, n_per_domain=3)` — queries multiple domains

**Index statistics (after full build):**

| Domain | Chunks |
|---|---|
| sentencing | ~2,200 |
| drug_offences | ~2,000 |
| criminal_procedure | ~1,500 |
| sexual_offences | ~1,400 |
| violent_crimes | ~500 |
| property_financial | ~300 |
| regulatory | ~100 |
| general | ~100 |
| **Total** | **~8,100** |

Build the index (run once):
```bash
python main.py --index
```

### 2.3 Manager Agent (`pipeline/agents/manager.py`)

Orchestrates the expert agents using Claude's native `tool_use`.

- Receives the user query and selects which expert domains to consult
- Each expert domain is registered as a Claude tool (`consult_<domain>`)
- Always routes to `sentencing` and `criminal_procedure` for any criminal query
- Runs an agentic loop until `stop_reason == "end_turn"` (all experts consulted)
- Returns `expert_results` list with each expert's findings and citations

### 2.4 Expert Agents (`pipeline/agents/experts.py`)

Seven specialist expert agents, one per domain.

Each expert agent:
1. Retrieves the top-5 most relevant chunks from its ChromaDB collection
2. Calls Claude with a domain-specific system prompt
3. Analyses the retrieved case law and returns structured findings + citations

**Expert profiles:**

| Expert | Specialisation |
|---|---|
| Drug Offences Expert | MDA, trafficking weights, presumptions, enhanced punishment |
| Sexual Offences Expert | Penal Code Part XI, sentencing bands, consent issues |
| Violent Crimes Expert | Homicide, grievous hurt, s 300 exceptions, provocation |
| Property & Financial Expert | CBT, cheating, money laundering, CDSA |
| Sentencing Expert | TIC charges, totality, proportionality, benchmark frameworks |
| Criminal Procedure Expert | CPC, Evidence Act, admissibility, confessions, appeals |
| Regulatory Expert | Road traffic, immigration, workplace safety, corporate offences |

### 2.5 QA Agent (`pipeline/agents/qa.py`)

Synthesises all expert findings into a final structured legal advisory.

Output format (Singapore High Court judgment style):

```
**CASE CLASSIFICATION**
e.g. "Drug Trafficking — Misuse of Drugs Act s 5"

**LEGAL ISSUES IDENTIFIED**
Key legal questions raised by the query.

**APPLICABLE LAW**
Relevant statutes, provisions, and leading cases.

**ANALYSIS**
Synthesised expert findings — coherent legal analysis noting any tensions.

**RECOMMENDED NEXT STEPS**
Numbered list of concrete, actionable next steps in priority order.

**CASES REFERENCED**
All case citations referenced in the analysis.
```

---

## Part 3: Streamlit Web App (`app.py`)

An interactive web interface for the full agentic pipeline.

### Features

- **API key input** in the sidebar (or set via `ANTHROPIC_API_KEY` env var)
- **Live pipeline status** — `st.status` shows each step as it runs (Manager routing → QA synthesis)
- **Case classification** and **experts consulted** displayed at a glance
- **Two result tabs:**
  - **Final Advisory** — full structured advisory with citations grid
  - **Expert Findings** — expandable per-expert analysis showing chunks retrieved and case citations
- **Session state** — results persist on re-interaction without re-running the pipeline

### Run the app

```bash
streamlit run app.py
```

Or with the API key pre-set:

```bash
ANTHROPIC_API_KEY=sk-ant-... streamlit run app.py
```

Opens at **http://localhost:8501**.

---

## Usage (CLI)

### Build the index (run once after scraping)
```bash
python main.py --index
```

### Run a single query
```bash
python main.py --query "What is the mandatory minimum sentence for trafficking 15g of heroin?"
```

### Interactive mode
```bash
python main.py
```

### Environment variable required
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Part 4: Evaluation Pipeline (`eval/`)

Three-layer evaluation covering retrieval quality, agent routing accuracy, and end-to-end advisory quality.

### Running the evaluation

```bash
# Retrieval only — no API key required, runs in ~30 seconds
python eval/run_eval.py --retrieval

# Routing accuracy — requires Anthropic API
python eval/run_eval.py --routing

# Advisory quality — requires Anthropic API, runs ~10 full pipeline queries
python eval/run_eval.py --advisory

# Everything
python eval/run_eval.py --all
```

### Layer 1 — Retrieval Evaluation (`eval/retrieval_eval.py`)

No API calls. Queries ChromaDB directly for each of 20 test cases and measures:

- **Hit rate** — does a relevant subtopic appear in the top-5 retrieved chunks?
- **Avg relevance score** — mean cosine similarity of the top-5 results (0–1, higher is better)

**Baseline results:**

| Domain | Hit Rate | Avg Score |
|---|---|---|
| drug_offences | 1.000 | 0.673 |
| sentencing | 1.000 | 0.648 |
| criminal_procedure | 0.750 | 0.654 |
| violent_crimes | 0.667 | 0.568 |
| sexual_offences | 0.667 | 0.543 |
| regulatory | 0.667 | 0.476 |
| property_financial | 0.500 | 0.479 |
| **Overall** | **0.750** | **0.583** |

> `property_financial` and `regulatory` are the weakest domains — directly motivating the plan to add 500 more cases targeting those areas.

### Layer 2 — Routing Evaluation (`eval/routing_eval.py`)

Runs the Manager Agent on 20 test queries and compares the actual domains routed to against hand-labelled ground truth.

Metrics (set-based, per query):
- **Precision** — of the domains the manager selected, what fraction were correct?
- **Recall** — of the correct domains, what fraction did the manager select?
- **F1** — harmonic mean of precision and recall
- **Exact match** — did the manager select exactly the right set of domains?

### Layer 3 — Advisory Evaluation (`eval/advisory_eval.py`)

Runs the full pipeline (Manager → Experts → QA Agent) on a 10-query subset, then uses **Claude as judge** to score each advisory.

#### How LLM-as-judge works

```
Query
  │
  ▼
Full pipeline runs  →  Manager → Experts → QA Agent
  │
  ▼
Advisory text produced  (~1,500 words)
  │
  ▼
New Claude call (the "judge"):
  Input:  original query + advisory
  Task:   score on 5 dimensions, output JSON only
  │
  ▼
{ "legal_accuracy": 4, "completeness": 5, "citation_quality": 3, ... }
```

The judge uses a deliberately adversarial rubric ("deduct marks for X") rather than a rewarding one, to counteract self-evaluation bias (Claude judging Claude's own output tends to be generous).

#### The 5 scoring dimensions (each 1–5)

| Dimension | What it checks |
|---|---|
| `legal_accuracy` | Correct statute sections, case law, legal principles — deducts for wrong section numbers or misattributed cases |
| `completeness` | All key legal issues from the query addressed — deducts for omissions a practitioner would expect |
| `citation_quality` | Citations in proper Singapore neutral citation format `[2024] SGHC 123`, relevant and on-point — deducts for hallucinated citations |
| `format_compliance` | All 6 advisory sections present and properly headed (Case Classification → Cases Referenced) |
| `actionability` | RECOMMENDED NEXT STEPS are numbered, prioritised, and specific — deducts for vague generalities |

#### Example judge output

```json
{
  "legal_accuracy": 4,
  "completeness": 5,
  "citation_quality": 3,
  "format_compliance": 5,
  "actionability": 4,
  "reasoning": {
    "legal_accuracy": "Correctly cited s 375(4) defence but omitted s 90 vitiating conditions.",
    "citation_quality": "Two citations appear fabricated — [2021] SGHC 999 does not exist.",
    "actionability": "Next steps are numbered and specific, covering voir dire and plea mitigation."
  }
}
```

The `reasoning` field identifies exactly why a score was low, making it actionable for improving the pipeline.

#### Known limitation

Claude judging its own output introduces self-evaluation bias — scores will tend to skew 3.5–4.5. For more rigorous evaluation, use a different model as judge (e.g. GPT-4o judging Claude's output) to eliminate this bias.

### Test Set (`eval/test_set.py`)

20 hand-crafted queries with ground truth covering all 7 expert domains, including cross-domain cases (e.g. drug trafficking + sentencing, sexual offences + criminal procedure).

Each test case includes:
- `expected_domains` — correct domains for routing eval
- `domain_for_retrieval` — which ChromaDB collection to test
- `relevant_subtopics` — exact subtopic strings from dataset metadata (for hit detection)
- `expected_keywords` — keywords that should appear in the final advisory

---

## File Structure

```
MLOPSproj/
├── scraper.py              # eLitigation scraper (500 criminal cases)
├── taxonomy.py             # Singapore criminal law taxonomy (~280 entries)
├── dataset.csv             # Labeled dataset (500 cases, 1,029 rows)
├── main.py                 # CLI entry point for the agentic pipeline
├── app.py                  # Streamlit web app
├── pipeline/
│   ├── extract.py          # PDF text extraction + chunking
│   ├── index.py            # ChromaDB vector index + retrieval
│   └── agents/
│       ├── manager.py      # Manager agent (tool_use orchestration)
│       ├── experts.py      # 7 specialist expert agents
│       └── qa.py           # QA agent (final advisory synthesis)
├── eval/
│   ├── test_set.py         # 20 test queries with ground truth
│   ├── retrieval_eval.py   # Hit rate + cosine similarity (no API)
│   ├── routing_eval.py     # Manager routing precision/recall/F1
│   ├── advisory_eval.py    # LLM-as-judge end-to-end scoring
│   └── run_eval.py         # Master runner (--retrieval/--routing/--advisory/--all)
├── cases/                  # Downloaded PDFs — gitignored (~500MB)
└── chroma_db/              # ChromaDB persistent store — gitignored
```

---

## Requirements

```
requests
beautifulsoup4
pandas
pdfplumber
chromadb
anthropic
streamlit
```

Install:
```bash
pip install requests beautifulsoup4 pandas pdfplumber chromadb anthropic streamlit
```

> **Note:** `chromadb`'s built-in `ONNXMiniLM_L6_V2` embedding function is used instead of `sentence-transformers` to avoid PyTorch version conflicts. No GPU required.
