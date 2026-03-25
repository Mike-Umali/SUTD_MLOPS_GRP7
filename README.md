# SUTD MLOPS Group 7 — Singapore Criminal Law Agentic RAG Pipeline

An end-to-end MLOps pipeline that scrapes Singapore criminal law judgments, builds a labeled dataset, indexes them into a vector store, and answers legal queries through a multi-agent RAG system powered by Claude.

---

## System Architecture

```
                        ┌─────────────────────────────────────┐
                        │           PART 1: DATA PIPELINE      │
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
                        │       PART 2: AGENTIC RAG PIPELINE   │
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
  User Query
         │
         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    MANAGER AGENT (Claude)                     │
  │  Analyses query, selects relevant expert domains via tool_use │
  │  Always routes to: Sentencing + Criminal Procedure            │
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
  │                      QA AGENT (Claude)                        │
  │  Synthesises all expert findings into a structured advisory   │
  │  Written in Singapore High Court judgment style               │
  └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
  Final Advisory: Classification | Legal Issues | Applicable Law |
                  Analysis | Next Steps | Cases Referenced
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
| `is_criminal` | Boolean — whether case is criminal |
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

### 1.4 EDA Summary

**Case distribution by year:**

| Year | Cases |
|---|---|
| 2020 | 21 |
| 2021 | 100 |
| 2022 | 95 |
| 2023 | 93 |
| 2024 | 90 |
| 2025 | 80 |
| 2026 | 21 |

**Top areas of law:**

| Area | Rows |
|---|---|
| Criminal Law | 635 |
| Criminal Procedure | 258 |
| Constitutional Law | 37 |
| Evidence | 23 |

**Top topics:**

| Topic | Count |
|---|---|
| Sentencing | 253 |
| Statutory offences | 177 |
| Sexual offences | 77 |
| Offences against person | 29 |

**Top subtopics:**

| Subtopic | Count |
|---|---|
| Misuse of Drugs Act | 102 |
| Sentencing principles | 56 |
| Appeals against sentence | 47 |
| General sexual offences | 38 |
| Murder | 17 |

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
- Each expert domain is registered as a Claude tool (`call_<domain>_expert`)
- Always routes to `sentencing` and `criminal_procedure` for any criminal query
- Runs an agentic loop until `stop_reason == "end_turn"` (all experts consulted)
- Returns `expert_results` list with each expert's findings and citations

### 2.4 Expert Agents (`pipeline/agents/experts.py`)

Seven specialist expert agents, one per domain.

Each expert agent:
1. Retrieves the top-5 most relevant chunks from its ChromaDB collection
2. Calls Claude with a domain-specific system prompt (e.g. "You are a Singapore criminal law expert specialising in drug offences under the MDA...")
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

## Usage

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

## File Structure

```
MLOPSproj/
├── scraper.py              # eLitigation scraper (500 criminal cases)
├── taxonomy.py             # Singapore criminal law taxonomy (~280 entries)
├── dataset.csv             # Labeled dataset (500 cases, 1,029 rows)
├── main.py                 # Entry point for the agentic pipeline
├── pipeline/
│   ├── extract.py          # PDF text extraction + chunking
│   ├── index.py            # ChromaDB vector index + retrieval
│   └── agents/
│       ├── manager.py      # Manager agent (tool_use orchestration)
│       ├── experts.py      # 7 specialist expert agents
│       └── qa.py           # QA agent (final advisory synthesis)
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
```

Install:
```bash
pip install requests beautifulsoup4 pandas pdfplumber chromadb anthropic
```

> **Note:** `chromadb`'s built-in `ONNXMiniLM_L6_V2` embedding function is used instead of `sentence-transformers` to avoid PyTorch version conflicts. No GPU required.
