# SUTD MLOPS Group 7 — Singapore Criminal Law Advisory System

An end-to-end MLOps pipeline for Singapore criminal law legal advisory. Scrapes 876 Supreme Court judgments, builds a labeled dataset, indexes them into a domain-specific vector store, and answers legal queries through a multi-agent RAG system. Supports two backends: **Claude Sonnet 4.6** (online) and a **fine-tuned Qwen2.5-1.5B** model running locally via Ollama.

---

## Quick Start

### Prerequisites

- Python 3.10
- [Ollama](https://ollama.com) (for local inference) **or** an Anthropic API key (for Claude backend)

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd MLOPSproj
pip install -r requirements.txt
```

### 2. Get the case PDFs

The `cases/` directory (876 PDFs, ~2GB) is not included in the repo. You have two options:

**Option A — Re-scrape from eLitigation (takes ~2–3 hours):**
```bash
python scraper.py
```

**Option B — Ask a teammate for the `cases/` folder** and place it in the project root.

### 3. Build the ChromaDB index (run once)

```bash
python -c "from pipeline.index import build_index; build_index()"
```

This embeds all 876 cases into 8 domain collections (~83,000 chunks). Takes ~10–20 minutes on CPU.

### 4. Run the app

**Option A — Claude backend (online):**
```bash
ANTHROPIC_API_KEY=sk-ant-... streamlit run app.py
```

**Option B — Ollama backend (fully offline):**

Install a model (choose one):
```bash
ollama pull qwen2.5:7b        # general purpose, good quality
```

Or use our fine-tuned 3B model (see Part 4 below for how to build it), then:
```bash
streamlit run app.py
```

Opens at **http://localhost:8501**. Select your backend in the sidebar.

> **Note:** If you have multiple Python versions, run Streamlit explicitly:
> ```bash
> /path/to/python3.10 -m streamlit run app.py
> ```

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
  dataset.csv  ──── 876 cases, 1,683 rows, 11 columns


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
  │                    MANAGER AGENT                             │
  │  Claude: agentic tool_use loop                               │
  │  Ollama: JSON routing prompt → sequential expert calls       │
  │  Always routes to: Sentencing + Criminal Procedure           │
  └──────────────┬───────────────────────────────────────────────┘
                 │  dispatches to (in parallel — Claude mode)
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
  │                      QA AGENT                                │
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
  │  Backend selector: Claude (online) or Ollama (local)        │
  │  Interactive UI — live pipeline status, tabbed results       │
  └──────────────────────────────────────────────────────────────┘


                        ┌─────────────────────────────────────┐
                        │       PART 3: FINE-TUNING           │
                        └─────────────────────────────────────┘

  dataset.csv + cases/ (876 judgments)
         │
         ▼
  generate_qa.py  ──── Claude Haiku generates Q&A pairs from 150 cases
         │                 (4 pairs per case → 598 pairs total)
         ▼
  data/qa_pairs.jsonl  ──── chat-format training data (JSONL)
         │
         ▼
  finetune/train.ipynb  ──── QLoRA fine-tuning on Colab T4
         │                    Base: Qwen2.5-1.5B-Instruct (4-bit)
         │                    LoRA: r=16, alpha=32, 2.04% trainable params
         │                    Epochs: 3 | LR: 2e-4 | Batch: 16 effective
         ▼
  LoRA adapters  ──── ~50MB saved checkpoint
         │
         ▼
  merge_lora.py  ──── merges adapters into base model (locally)
         │
         ▼
  llama.cpp convert  ──── GGUF export (Q4_K_M quantization → 940MB)
         │
         ▼
  ollama create sg-law-qwen2.5  ──── registered as local Ollama model
         │
         ▼
  Streamlit sidebar  ──── selectable alongside Claude backend
```

---

## Part 1: Data Pipeline

### 1.1 Scraper (`scraper.py`)

Scrapes Singapore Supreme Court criminal judgments from eLitigation.

**Key design decisions:**
- Collects up to 14× buffer of candidate URLs — ~85% of SUPCT cases are civil
- Checks `is_criminal_case()` **before** downloading PDFs (saves bandwidth and storage)
- Supports incremental runs: skips already-scraped case IDs, appends new rows to `dataset.csv`
- Per-domain quotas (`DOMAIN_TARGETS`) allow targeted expansion of under-represented domains
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

**Shape:** 1,683 rows × 11 columns | 876 unique cases | Years: 2015–2026

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
| Total rows | 1,683 |
| Unique cases | 876 |
| Avg catchwords per case | 1.92 |
| Max catchwords per case | 11 |
| Taxonomy match rate | 98.0% |
| Null subtopic | 12 rows |
| Null primary_statute | 34 rows |

#### Court breakdown (unique cases)

| Court | Cases | Description |
|---|---|---|
| SGHC | 627 (71.6%) | High Court (General Division) |
| SGCA | 223 (25.5%) | Court of Appeal |
| SGHCF | 4 (0.5%) | High Court (Family Division) |
| Unknown | 22 (2.5%) | Citation format varies |

#### Cases by year

| Year | Unique Cases |
|---|---|
| 2015 | 2 |
| 2016 | 59 |
| 2017 | 84 |
| 2018 | 76 |
| 2019 | 89 |
| 2020 | 86 |
| 2021 | 99 |
| 2022 | 91 |
| 2023 | 84 |
| 2024 | 85 |
| 2025 | 77 |
| 2026 | 22 |

> Dataset now covers 2015–2026. 2015 and 2026 are partial years (scraper captured up to March 2026).

#### Area of law distribution (rows)

| Area of Law | Rows | % |
|---|---|---|
| Criminal Law | 1,148 | 68.3% |
| Criminal Procedure | 393 | 23.4% |
| Constitutional Law | 42 | 2.5% |
| Evidence | 34 | 2.0% |
| Statutory Interpretation | 26 | 1.5% |
| Administrative Law | 10 | 0.6% |
| Civil Procedure | 7 | 0.4% |
| Other | 23 | 1.4% |

#### Topic distribution (top 10)

| Topic | Rows | % |
|---|---|---|
| Sentencing | 446 | 26.5% |
| Statutory offences | 366 | 21.7% |
| General | 208 | 12.4% |
| Sexual offences | 127 | 7.5% |
| Offences against person | 72 | 4.3% |
| General exceptions / Defences | 34 | 2.0% |
| Review | 33 | 2.0% |
| Property offences | 32 | 1.9% |
| Appeals | 30 | 1.8% |
| Statements | 29 | 1.7% |

> **Sentencing** remains the single largest topic (26.5%), reflecting that Singapore's criminal appellate practice focuses heavily on sentence calibration rather than conviction challenges.

#### Subtopic distribution (top 15)

| Subtopic | Rows | % |
|---|---|---|
| General | 555 | 33.0% |
| Misuse of Drugs Act | 220 | 13.1% |
| Appeals against sentence | 83 | 4.9% |
| Sentencing principles | 81 | 4.8% |
| Rape | 46 | 2.7% |
| General sexual offences | 41 | 2.4% |
| Criminal review | 33 | 2.0% |
| Criminal appeal | 29 | 1.7% |
| Murder | 28 | 1.7% |
| Criminal reference | 27 | 1.6% |
| Property offence | 26 | 1.5% |
| Outrage of modesty | 22 | 1.3% |
| Penal Code | 20 | 1.2% |
| Prevention of Corruption Act | 13 | 0.8% |
| Culpable homicide | 14 | 0.8% |

#### Primary statute distribution (top 10)

| Primary Statute | Rows | % |
|---|---|---|
| Criminal Procedure Code 2010 | 697 | 41.4% |
| Misuse of Drugs Act | 220 | 13.1% |
| Penal Code | 109 | 6.5% |
| Penal Code s 375 (Rape) | 46 | 2.7% |
| Penal Code Pt XI (Sexual offences) | 41 | 2.4% |
| Criminal Procedure Code 2010 s 394H (Review) | 33 | 2.0% |
| Evidence Act | 31 | 1.8% |
| Penal Code s 300 (Murder) | 28 | 1.7% |
| Criminal Procedure Code 2010 s 397 (Reference) | 27 | 1.6% |
| Interpretation Act | 26 | 1.5% |

#### is_criminal label distribution

| Label | Rows | % |
|---|---|---|
| True (criminal) | 1,542 | 91.6% |
| False (non-criminal catchword) | 141 | 8.4% |

> Cases with `is_criminal=False` rows are mixed criminal/civil matters (e.g. constitutional challenges arising from criminal proceedings, civil procedure questions in criminal appeals). The case itself is criminal — the individual catchword touches a non-criminal area.

#### Key observations

1. **Sentencing dominates** — 26.5% of all catchwords relate to sentencing. Singapore's criminal appellate practice heavily focuses on sentence calibration rather than conviction challenges.

2. **MDA is the top specific statute** — Misuse of Drugs Act cases make up 13.1% of all catchwords, driven by Singapore's mandatory death penalty jurisprudence and the large volume of trafficking cases.

3. **Court of Appeal vs High Court split** — 25.5% of cases are Court of Appeal decisions, which tend to be high-value precedent-setting judgments on sentencing frameworks and legal principles.

4. **Sexual offences are well-represented** — 7.5% of rows, driven by evolving sentencing band frameworks generating significant appellate activity.

5. **Expanded historical coverage** — Dataset now spans 2015–2026 (vs 2020–2026 previously), adding key precedent cases from before 2020.

6. **Taxonomy coverage** — Unmatched catchwords at **2.0%** (34 rows out of 1,683), consistent with the expanded dataset.

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

**Index statistics (876 cases — full build):**

| Domain | Chunks |
|---|---|
| drug_offences | 23,892 |
| sentencing | 20,496 |
| sexual_offences | 14,243 |
| criminal_procedure | 13,725 |
| violent_crimes | 7,480 |
| general | 5,382 |
| property_financial | 5,385 |
| regulatory | 1,019 |
| **Total** | **83,322** |

Build the index (run once):
```bash
python main.py --index
```

### 2.3 Manager Agent (`pipeline/agents/manager.py`)

Orchestrates the expert agents.

**Claude mode:** Uses native `tool_use` — each expert domain is registered as a tool (`consult_<domain>`). The manager runs an agentic loop until `stop_reason == "end_turn"`.

**Ollama mode:** JSON routing prompt — the model returns a list of domains, then experts are called sequentially.

Both modes always route to `sentencing` and `criminal_procedure` for any criminal query.

### 2.4 Expert Agents (`pipeline/agents/experts.py`)

Seven specialist expert agents, one per domain.

Each expert agent:
1. Retrieves the top-5 most relevant chunks from its ChromaDB collection
2. Calls the LLM backend with a domain-specific system prompt
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

An interactive web interface for the full agentic pipeline with dual backend support.

### Features

- **Backend selector** — toggle between Claude (online) and Ollama (local/offline) in the sidebar
- **Dynamic model list** — Ollama mode auto-detects all locally installed models via `ollama list`
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

## Part 4: Fine-Tuning (`finetune/`, `generate_qa.py`)

A QLoRA fine-tuned Qwen2.5-1.5B model trained on Singapore criminal law Q&A pairs, served locally via Ollama.

### 4.1 Training Data Generation (`generate_qa.py`)

598 instruction-following Q&A pairs generated from 150 sampled cases using **Claude Haiku** as a data synthesiser.

**Generation approach:**
- Samples 150 cases proportionally across `area_of_law` buckets to ensure domain coverage
- Extracts the first ~5,000 chars of each judgment (intro + key analysis)
- Prompts Claude Haiku to produce 4 diverse Q&A pairs per case covering:
  - The specific charge and statutory provisions
  - The key legal test or principle applied
  - Outcome, sentence, and driving factors
  - Any defence raised, procedural point, or evidential issue
- Outputs in chat format (`system` / `user` / `assistant` messages) for direct SFT

**Final dataset (`data/qa_pairs.jsonl`):**

| Metric | Value |
|---|---|
| Total pairs | 598 |
| Cases covered | ~150 |
| Avg answer length | ~200 words |
| Format | JSONL, chat-format (Qwen2.5 template) |

**Domain distribution:**

| Domain | Pairs | % |
|---|---|---|
| Criminal Law | 438 | 73.2% |
| Criminal Procedure | 140 | 23.4% |
| Constitutional Law | 8 | 1.3% |
| Administrative Law | 4 | 0.7% |
| Agency | 4 | 0.7% |
| Civil Procedure | 4 | 0.7% |

Generate the dataset:
```bash
python generate_qa.py                  # 150 cases, 4 pairs each
python generate_qa.py --cases 200      # more cases
python generate_qa.py --resume         # skip already-processed citations
```

### 4.2 Fine-Tuning (`finetune/train.ipynb`)

QLoRA fine-tuning on Google Colab T4 GPU using [Unsloth](https://github.com/unslothai/unsloth).

**Hardware:** Google Colab T4 (16 GB VRAM) — ~60–90 min training time

**Base model:** `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (pre-quantized)

**LoRA configuration:**

| Hyperparameter | Value |
|---|---|
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 18,464,768 / 907,081,216 **(2.04%)** |

**Training configuration:**

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 4 (per device) |
| Gradient accumulation | 4 → effective batch 16 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Optimizer | AdamW 8-bit |
| Max sequence length | 2,048 tokens |
| Train / Val split | 90% / 10% (538 / 60 pairs) |
| Eval steps | Every 50 steps |
| W&B project | `mlops-sg-law` |
| W&B run | `qwen2.5-1.5b-qlora-sg-criminal-law` |

Training and eval loss logged to Weights & Biases. Sample model outputs on 5 test queries logged as a W&B Table at end of training.

### 4.3 GGUF Export & Ollama Integration

After training, the LoRA adapters are merged into the base model and exported for local inference.

**1.5B model — conversion pipeline:**

```bash
# 1. Merge LoRA adapters into base model
python merge_lora.py

# 2. Convert to GGUF (f16)
convert_hf_to_gguf.py sg-law-merged --outfile sg-law-1.5b-f16.gguf --outtype f16

# 3. Quantize to Q4_K_M (~940MB)
llama-quantize sg-law-1.5b-f16.gguf sg-law-qwen2.5-1.5b-q4_k_m.gguf Q4_K_M

# 4. Register with Ollama
ollama create sg-law-qwen2.5 -f Modelfile
```

**3B model — conversion pipeline (better quality):**

```bash
# 1. Merge 3B LoRA adapters (base: Qwen/Qwen2.5-3B-Instruct)
python merge_lora_3b.py   # downloads base model from HuggingFace if not cached

# 2. Convert to GGUF (f16)
convert_hf_to_gguf.py sg-law-qwen2.5-3b-merged --outfile sg-law-3b-f16.gguf --outtype f16

# 3. Quantize to Q4_K_M (~1.8GB) — recommended for CPU inference
llama-quantize sg-law-3b-f16.gguf sg-law-qwen2.5-3b-q4_k_m.gguf Q4_K_M

# 4. Register with Ollama
ollama create sg-law-3b-q4 -f Modelfile-3b-q4
```

**Model sizes:**

| Model | Format | Size |
|---|---|---|
| Qwen2.5-1.5B fine-tuned | Q4_K_M GGUF | 940 MB |
| Qwen2.5-3B fine-tuned | f16 GGUF | 5.8 GB |
| Qwen2.5-3B fine-tuned | Q4_K_M GGUF | **1.8 GB** ← recommended |

Once registered, the model appears automatically in the Streamlit sidebar model dropdown. No code changes needed.

> **Tip:** Use Q4_K_M for CPU inference — f16 is too slow without a GPU.

### 4.4 Fine-Tuned Model Sample Outputs

Sample responses from `sg-law-qwen2.5` on held-out test queries:

**Q: What is the mandatory minimum sentence for drug trafficking under MDA s 5 if the accused has given substantive assistance to CNB?**
> The Sentencing Act (2019 Rev Ed) sets out clear sentencing frameworks for drug trafficking offences. For first-time offenders, the court may impose up to ten years' imprisonment or a fine of not less than $5,000... The sentencing framework for trafficking in diamorphine is particularly detailed: under s 14(2), a judge must consider whether the offender's age or mental state may mitigate culpability...

**Q: What elements must the prosecution prove for culpable homicide not amounting to murder under s 299 Penal Code?**
> In PP v Chai Wee Hooong [2018] SGHC 163, the court identified three critical elements that must be proven beyond reasonable doubt: (1) the accused's actus reus — the accused must have caused death; (2) the accused's mens rea — knowledge or intention that his act was likely to cause death; and (3) causation...

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

### Environment variable required (Claude backend)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Part 5: Evaluation Pipeline (`eval/`)

Three-layer evaluation covering retrieval quality, agent routing accuracy, and end-to-end advisory quality.

### Running the evaluation

```bash
# Retrieval only — no API key required, runs in ~30 seconds
python eval/run_eval.py --retrieval

# Routing accuracy — no API key needed for local backends
python eval/run_eval.py --routing --backend ollama --model sg-law-3b-q4:latest
python eval/run_eval.py --routing --backend ollama --model qwen2.5:7b
python eval/run_eval.py --routing --backend claude   # requires ANTHROPIC_API_KEY

# Advisory quality — judge always uses Claude (requires ANTHROPIC_API_KEY)
# Pipeline generation backend is configurable
python eval/run_eval.py --advisory --backend ollama --model sg-law-3b-q4:latest
python eval/run_eval.py --advisory --backend ollama --model qwen2.5:7b
python eval/run_eval.py --advisory --backend claude

# Everything (Claude pipeline)
python eval/run_eval.py --all
```

The `--backend` flag controls which model generates the advisory. The judge always uses `claude-sonnet-4-6` for consistent scoring regardless of backend. Supported backends:
- `claude` — Anthropic API (requires `ANTHROPIC_API_KEY`)
- `ollama` — local Ollama model (use `--model` to specify, e.g. `sg-law-3b-q4:latest`)
- `transformers` — HuggingFace model on GPU (use `--model` for HF repo ID or local path)

### Layer 1 — Retrieval Evaluation (`eval/retrieval_eval.py`)

No API calls. Queries ChromaDB directly for each of 20 test cases and measures:

- **Hit rate** — does a relevant subtopic appear in the top-5 retrieved chunks?
- **Avg relevance score** — mean cosine similarity of the top-5 results (0–1, higher is better)

**Baseline results (500 cases):**

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

> `property_financial` and `regulatory` were the weakest domains — directly motivating a targeted expansion to 876 cases.

**Post-expansion results (876 cases — 83,322 chunks):**

| Domain | Hit Rate | Avg Score |
|---|---|---|
| drug_offences | 1.000 | 0.741 |
| sentencing | 1.000 | 0.709 |
| criminal_procedure | 1.000 | 0.693 |
| violent_crimes | 1.000 | 0.679 |
| sexual_offences | 1.000 | 0.639 |
| property_financial | 1.000 | 0.663 |
| regulatory | 0.667 | 0.533 |
| **Overall** | **0.950** | **0.664** |

**Before vs After:**

| Metric | Baseline (500 cases) | Post-expansion (876 cases) | Change |
|---|---|---|---|
| Hit rate | 0.750 (15/20) | **0.950 (19/20)** | +26.7% |
| Avg cosine score | 0.583 | **0.664** | +13.9% |

> `property_financial` improved from 0.500 → 1.000 after targeted expansion. `regulatory` remains at 0.667 (1 miss: WSHA workplace fatality) — regulatory cases are predominantly in SGDC (District Court) which is not scraped from the SUPCT source.

### Layer 2 — Routing Evaluation (`eval/routing_eval.py`)

Runs the Manager Agent on 20 test queries and compares actual domains routed to against hand-labelled ground truth.

Metrics (set-based, per query):
- **Precision** — of the domains the manager selected, what fraction were correct?
- **Recall** — of the correct domains, what fraction did the manager select?
- **F1** — harmonic mean of precision and recall
- **Exact match** — did the manager select exactly the right set of domains?

**Results (20 test queries, run 2026-04-21):**

| Model | Macro Precision | Macro Recall | Macro F1 | Exact Match |
|---|---|---|---|---|
| `sg-law-3b-q4:latest` (fine-tuned) | 0.520 | **0.975** | 0.669 | 0/20 |
| `qwen2.5:7b` (base) | 0.520 | **0.975** | 0.669 | 0/20 |

Both models produce identical routing scores because the Manager Agent applies a keyword-based domain fallback that always appends `sentencing` and `criminal_procedure`. This means:
- **Recall is excellent (0.975)** — the correct domain is almost never missed
- **Precision is lower (0.520)** — extra domains are consistently added (over-routing)
- **Exact match is 0** — entirely due to extra domains being appended, not missing ones

The routing behaviour is dominated by the fallback heuristic rather than the LLM, so fine-tuning does not affect routing scores. The real differentiation between models appears in advisory quality (Layer 3).

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

**Results (10-query subset, run 2026-04-21):**

| Dimension | `sg-law-3b-q4` (fine-tuned GGUF) | `qwen2.5:7b` (base) | `claude` (pipeline) |
|---|:---:|:---:|:---:|
| Legal Accuracy | 1.00 | 1.00 | **3.90** |
| Completeness | 1.00 | 1.00 | **5.00** |
| Citation Quality | 1.00 | 1.00 | **3.10** |
| Format Compliance | 1.00 | 2.00 | **5.00** |
| Actionability | 1.00 | 1.30 | **5.00** |
| **Overall Avg** | **1.00 / 5.00** | **1.26 / 5.00** | **4.40 / 5.00** |

**Key findings:**
- **Claude pipeline (4.40/5.00)** — consistently complete, perfectly formatted, and actionable. Main deductions on `legal_accuracy` (3.90) and `citation_quality` (3.10) reflect hallucinated or imprecise citations, a known LLM limitation.
- **`qwen2.5:7b` base (1.26/5.00)** — marginally better than the fine-tuned GGUF on format (2.00 vs 1.00) but fails on all content dimensions. The model produces structurally present but legally incorrect output.
- **`sg-law-3b-q4` fine-tuned GGUF (1.00/5.00)** — severe hallucination in the quantized GGUF format: words run together, legal content replaced with random keyboard shortcut documentation. Root cause: 1200-token limit (now fixed to 2500) caused the model to hallucinate when forced to compress a full advisory into insufficient context. The HuggingFace LoRA version on GPU performs significantly better.

**Why local 3B models score low:** A 3B parameter model compressed to 4-bit quantization (Q4_K_M) has insufficient capacity to simultaneously (a) follow a 6-section structured format, (b) produce accurate Singapore legal citations, and (c) synthesise multiple expert findings coherently. The Claude pipeline uses a 200B+ model with much higher capacity for structured generation.

#### Known limitation

Claude judging its own output introduces self-evaluation bias — Claude pipeline scores will tend to skew 3.5–4.5. For cross-model comparison, using GPT-4o as judge would eliminate this bias and likely show a larger gap between Claude and local models.

### Test Set (`eval/test_set.py`)

20 hand-crafted queries with ground truth covering all 7 expert domains, including cross-domain cases (e.g. drug trafficking + sentencing, sexual offences + criminal procedure).

Each test case includes:
- `expected_domains` — correct domains for routing eval
- `domain_for_retrieval` — which ChromaDB collection to test
- `relevant_subtopics` — exact subtopic strings from dataset metadata (for hit detection)
- `expected_keywords` — keywords that should appear in the final advisory

---

## Technique Comparison Summary

Three distinct techniques were implemented and evaluated across all three evaluation layers (retrieval, routing, advisory quality). All advisory scores use Claude-as-judge on the same 10-query test subset.

| Technique | Model | Retrieval Hit Rate | Routing Macro-F1 | Advisory Score |
|---|---|:---:|:---:|:---:|
| **Agentic RAG + Claude** | `claude-sonnet-4-6` | 0.950 | — | **4.40 / 5.00** |
| **Agentic RAG + Base LLM** | `qwen2.5:7b` (Ollama) | 0.950 | 0.669 | 1.26 / 5.00 |
| **Agentic RAG + Fine-Tuned LLM** | `sg-law-3b-q4` (Ollama GGUF) | 0.950 | 0.669 | 1.00 / 5.00 |
| **QLoRA Fine-Tuning** | `MikeUmali/sg-law-qwen2.5-3b-lora` (HF GPU) | — | — | qualitative only |

**Advisory quality breakdown (Claude-as-judge, 1–5 scale):**

| Dimension | `sg-law-3b-q4` GGUF | `qwen2.5:7b` base | `claude` pipeline |
|---|:---:|:---:|:---:|
| Legal Accuracy | 1.00 | 1.00 | **3.90** |
| Completeness | 1.00 | 1.00 | **5.00** |
| Citation Quality | 1.00 | 1.00 | **3.10** |
| Format Compliance | 1.00 | 2.00 | **5.00** |
| Actionability | 1.00 | 1.30 | **5.00** |
| **Overall** | **1.00** | **1.26** | **4.40** |

**Key findings:**

1. **Retrieval quality is model-independent** — ChromaDB hit rate (0.950) is identical across all backends since retrieval uses fixed ONNX embeddings, not the LLM.

2. **Routing is dominated by keyword heuristics** — both local models achieve identical F1 (0.669) with near-perfect recall (0.975) because the Manager Agent falls back to keyword matching. The LLM choice does not affect routing.

3. **Advisory quality is where model choice matters most** — Claude (4.40/5.00) vastly outperforms local 3B models (1.00–1.26/5.00). A 3B quantized model lacks sufficient capacity to simultaneously follow a structured 6-section format, produce accurate legal citations, and synthesise multiple expert findings.

4. **Fine-tuning improves domain grounding but not synthesis** — the fine-tuned LoRA model (`sg-law-qwen2.5-3b-lora`) produces better Singapore-specific terminology and citation style for targeted factual Q&A, but the 3B parameter size is a hard limit for full multi-issue advisory synthesis. The GGUF quantized variant (Q4_K_M) further degrades quality due to truncation at 1200 tokens (now fixed to 2500).

5. **Final architecture decision** — Claude pipeline selected for production advisory generation; local fine-tuned model retained as the offline/cluster fallback with improved prompting and token limits applied.


## Demo Video
https://www.youtube.com/watch?v=rY_FEKoa0Hs


## File Structure

```
MLOPSproj/
├── scraper.py              # eLitigation scraper (876 criminal cases)
├── taxonomy.py             # Singapore criminal law taxonomy (~280 entries)
├── dataset.csv             # Labeled dataset (876 cases, 1,683 rows)
├── generate_qa.py          # Q&A pair generation via Claude Haiku
├── merge_lora.py           # Merge LoRA adapters into base model (local)
├── Modelfile               # Ollama Modelfile for sg-law-qwen2.5
├── main.py                 # CLI entry point for the agentic pipeline
├── app.py                  # Streamlit web app (Claude + Ollama backends)
├── pipeline/
│   ├── extract.py          # PDF text extraction + chunking
│   ├── index.py            # ChromaDB vector index + retrieval
│   ├── llm.py              # LLM backend abstraction (Claude / Ollama)
│   └── agents/
│       ├── manager.py      # Manager agent (tool_use / JSON routing)
│       ├── experts.py      # 7 specialist expert agents
│       └── qa.py           # QA agent (final advisory synthesis)
├── finetune/
│   └── train.ipynb         # QLoRA fine-tuning notebook (Colab T4)
├── data/
│   └── qa_pairs.jsonl      # 598 Q&A training pairs (chat format)
├── eval/
│   ├── test_set.py         # 20 test queries with ground truth
│   ├── retrieval_eval.py   # Hit rate + cosine similarity (no API)
│   ├── routing_eval.py     # Manager routing precision/recall/F1
│   ├── advisory_eval.py    # LLM-as-judge end-to-end scoring
│   └── run_eval.py         # Master runner (--retrieval/--routing/--advisory/--all)
├── lora_adapters/          # Fine-tuned LoRA weights — gitignored
├── sg-law-merged/          # Merged HuggingFace model — gitignored
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
ollama
```

Install:
```bash
pip install requests beautifulsoup4 pandas pdfplumber chromadb anthropic streamlit ollama
```

For fine-tuning (Colab):
```bash
pip install unsloth wandb datasets
```

For local GGUF conversion:
```bash
brew install llama.cpp
pip install gguf peft transformers accelerate
```

> **Note:** `chromadb`'s built-in `ONNXMiniLM_L6_V2` embedding function is used instead of `sentence-transformers` to avoid PyTorch version conflicts. No GPU required for inference.

---

## Compute

| Task | Hardware | Est. GPU Hours |
|---|---|---|
| Fine-tuning (3 epochs, 538 samples) | Google Colab T4 (16GB) | ~1.5 hours |
| GGUF conversion + quantization | MacBook (CPU) | ~15 min |
| Index build (83,322 chunks) | MacBook (CPU) | ~45 min |
| Evaluation runs | MacBook (CPU) | ~10 min (retrieval), ~30 min (advisory) |

---

## Running on SUTD GPU Cluster

The app supports a **GPU (Local Model)** backend that runs the fine-tuned Qwen2.5-3B model on the SUTD JupyterHub cluster (RTX PRO 6000 Blackwell GPU).

### Cluster details

| Item | Value |
|---|---|
| URL | http://192.168.33.15:8888 |
| GPU | NVIDIA RTX PRO 6000 Blackwell (sm_120) |
| Python | 3.11 (via `/opt/conda`) |
| Login | Student ID + password |

### 1. SSH / open a terminal

Log in at http://192.168.33.15:8888, open a terminal from JupyterLab.

### 2. Clone the repo

```bash
git clone https://github.com/Mike-Umali/SUTD_MLOPS_GRP7.git MLOPSproj
cd MLOPSproj
```

### 3. Install dependencies

```bash
pip install anthropic streamlit pdfplumber chromadb transformers accelerate peft huggingface_hub bitsandbytes --user
```

> `bitsandbytes` is required because the fine-tuned LoRA was trained with 4-bit QLoRA.

### 4. Upload the ChromaDB index

The ChromaDB index (`chroma_db/`, ~750MB) must be copied to the cluster — it cannot be rebuilt without the original PDFs. From your Mac:

```bash
# Zip on Mac
cd ~/Desktop/MLOPSproj
zip -r chroma_db.zip chroma_db/
```

Then upload `chroma_db.zip` via JupyterLab → File Browser → Upload, and unzip on the cluster:

```python
# In a cluster notebook or terminal
import zipfile
with zipfile.ZipFile('chroma_db.zip', 'r') as z:
    z.extractall('.')
```

### 5. Run the app

```bash
cd ~/MLOPSproj
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Access at: **http://192.168.33.15:8501**

### 6. Select the GPU backend

In the Streamlit sidebar:
- **Backend:** `GPU (Local Model)`
- **Model path:** `MikeUmali/sg-law-qwen2.5-3b-lora`

The app will automatically:
1. Download the base `Qwen/Qwen2.5-3B-Instruct` model from HuggingFace
2. Download and apply the LoRA adapter (`MikeUmali/sg-law-qwen2.5-3b-lora`)
3. Merge the adapter into the base model and run on GPU

> First load takes ~2–3 minutes to download models. Subsequent queries are fast.

### Cluster-specific notes

- **llama-cpp-python (GGUF) is not supported** — the cluster's CUDA 12.0 toolkit does not support Blackwell's `compute_120a` architecture. Use the HF Transformers backend instead.
- **Ollama is not available** on the cluster — use GPU (Local Model) backend.
- **`streamlit` command may not be on PATH** — always use `python -m streamlit run app.py`.
- If you get `ModuleNotFoundError` for any package, reinstall with `pip install <package> --user`.
- The cluster does not persist pip packages between sessions if the home directory is reset — re-run the install command if needed.

### Git pull on cluster

If git pull fails (no credentials), set up a token:

```bash
git config credential.helper store
git pull
# Enter your GitHub username and a Personal Access Token when prompted
```

Create a token at github.com/settings/tokens with `repo` scope.
