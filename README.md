# SUTD MLOPS Group 7 — Singapore Criminal Law Case Dataset

A labeled dataset of Singapore criminal law judgments scraped from [eLitigation](https://www.elitigation.sg), built as the data pipeline foundation for a Retrieval-Augmented Generation (RAG) system.

---

## Project Overview

This pipeline scrapes Supreme Court criminal judgments from eLitigation, classifies each case using a handcrafted Singapore criminal law taxonomy, and produces a structured labeled CSV dataset for downstream ML use.

---

## Pipeline

```
eLitigation (SUPCT) → collect_links() → scrape_case() → criminal filter
       → taxonomy.classify_catchword() → dataset.csv + cases/ PDFs
```

### Step 1: Link Collection
- Scrapes case listing pages from eLitigation (`Filter=SUPCT`)
- Collects up to 3,500 case URLs (7x buffer to account for ~85% civil cases)
- Stops once 500 confirmed criminal cases are found

### Step 2: Case Scraping
- For each URL, scrapes case title, citation, and catchwords
- Filters out non-criminal cases using `taxonomy.is_criminal_case()` **before** downloading the PDF (saves bandwidth and storage)
- Downloads PDF only for confirmed criminal cases

### Step 3: Taxonomy Classification
- Each catchword is matched against `CRIMINAL_TAXONOMY` using longest-prefix matching
- Matching is case-insensitive and dash-normalized (handles em-dash, en-dash, double en-dash)
- Falls back to `split_catchword()` for unmatched catchwords
- Produces: `area_of_law`, `topic`, `subtopic`, `primary_statute`, `is_criminal`, `taxonomy_key`

---

## Files

| File | Description |
|---|---|
| `scraper.py` | Main scraping pipeline |
| `taxonomy.py` | Singapore criminal law taxonomy + classification helpers |
| `dataset.csv` | Labeled dataset (500 cases, 1,029 rows) |
| `cases/` | Downloaded PDFs (gitignored — ~500MB) |

---

## Taxonomy (`taxonomy.py`)

A handcrafted Singapore criminal law taxonomy with **~280 entries** covering:

| Area | Categories |
|---|---|
| Offences against person | Murder, Culpable homicide, Hurt, Assault, Kidnapping, Stalking |
| Sexual offences | Rape, SAP, Outrage of modesty, Voyeurism, Exploitation of minor |
| Property offences | Theft, Robbery, CBT, Cheating, House-breaking, Extortion |
| Forgery and fraud | Forgery, Using forged document |
| Public order | Rioting, Unlawful assembly, Affray |
| Offences against administration | Obstructing public servant, False evidence, Fabricating evidence |
| Statutory offences (MDA) | Drug trafficking, possession, consumption, importation |
| Statutory offences (other) | Arms, Corruption, Money laundering, Computer Misuse, Immigration, Road Traffic, Companies Act, WSH Act, Women's Charter, and more |
| Elements of crime | Mens rea, Actus reus, Intention, Knowledge, Recklessness, Causation |
| Participation | Common intention, Abetment, Criminal conspiracy, Attempt |
| General exceptions | Private defence, Intoxication, Unsoundness of mind, Diminished responsibility, Provocation |
| Sentencing | Benchmarks, Bands, Mandatory minimum, Probation, CBS, Caning, Totality, Deterrence |
| Criminal Procedure | Arrest, Bail, Charge, Trial, Statements, Appeals, Review, Reference |
| Evidence | Hearsay, Expert evidence, Admissibility, Similar fact, Corroboration, Presumptions |
| Constitutional Law | Equal protection, Fundamental liberties, Judicial review, Accused person rights |

**Key functions:**
- `classify_catchword(cw)` — longest-prefix match → `(area, topic, subtopic, statute, is_criminal, taxonomy_key)`
- `split_catchword(raw)` — dash-normalized fallback split
- `is_criminal_case(area)` — keyword-based criminal detection
- `_normalize_dashes(text)` — normalizes em-dash, en-dash, double en-dash, spaced hyphens

---

## Dataset (`dataset.csv`)

**Shape:** 1,029 rows × 11 columns (500 unique cases, ~2.1 catchwords per case)

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

---

## EDA Summary

### Case distribution by year
| Year | Cases |
|---|---|
| 2020 | 21 |
| 2021 | 100 |
| 2022 | 95 |
| 2023 | 93 |
| 2024 | 90 |
| 2025 | 80 |
| 2026 | 21 |

### Top areas of law
| Area | Rows |
|---|---|
| Criminal Law | 635 |
| Criminal Procedure | 258 |
| Constitutional Law | 37 |
| Evidence | 23 |

### Top topics
| Topic | Count |
|---|---|
| Sentencing | 253 |
| Statutory offences | 177 |
| Sexual offences | 77 |
| Offences against person | 29 |

### Top subtopics
| Subtopic | Count |
|---|---|
| Misuse of Drugs Act | 102 |
| Sentencing principles | 56 |
| Appeals against sentence | 47 |
| General sexual offences | 38 |
| Murder | 17 |

### Data quality fixes applied
| Issue | Fix |
|---|---|
| Dash variants (em, en, double en-dash) | Normalized in `_normalize_dashes()` |
| Case mismatch (`Criminal law` vs `Criminal Law`) | Case-insensitive prefix matching in `classify_catchword()` |
| 1 corrupt row (case text leaked into catchword) | Dropped |
| 38 null citations | Filled from filename |
| 21.5% taxonomy mismatch → 1.7% | Added ~100 missing taxonomy entries |

---

## Next Steps

- [ ] Extract text from PDFs (`pdfplumber` / `pymupdf`)
- [ ] Chunk and embed case text for RAG
- [ ] Build retrieval index (FAISS / ChromaDB)
- [ ] Wire up LLM for generation (Claude API)
- [ ] Evaluate retrieval quality

---

## Requirements

```
requests
beautifulsoup4
pandas
```

Install:
```
pip install requests beautifulsoup4 pandas
```
