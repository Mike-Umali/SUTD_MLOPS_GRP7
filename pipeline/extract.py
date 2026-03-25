"""
PDF text extraction and chunking.
Reads PDFs from cases/, extracts text, chunks by paragraph, maps to dataset.csv metadata.
"""

import os
import pdfplumber
import pandas as pd
from typing import Generator

CASES_DIR = "cases"
CHUNK_SIZE = 800    # characters per chunk
CHUNK_OVERLAP = 150


def extract_text(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def load_case_metadata(csv_path: str = "dataset.csv") -> dict:
    """
    Load dataset.csv and return a dict mapping filename → list of metadata dicts.
    Each metadata dict has one entry per catchword row.
    """
    df = pd.read_csv(csv_path)
    metadata = {}
    for filename, group in df.groupby("filename"):
        rows = group.to_dict("records")
        metadata[filename] = rows
    return metadata


def iter_case_chunks(csv_path: str = "dataset.csv") -> Generator[dict, None, None]:
    """
    Yield one dict per chunk across all cases.
    Each dict contains: text, chunk_id, filename, citation, case_name,
    area_of_law, topic, subtopic, primary_statute, domain
    """
    metadata = load_case_metadata(csv_path)
    pdf_files = [f for f in os.listdir(CASES_DIR) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(CASES_DIR, pdf_file)
        rows = metadata.get(pdf_file, [])

        if not rows:
            continue

        # Use first row for case-level metadata
        first = rows[0]
        citation = first.get("citation", "")
        case_name = first.get("case_name", "")

        # Collect all areas/topics for this case
        areas = list({r.get("area_of_law", "") for r in rows if r.get("area_of_law")})
        topics = list({r.get("topic", "") for r in rows if r.get("topic")})
        subtopics = list({r.get("subtopic", "") for r in rows if r.get("subtopic")})
        statutes = list({r.get("primary_statute", "") for r in rows if r.get("primary_statute")})
        domain = assign_domain(areas, topics, subtopics)

        text = extract_text(pdf_path)
        if not text:
            continue

        chunks = chunk_text(text)
        print(f"  {pdf_file}: {len(chunks)} chunks | domain={domain}")

        for i, chunk in enumerate(chunks):
            yield {
                "text": chunk,
                "chunk_id": f"{pdf_file}::chunk_{i}",
                "filename": pdf_file,
                "citation": citation,
                "case_name": case_name,
                "area_of_law": "; ".join(areas),
                "topic": "; ".join(topics),
                "subtopic": "; ".join(subtopics),
                "primary_statute": "; ".join(statutes),
                "domain": domain,
            }


def assign_domain(areas: list, topics: list, subtopics: list) -> str:
    """
    Map a case's areas/topics/subtopics to one of the expert agent domains.
    Returns the primary domain string.
    """
    all_text = " ".join(areas + topics + subtopics).lower()

    if any(k in all_text for k in ["misuse of drugs", "drug trafficking", "drug consumption",
                                    "drug possession", "drug importation", "enhanced trafficking"]):
        return "drug_offences"

    if any(k in all_text for k in ["rape", "sexual assault", "outrage of modesty", "voyeurism",
                                    "sexual exploitation", "unnatural offences", "sexual offences",
                                    "intimate image"]):
        return "sexual_offences"

    if any(k in all_text for k in ["murder", "culpable homicide", "hurt", "assault", "kidnapping",
                                    "stalking", "grievous hurt", "wrongful confinement",
                                    "offences against person", "attempted murder"]):
        return "violent_crimes"

    if any(k in all_text for k in ["theft", "robbery", "cheating", "criminal breach of trust",
                                    "forgery", "house-breaking", "extortion", "corruption",
                                    "money laundering", "mischief", "property offences",
                                    "fraud", "bribery", "cdsa"]):
        return "property_financial"

    if any(k in all_text for k in ["sentencing", "benchmark", "mitigating", "aggravating",
                                    "probation", "reformative", "caning", "preventive detention",
                                    "mandatory minimum", "totality"]):
        return "sentencing"

    if any(k in all_text for k in ["road traffic", "drink driving", "dangerous driving",
                                    "careless driving", "immigration", "workplace safety",
                                    "customs", "wildlife", "vandalism", "moneylenders",
                                    "computer misuse", "arms offences"]):
        return "regulatory"

    if any(k in all_text for k in ["bail", "charge", "arrest", "trial", "appeal", "review",
                                    "revision", "criminal procedure", "statements", "confession",
                                    "acquittal", "extradition", "criminal reference"]):
        return "criminal_procedure"

    return "general"
