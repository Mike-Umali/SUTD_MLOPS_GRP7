import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

import taxonomy
from pipeline.extract import assign_domain

BASE = "https://www.elitigation.sg/gd/Home/Index"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

MAX_NEW_CASES = 500       # How many new cases to add on top of existing
LINK_BUFFER = 14          # 14x buffer — being selective so need more links

# Per-domain quotas for the new 500 cases.
# Targets the three weakest domains from retrieval eval.
# Domains not listed here have no cap — they fill any remaining slots.
DOMAIN_TARGETS = {
    "property_financial": 150,
    "regulatory":         150,
    "violent_crimes":     100,
    "sexual_offences":    100,
}

os.makedirs("cases", exist_ok=True)

dataset = []
CSV_PATH = "dataset.csv"


def load_existing():
    """Load already-scraped case IDs and rows from dataset.csv."""
    if not os.path.exists(CSV_PATH):
        return set(), []
    df = pd.read_csv(CSV_PATH)
    existing_ids = set(df["filename"].str.replace(".pdf", "", regex=False))
    return existing_ids, df.to_dict("records")


# -----------------------------------------
# STEP 1 collect case links
# -----------------------------------------

def collect_links():

    links = []
    page = 1
    target = MAX_NEW_CASES * LINK_BUFFER

    print("\nStarting scraper...\n")

    while len(links) < target:

        url = f"{BASE}?CurrentPage={page}&Filter=SUPCT"

        print(f"\nOpening page {page}")

        r = requests.get(url, headers=HEADERS)

        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.find_all("a", href=True):

            href = a["href"]

            if "/gd/s/" in href:

                full = "https://www.elitigation.sg" + href

                if full not in links:

                    print("Found case:", full)

                    links.append(full)

            if len(links) >= target:
                break

        page += 1
        time.sleep(1)

    print("\nCollected", len(links), "links")

    return links


# -----------------------------------------
# STEP 2 scrape case page
# -----------------------------------------

def scrape_case(url):

    print("\nOpening case:", url)

    r = requests.get(url, headers=HEADERS)

    soup = BeautifulSoup(r.text, "html.parser")

    case_id = url.split("/")[-1]

    pdf_url = f"https://www.elitigation.sg/gd/gd/{case_id}/pdf"

    title = soup.title.text.strip() if soup.title else ""

    citation_tag = soup.select_one(".HN-NeutralCit")
    citation = citation_tag.text.strip() if citation_tag else ""

    catchwords = [
        c.text.strip().replace("[", "").replace("]", "")
        for c in soup.select(".catchwords")
    ]

    # Filter blank catchwords
    catchwords = [cw for cw in catchwords if cw]

    print("Citation:", citation)
    print("Catchwords:", catchwords)

    return case_id, title, citation, catchwords, pdf_url


# -----------------------------------------
# STEP 3 download pdf
# -----------------------------------------

def download_pdf(pdf_url, filename):

    print("Downloading:", filename)

    r = requests.get(pdf_url, headers=HEADERS)

    with open(f"cases/{filename}", "wb") as f:
        f.write(r.content)


# -----------------------------------------
# MAIN
# -----------------------------------------

def _case_domain(catchwords):
    """Classify a case's domain from its catchwords using taxonomy + assign_domain."""
    areas, topics, subtopics = [], [], []
    for cw in catchwords:
        result = taxonomy.classify_catchword(cw)
        if result:
            area, topic, subtopic, _, _, _ = result
        else:
            area, topic, subtopic = taxonomy.split_catchword(cw)
        if area:   areas.append(area)
        if topic:  topics.append(topic)
        if subtopic: subtopics.append(subtopic)
    return assign_domain(areas, topics, subtopics)


def main():

    existing_ids, existing_rows = load_existing()
    print(f"Loaded {len(existing_ids)} existing cases from {CSV_PATH}")

    # Seed dataset with existing rows so we append, not overwrite
    dataset.extend(existing_rows)

    links = collect_links()

    new_count = 0
    domain_counts = {d: 0 for d in DOMAIN_TARGETS}

    for url in links:

        if new_count >= MAX_NEW_CASES:
            break

        try:

            case_id, title, citation, catchwords, pdf_url = scrape_case(url)

            # Skip already-scraped cases
            if case_id in existing_ids:
                print(f"  Already scraped: {case_id} — skipping")
                continue

            # Skip non-criminal cases before downloading PDF
            is_criminal = any(
                taxonomy.is_criminal_case(taxonomy.split_catchword(cw)[0])
                for cw in catchwords
            ) if catchwords else False

            if not is_criminal:
                print("Skipping non-criminal case:", citation)
                continue

            # Determine domain and check quota
            domain = _case_domain(catchwords)
            if domain in DOMAIN_TARGETS and domain_counts[domain] >= DOMAIN_TARGETS[domain]:
                print(f"  Quota full for {domain} — skipping")
                continue

            new_count += 1
            if domain in domain_counts:
                domain_counts[domain] += 1

            print(f"New case {new_count}/{MAX_NEW_CASES} | domain={domain} | {citation}")
            print(f"  Domain counts: { {d: domain_counts[d] for d in DOMAIN_TARGETS} }")

            filename = case_id + ".pdf"
            download_pdf(pdf_url, filename)
            existing_ids.add(case_id)

            if not catchwords:
                dataset.append({
                    "filename": filename,
                    "case_name": title,
                    "citation": citation,
                    "catchword": "",
                    "area_of_law": "",
                    "topic": "",
                    "subtopic": "",
                    "primary_statute": "",
                    "is_criminal": False,
                    "taxonomy_key": "",
                    "pdf_url": pdf_url,
                })
            else:
                for cw in catchwords:

                    result = taxonomy.classify_catchword(cw)

                    if result:
                        area, topic, subtopic, statute, is_criminal, tax_key = result
                    else:
                        area, topic, subtopic = taxonomy.split_catchword(cw)
                        statute = ""
                        is_criminal = taxonomy.is_criminal_case(area)
                        tax_key = ""

                    dataset.append({
                        "filename": filename,
                        "case_name": title,
                        "citation": citation,
                        "catchword": cw,
                        "area_of_law": area,
                        "topic": topic,
                        "subtopic": subtopic,
                        "primary_statute": statute,
                        "is_criminal": is_criminal,
                        "taxonomy_key": tax_key,
                        "pdf_url": pdf_url,
                    })

            time.sleep(1)

        except Exception as e:
            print("Error:", e)

    df = pd.DataFrame(dataset)
    df.to_csv(CSV_PATH, index=False)

    total_cases = df["filename"].nunique()
    print(f"\nDONE. {new_count} new cases added. Total unique cases: {total_cases}")
    print(f"Domain counts for new cases: { {d: domain_counts[d] for d in DOMAIN_TARGETS} }")
    print(f"{CSV_PATH} saved ({len(df)} rows)")


if __name__ == "__main__":
    main()
