import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

import taxonomy

BASE = "https://www.elitigation.sg/gd/Home/Index"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

MAX_CRIMINAL_CASES = 500

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

    print("\nStarting scraper...\n")

    while len(links) < MAX_CRIMINAL_CASES * 7:  # fetch ~7x to account for civil cases

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

            if len(links) >= MAX_CRIMINAL_CASES * 7:
                break

        page += 1
        time.sleep(1)

    print("\nCollected", len(links), "cases")

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

def main():

    links = collect_links()

    criminal_count = 0

    for url in links:

        if criminal_count >= MAX_CRIMINAL_CASES:
            break

        try:

            case_id, title, citation, catchwords, pdf_url = scrape_case(url)

            # Skip non-criminal cases before downloading PDF
            is_criminal = any(
                taxonomy.is_criminal_case(taxonomy.split_catchword(cw)[0])
                for cw in catchwords
            ) if catchwords else False

            if not is_criminal:
                print("Skipping non-criminal case:", citation)
                continue

            criminal_count += 1
            print(f"Criminal case {criminal_count}/{MAX_CRIMINAL_CASES}")

            filename = case_id + ".pdf"

            download_pdf(pdf_url, filename)

            if not catchwords:
                # Still record the case with empty labels
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
                        # Fall back to split_catchword for unmatched entries
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

    df.to_csv("dataset.csv", index=False)

    print("\nDONE. dataset.csv saved")


if __name__ == "__main__":
    main()
