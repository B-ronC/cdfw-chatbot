import hashlib
import logging
import os
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

REGULATIONS_URL = "https://wildlife.ca.gov/Regulations"
DATA_DIR = Path("./data")
REQUEST_TIMEOUT = 30
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CDFW-RegUpdater/1.0)"}
LINK_TEXT_KEYWORDS = ["freshwater", "sport", "fishing", "regulations", "booklet"]

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def find_pdf_url(page_url: str) -> str:
    resp = requests.get(page_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup.find_all("a", href=True):
        href: str = tag["href"].strip()
        text: str = tag.get_text(separator=" ", strip=True).lower()
        lower_href = href.lower()

        if not (lower_href.endswith(".pdf") or "filehandler.ashx" in lower_href):
            continue

        if all(kw in text for kw in LINK_TEXT_KEYWORDS):
            return urllib.parse.urljoin(page_url, href)

    raise RuntimeError(
        f"No freshwater sport fishing PDF link found on {page_url}. "
        "The page layout may have changed."
    )


def download_pdf(url: str) -> bytes:
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
    resp.raise_for_status()
    return b"".join(chunk for chunk in resp.iter_content(65536) if chunk)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_filename_from_response(url: str) -> str:
    """Get the filename from Content-Disposition header, or fall back to URL path."""
    resp = requests.head(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        name = cd.split("filename=")[-1].strip().strip('"')
        if name:
            return name
    # Fall back to the path basename
    path = urllib.parse.urlparse(resp.url).path
    name = os.path.basename(path)
    return name if name else "freshwater_regulations.pdf"


def main() -> None:
    log.info("Checking for updated regulations PDF...")

    pdf_url = find_pdf_url(REGULATIONS_URL)
    log.info("Found PDF URL: %s", pdf_url)

    new_bytes = download_pdf(pdf_url)
    new_hash = sha256_bytes(new_bytes)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if any existing PDF matches the hash (already up-to-date)
    existing_pdfs = list(DATA_DIR.glob("*.pdf"))
    for existing in existing_pdfs:
        if sha256_file(existing) == new_hash:
            log.info("Already up-to-date: %s", existing.name)
            return

    # Delete all old PDFs in data dir
    for existing in existing_pdfs:
        log.info("Deleting old PDF: %s", existing.name)
        existing.unlink()

    # Save new PDF with its original filename
    filename = get_filename_from_response(pdf_url)
    dest = DATA_DIR / filename
    dest.write_bytes(new_bytes)
    log.info("Saved new PDF: %s (%d bytes)", dest, len(new_bytes))


if __name__ == "__main__":
    main()
