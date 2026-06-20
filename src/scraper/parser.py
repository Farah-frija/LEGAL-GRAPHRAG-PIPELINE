"""
Phase 1 — Individual document page parser.

Content source matrix:

AR content priority:
  1. HTML body 
  2. AR PDF 
  3. None

EN content priority:
  1. decree.om HTML page 
  2. EN PDF 
  3. None


Error handling:
  PaywallError     → log WARNING, try EN PDF fallback, else content_en = None
  ImagePDFError    → log ERROR with doc_id + URL, content = None
  EmptyPDFError    → log ERROR with doc_id + URL, content = None
  Network errors   → log ERROR, content = None (tenacity retries upstream)
"""

# ============================================================
# IMPORTS
# ============================================================

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup
from loguru import logger

from src.scraper.config import AR_TO_KEY
from src.scraper.markdown_converter import (
    html_body_to_markdown,
    pdf_to_markdown,
    download_pdf,
    PaywallError,
    ImagePDFError,
    EmptyPDFError,
)
from src.scraper.models import AmendmentRelation, ScrapedDocument
from src.scraper.utils import fetch


# ============================================================
# CONSTANTS
# ============================================================

_NOISE_PATTERNS = (
    "نشر في عدد الجريدة الرسمية",
    "تم التصديق بموجب",
    "للأسف نص هذه الوثيقة غير متوفر",
)

AR_MONTHS = {
    "يناير": 1,  "فبراير": 2,  "مارس": 3,     "أبريل": 4,
    "مايو": 5,   "يونيو": 6,   "يوليو": 7,    "أغسطس": 8,
    "سبتمبر": 9, "أكتوبر": 10, "نوفمبر": 11,  "ديسمبر": 12,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _doc_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _entry_content(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Return article content if available, fallback to full soup.
    """
    el = soup.find("article")
    if el:
        return el
    return soup


def _title(soup: BeautifulSoup) -> str:
    header = soup.find("header")
    h1 = header.find("h1") if header else soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def _is_noise_p(tag) -> bool:
    return (
        tag.name == "p"
        and any(p in tag.get_text() for p in _NOISE_PATTERNS)
    )


def _get_entry_content(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    post_inner = soup.find("div", class_="post-inner")
    if post_inner:
        return post_inner.find("div", class_="entry-content")
    return soup.find("div", class_="entry-content")


# ============================================================
# PDF EXTRACTION
# ============================================================

def ar_pdf(soup: BeautifulSoup) -> Optional[str]:
    entry = _get_entry_content(soup)
    if not entry:
        return None

    first_p = entry.find("p")
    if not first_p:
        return None

    for a in first_p.find_all("a", href=True):
        cls = a.get("class", [])
        if "pdf-link" in cls or "treaty-ar" in cls:
            href = a["href"]
            if href.endswith(".pdf") and "/en/" not in href and "EN" not in href:
                return href
    return None


def en_pdf(soup: BeautifulSoup) -> Optional[str]:
    entry = _get_entry_content(soup)
    if not entry:
        return None

    first_p = entry.find("p")
    if not first_p:
        return None

    for a in first_p.find_all("a", href=True):
        cls = a.get("class", [])
        href = a["href"]

        if "decree-link" in cls and href.endswith(".pdf"):
            return href

        if "treaty-ar" in cls and href.endswith(".pdf") and ("EN" in href or "/en/" in href):
            return href

    return None


# ============================================================
# HTML BODY EXTRACTION
# ============================================================

def html_body(soup: BeautifulSoup) -> Optional[str]:
    entry = _get_entry_content(soup)
    if not entry:
        return None

    children = list(entry.children)

    start = 0
    for i, child in enumerate(children):
        if hasattr(child, "name") and child.name == "p":
            if child.find("a", class_=["pdf-link", "decree-link", "treaty-ar"]):
                start = i + 1
            break

    body_parts = []
    for child in children[start:]:
        if hasattr(child, "name") and _is_noise_p(child):
            break
        body_parts.append(str(child))

    result = "".join(body_parts).strip()
    return result if result else None


def en_html_body(soup: BeautifulSoup) -> Optional[str]:
    # Try paywalled excerpt first, fall back to full entry content
    container = soup.find("div", class_="mepr-unauthorized-excerpt") or \
                soup.find("div", class_="entry-content")
    if not container:
        return None

    # Remove Arabic link paragraph
    ar_link = container.find("a", class_="ar-link")
    if ar_link:
        ar_link.find_parent("p").decompose()

    # Remove consolidated text link paragraph
    cl_link = container.find("a", class_="cl-link")
    if cl_link:
        cl_link.find_parent("p").decompose()

    result = container.decode_contents().strip()
    return result if result else None


# ============================================================
# DATE PARSING
# ============================================================

def _date(soup: BeautifulSoup) -> Optional[str]:
    t = soup.find("time")
    if t and t.get("datetime"):
        return t["datetime"][:10]

    date_li = soup.find("li", class_="post-date")
    if date_li:
        span = date_li.find("span", class_="meta-text")
        if span:
            text = span.get_text(strip=True)
            parts = text.split()

            if len(parts) == 3:
                day, month_ar, year = parts
                month = AR_MONTHS.get(month_ar)
                if month:
                    return f"{year}-{month:02d}-{int(day):02d}"

    return None


# ============================================================
# ISSUER EXTRACTION
# ============================================================

def _issuer_from_html(entry: BeautifulSoup) -> Optional[str]:
    tag_container = entry.select_one(".post-tags")
    if not tag_container:
        return None

    tags = [
        a.get_text(strip=True)
        for a in tag_container.select("a[rel='tag']")
        if a.get_text(strip=True)
    ]

    if not tags:
        return None

    priority_keywords = ["شرطة", "وزارة", "هيئة", "مجلس", "الحكومة", "سلطنة"]

    for kw in priority_keywords:
        for t in tags:
            if kw in t:
                return t

    return tags[0]


def _issuer(category_ar: str, entry: BeautifulSoup) -> Optional[str]:
    issuer = _issuer_from_html(entry)
    if issuer:
        return issuer
   
    return {
        "مرسوم سلطاني": "السلطان",
        "أمر سامي": "السلطان",
        "قرار وزاري": "الوزارة",
        "اتفاقية دولية": "الحكومة",
        "الجريدة الرسمية": "الجهاز المركزي للتشريع",
    }.get(category_ar)


# ============================================================
# AMENDMENTS HANDLING
# ============================================================

def _is_amendment_case(soup: BeautifulSoup, category_ar: str) -> bool:
    if not category_ar:
        return False

    is_amended_category = any(k in category_ar for k in ["معدل", "تعديل", "amended"])
    has_relation_table = _get_entry_content(soup).select_one("figure.wp-block-table") is not None

    return is_amended_category and has_relation_table


def _extract_amendment_docs(soup: BeautifulSoup) -> dict | None:
    """
    Extracts the 'issued by' and 'amended up to' document IDs
    from the relation table figure.

    Expected cells:
      cell[0]: صدر بموجب  → the original/issuing decree
      cell[1]: معدل لغاية → the latest amending document
    """
    fig = soup.select_one("figure.wp-block-table")
    if not fig:
        return None

    cells = fig.find_all("td")
    if len(cells) < 2:
        return None

    issued_by_link  = cells[0].find("a", href=True)
    amended_to_link = cells[1].find("a", href=True)

    if not issued_by_link or not amended_to_link:
        return None
    logger.info(issued_by_link["href"])
    return {
        "issued_by":  _doc_id(issued_by_link["href"]),
        "amended_to": _doc_id(amended_to_link["href"]),
    }


# ============================================================
# EN PAGE URL
# ============================================================

def en_page_url(soup: BeautifulSoup) -> Optional[str]:
    entry = _get_entry_content(soup)
    if not entry:
        return None

    a = entry.find("a", class_="decree-link")
    return a["href"] if a else None


# ============================================================
# MAIN PARSER
# ============================================================

async def parse_document_page(url_ar: str, category_ar: str) -> dict:
    try:
        doc_id = _doc_id(url_ar)
        if not doc_id:
            return None

        html_ar = await fetch(url_ar, referer="https://qanoon.om/")
        soup_ar = BeautifulSoup(html_ar, "lxml")
        entry_ar = _entry_content(soup_ar)
        # ── AMENDMENT SHORT CIRCUIT ───────────────────────────────
        if _is_amendment_case(entry_ar, category_ar):
            docs = _extract_amendment_docs(entry_ar)

            if docs:
                return AmendmentRelation(
            url_ar=url_ar,
            id=doc_id,
            source_id=docs["issued_by"],
            target_id=docs["amended_to"],
            relation_type="AMENDS"
        )
            else:
                return None

        # ── CORE EXTRACTION ────────────────────────────────────────
        issuer = _issuer(category_ar, entry_ar)

        pdf_ar_url = ar_pdf(entry_ar)
        pdf_en_url = en_pdf(entry_ar)
        title = _title(entry_ar)
    
        body = html_body(entry_ar)
        date = _date(entry_ar)

        # ── AR CONTENT ─────────────────────────────────────────────
        if body:
            content_ar = html_body_to_markdown(title, body)
        elif pdf_ar_url:
            pdf_bytes_ar = await download_pdf(pdf_ar_url)
            content_ar = pdf_to_markdown(pdf_bytes_ar, title=title, lang="AR")
        else:
            content_ar = None

        # ── EN CONTENT ─────────────────────────────────────────────
        title_en = None
        body_en = None
        en_url = en_page_url(entry_ar)

        if en_url:
            html_en = await fetch(en_url, referer="https://qanoon.om/")
            soup_en = BeautifulSoup(html_en, "lxml")
            entry_en = _entry_content(soup_en)
            body_en = en_html_body(entry_en)
            title_en = _title(entry_en)

        if body_en:
            content_en = html_body_to_markdown(title_en, body_en)
        elif pdf_en_url:
            pdf_bytes_en = await download_pdf(pdf_en_url)
            content_en = pdf_to_markdown(pdf_bytes_en, title=title_en, lang="EN")
        else:
            content_en = None
        status="SUCCESS_CONTENT"
        if (not content_en and not content_ar):
            status="EMPTY_CONTENT"
        elif content_en and not content_ar:
            status="ONLY_ENGLISH"
        elif content_ar and not content_en:
            status="ONLY_ARABIC"
        # ── RETURN ─────────────────────────────────────────────────
        document_node = ScrapedDocument(
    id=doc_id,
    title=title,
    date=date,
    document_type=category_ar,
    issuer=issuer,
    contentAr=content_ar,
    contentEn=content_en,
    status=status,
    url_ar=url_ar,
    url_en=en_url,
    pdf_url_ar=pdf_ar_url,
    pdf_url_en=pdf_en_url,

    ar_source="html" if body else ("pdf" if pdf_ar_url else None),
    en_source="html" if body_en else ("pdf" if pdf_en_url else None),
)
        return document_node
    except Exception as e:
        logger.exception(f"[PARSER FAILED] {url_ar} -> {e}")
        return None