"""
Phase 2 — Content to structured Markdown converter.

Two conversion paths:
  html_to_markdown(html)     → used for qanoon.om + decree.om pages
  pdf_to_markdown(pdf_bytes) → used for data.qanoon.om PDF attachments

PDF edge cases handled:
  - Image-based PDFs (scanned): PyMuPDF extracts no text → raises ImagePDFError
  - Corrupt/unreadable PDFs:    fitz.open fails          → raises exception
  - Empty PDFs:                 no blocks at all          → raises EmptyPDFError

HTML edge cases handled:
  - decree.om paywall: body contains subscription message → raises PaywallError
  - Empty body:        detected upstream by _has_real_content() in parser.py

Legal hierarchy (HTML path):
  Site uses <h3> for article markers → markdownify produces ### correctly
  No manual promotion needed.

Legal hierarchy (PDF path):
  First text block on page 1         → # H1 (document title)
  Blocks matching _ARTICLE_PATTERN   → ## H2 (article/section markers)
  Everything else                    → paragraph
"""

from __future__ import annotations
import re
import fitz           # PyMuPDF
import httpx
import markdownify
from loguru import logger   # ✅ ADDED (ONLY ADDITION)

_STRIP_TAGS = ["script", "style", "nav", "footer",
               "iframe", "noscript", "img", "svg"]

_ARTICLE_PATTERN = re.compile(
    r"^(المادة|الفصل|البند|الفرع|Article|Chapter|Section|Clause)"
    r"\s+[\dIVXأ-ي٠-٩\u0660-\u0669()]+",
    re.IGNORECASE,
)



# ── Custom exceptions ─────────────────────────────────────────────

class PaywallError(Exception):
    """decree.om returned a subscription page instead of content."""


class ImagePDFError(Exception):
    """PDF contains only scanned images — no extractable text."""


class EmptyPDFError(Exception):
    """PDF opened successfully but contains no text blocks."""


class EncodingErrorPDF(Exception):
    """PDF text layer has broken/missing font encoding — needs OCR."""


# ══════════════════════════════════════════════════════════════════
# HTML → Markdown
# ══════════════════════════════════════════════════════════════════

def html_body_to_markdown(title: str, html: str) -> str:
    md = markdownify.markdownify(
        html,
        heading_style="ATX",
        bullets="-",
        strip=_STRIP_TAGS + ["a"],
        convert_links=False,
        newline_style="\n\n",
    )

    has_h2 = bool(re.search(r"^## ", md, flags=re.MULTILINE))
    has_h3 = bool(re.search(r"^### ", md, flags=re.MULTILINE))

    if not has_h2 and has_h3:
        md = re.sub(r"^### ", "## ", md, flags=re.MULTILINE)
        for level in range(4, 6):
            pattern = "#" * level
            replacement = "#" * (level - 1)
            md = re.sub(rf"^{pattern} ", f"{replacement} ", md, flags=re.MULTILINE)

    md = _clean(md)
    return f"# {title}\n\n{md}"


# ══════════════════════════════════════════════════════════════════
# PDF download
# ══════════════════════════════════════════════════════════════════

async def download_pdf(url: str) -> bytes | None:
   
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; LegalScraper/1.0)",
                    "Referer": "https://qanoon.om/",
                },
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not resp.content.startswith(b"%PDF"):
                logger.warning(f"[PDF DOWNLOAD INVALID] {url}")
                return None

            return resp.content



# ══════════════════════════════════════════════════════════════════
# PDF → Markdown
# ══════════════════════════════════════════════════════════════════

def _is_garbled_arabic(text: str) -> bool:
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic_count = sum(1 for c in letters if "\u0600" <= c <= "\u06FF")
    return (arabic_count / len(letters)) < 0.15
def _is_garbled_english(text: str) -> bool:
    if not text:
        return False

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False

    # count Latin characters (English + general Latin scripts)
    latin_count = sum(
        1 for c in letters
        if (
            "A" <= c <= "Z" or
            "a" <= c <= "z"
        )
    )

    ratio = latin_count / len(letters)

    # if too little Latin content → likely broken / not real English
    return ratio < 0.15

def pdf_to_markdown(pdf_bytes: bytes, title: str = "", lang: str = "") -> str | None:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        lines = []
        total_pages = len(doc)
        image_only_pages = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            blocks = page.get_text("blocks", sort=True)

            text_blocks = [b for b in blocks if b[6] == 0]
            image_blocks = [b for b in blocks if b[6] == 1]

            if not text_blocks and image_blocks:
                image_only_pages += 1
                continue

            for block in text_blocks:
                text = block[4].strip()
                if not text:
                    continue

                if _ARTICLE_PATTERN.match(text):
                    lines.append(f"\n## {text}")
                else:
                    lines.append(text)

        doc.close()

        if image_only_pages == total_pages:
            raise ImagePDFError("All pages are scanned images")

        if not lines:
            raise EmptyPDFError("No extractable text found")

        raw = "\n\n".join(lines)

        if lang == "AR" and _is_garbled_arabic(raw):
            raise EncodingErrorPDF("Broken Arabic encoding detected")
        elif lang=="EN" and _is_garbled_english(raw):
            raise EncodingErrorPDF("Broken English encoding detected")

        md = _clean(raw)

        if title:
            md = f"# {title}\n\n{md}"

        return md

    except (ImagePDFError, EmptyPDFError, EncodingErrorPDF) as e:
        logger.warning(f"[PDF SKIPPED] {e}")
        return None

 
        


# ══════════════════════════════════════════════════════════════════
# Shared post-processing
# ══════════════════════════════════════════════════════════════════

def _clean(md: str) -> str:
    md = re.sub(r"\n{3,}", "\n\n", md)
    lines = [ln.rstrip() for ln in md.splitlines()]
    lines = [ln for ln in lines if not re.fullmatch(r"[-_*]{3,}", ln)]
    return "\n".join(lines).strip()