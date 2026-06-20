from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union


# ─────────────────────────────────────────────
# NODE (Document output exactly like parser dict)
# ─────────────────────────────────────────────

@dataclass
class ScrapedDocument:
    

    # Identity
    id: str

    # Metadata
    title: Optional[str] = None
    date: Optional[str] = None
    document_type: Optional[str] = None   # category_ar
    issuer: Optional[str] = None
    url_ar: Optional[str] = None
    url_en: Optional[str] = None
    pdf_url_ar: Optional[str] = None
    pdf_url_en: Optional[str] = None

    # ✅ provenance (VERY useful for debugging / graph QA)
    ar_source: Optional[str] = None   # "html" | "pdf"
    en_source: Optional[str] = None   # "html" | "pdf"
    # Content (MATCHING YOUR KEYS EXACTLY)
    contentAr: Optional[str] = None
    contentEn: Optional[str] = None
    type: str = "NODE"
    # Status (same logic as parser)
    status: str = "SUCCESS_CONTENT"
    # SUCCESS_CONTENT | EMPTY_CONTENT | ONLY_ENGLISH | ONLY_ARABIC


# ─────────────────────────────────────────────
# EDGE (amendment relation unchanged)
# ─────────────────────────────────────────────

@dataclass
class AmendmentRelation:
    id: str
    source_id: str
    target_id: str
    relation_type: str = "AMENDS"
    type: str = "EDGE"
    url_ar: Optional[str] = None


# ─────────────────────────────────────────────
# Unified parser output
# ─────────────────────────────────────────────

ParserOutput = Union[ScrapedDocument, AmendmentRelation]