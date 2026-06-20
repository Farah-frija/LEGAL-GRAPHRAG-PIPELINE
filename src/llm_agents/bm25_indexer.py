# src/retrieval/bm25_indexer.py
"""
BM25 Indexer — precomputes BM25 indexes from Document contentEn and contentAr.

Handles all null combinations:
  - contentEn only  → goes into EN index only
  - contentAr only  → goes into AR index only
  - both            → goes into both indexes
  - neither         → skipped

Stores in Redis:
  bm25:index_en    → serialized BM25Okapi (pickle bytes)
  bm25:index_ar    → serialized BM25Okapi (pickle bytes)
  bm25:doc_ids_en  → JSON list of doc_ids (same order as EN corpus)
  bm25:doc_ids_ar  → JSON list of doc_ids (same order as AR corpus)

Run once before starting hybrid_search.py, or re-run when new docs are added.

Usage:
    python src/retrieval/bm25_indexer.py
"""

import json
import os
import pickle
import time

import redis
from gqlalchemy import Memgraph
from loguru import logger
from rank_bm25 import BM25Okapi

from src.llm_agents.translator import tokenize_en, tokenize_ar

# ── Redis keys ────────────────────────────────────────────────────────────────

REDIS_BM25_EN     = "bm25:index_en"
REDIS_BM25_AR     = "bm25:index_ar"
REDIS_DOC_IDS_EN  = "bm25:doc_ids_en"
REDIS_DOC_IDS_AR  = "bm25:doc_ids_ar"

# ── Clients ───────────────────────────────────────────────────────────────────

mg = Memgraph(
    host=os.getenv("MEMGRAPH_HOST", "localhost"),
    port=int(os.getenv("MEMGRAPH_PORT", 7687)),
)

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=False,  # bytes for pickle
)

# ── Fetch documents ───────────────────────────────────────────────────────────

def fetch_documents() -> list[dict]:
    """Fetch all documents that have at least one content field."""
    logger.info("Fetching documents from Memgraph...")
    t0 = time.perf_counter()

    rows = list(mg.execute_and_fetch(
        """
        MATCH (d:Document)
        WHERE d.contentEn IS NOT NULL OR d.contentAr IS NOT NULL
        RETURN d.id AS id, d.contentEn AS en, d.contentAr AS ar
        """
    ))

    logger.success(
        f"Fetched {len(rows)} documents in {time.perf_counter() - t0:.3f}s"
    )
    return rows

# ── Build indexes ─────────────────────────────────────────────────────────────

def build_indexes(
    docs: list[dict],
) -> tuple[BM25Okapi, list[str], BM25Okapi, list[str]]:
    """
    Build BM25 indexes for EN and AR content separately.
    Handles null contentEn / contentAr gracefully.
    Returns (bm25_en, doc_ids_en, bm25_ar, doc_ids_ar).
    """
    logger.info(f"Building BM25 indexes from {len(docs)} documents...")
    t0 = time.perf_counter()

    corpus_en:  list[list[str]] = []
    doc_ids_en: list[str]       = []
    corpus_ar:  list[list[str]] = []
    doc_ids_ar: list[str]       = []

    skipped_both = 0
    only_en      = 0
    only_ar      = 0
    both         = 0

    for doc in docs:
        doc_id = doc["id"]
        en     = (doc.get("en") or "").strip()
        ar     = (doc.get("ar") or "").strip()

        has_en = bool(en)
        has_ar = bool(ar)

        if not has_en and not has_ar:
            logger.debug(f"  Skipping {doc_id} — both contentEn and contentAr are null")
            skipped_both += 1
            continue

        if has_en:
            tokens = tokenize_en(en)
            if tokens:
                corpus_en.append(tokens)
                doc_ids_en.append(doc_id)
                logger.debug(f"  EN {doc_id} → {len(tokens)} tokens")

        if has_ar:
            tokens = tokenize_ar(ar)
            if tokens:
                corpus_ar.append(tokens)
                doc_ids_ar.append(doc_id)
                logger.debug(f"  AR {doc_id} → {len(tokens)} tokens")

        if has_en and has_ar:
            both += 1
        elif has_en:
            only_en += 1
        else:
            only_ar += 1

    logger.info(
        f"Corpus stats — both: {both}, EN only: {only_en}, "
        f"AR only: {only_ar}, skipped: {skipped_both}"
    )

    logger.info(f"Building BM25Okapi EN ({len(corpus_en)} docs)...")
    t1     = time.perf_counter()
    bm25_en = BM25Okapi(corpus_en) if corpus_en else BM25Okapi([[]])
    logger.debug(f"  EN index built in {time.perf_counter() - t1:.3f}s")

    logger.info(f"Building BM25Okapi AR ({len(corpus_ar)} docs)...")
    t1      = time.perf_counter()
    bm25_ar = BM25Okapi(corpus_ar) if corpus_ar else BM25Okapi([[]])
    logger.debug(f"  AR index built in {time.perf_counter() - t1:.3f}s")

    logger.success(
        f"BM25 indexes built in {time.perf_counter() - t0:.3f}s — "
        f"EN: {len(corpus_en)} docs, AR: {len(corpus_ar)} docs"
    )
    return bm25_en, doc_ids_en, bm25_ar, doc_ids_ar

# ── Store in Redis ────────────────────────────────────────────────────────────

def store_indexes(
    bm25_en,
    doc_ids_en: list[str],
    bm25_ar,
    doc_ids_ar: list[str],
) -> None:
    """Serialize and store BM25 indexes + doc_id lists in Redis."""
    logger.info("Serializing and storing indexes in Redis...")
    t0 = time.perf_counter()

    bm25_en_bytes = pickle.dumps(bm25_en)
    bm25_ar_bytes = pickle.dumps(bm25_ar)

    logger.debug(
        f"Serialized sizes — "
        f"EN: {len(bm25_en_bytes)/1024:.1f}KB, "
        f"AR: {len(bm25_ar_bytes)/1024:.1f}KB"
    )

    pipe = redis_client.pipeline()
    pipe.set(REDIS_BM25_EN,    bm25_en_bytes)
    pipe.set(REDIS_BM25_AR,    bm25_ar_bytes)
    pipe.set(REDIS_DOC_IDS_EN, json.dumps(doc_ids_en).encode())
    pipe.set(REDIS_DOC_IDS_AR, json.dumps(doc_ids_ar).encode())
    pipe.execute()

    logger.success(
        f"Indexes stored in Redis in {time.perf_counter() - t0:.3f}s"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.perf_counter()
    logger.info("BM25 Indexer started")

    docs = fetch_documents()
    if not docs:
        logger.warning("No documents found — nothing to index")
        return

    bm25_en, doc_ids_en, bm25_ar, doc_ids_ar = build_indexes(docs)
    store_indexes(bm25_en, doc_ids_en, bm25_ar, doc_ids_ar)

    logger.success("─" * 50)
    logger.success("BM25 Indexing Report")
    logger.success("─" * 50)
    logger.success(f"  Total documents  : {len(docs)}")
    logger.success(f"  EN indexed       : {len(doc_ids_en)}")
    logger.success(f"  AR indexed       : {len(doc_ids_ar)}")
    logger.success(f"  Total time       : {time.perf_counter() - t_total:.2f}s")
    logger.success("─" * 50)


if __name__ == "__main__":
    main()