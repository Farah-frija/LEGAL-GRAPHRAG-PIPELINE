"""
Phase 5 — Semantic chunking of Document content (continuous worker).

Splits contentAr / contentEn into overlapping chunks of ~800 tokens.
Creates Chunk nodes linked to their parent Document:
    (Document)-[:HAS_CHUNK {language: "en"|"ar"}]->(Chunk)

Runs forever: fetches unchunked docs in batches, splits them locally,
bulk-inserts all chunks in a single UNWIND query, then sleeps.

Usage:
    python src/llm_agents/chunker.py [--interval 60] [--batch-size 100]
"""

import argparse
import os
import time
import pandas as pd
from gqlalchemy import Memgraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from src.vector_ops.embedder import embed_texts
from transformers import AutoTokenizer
mg = Memgraph(
    host=os.getenv("MEMGRAPH_HOST", "localhost"),
    port=int(os.getenv("MEMGRAPH_PORT", 7687)),
)

_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=510,
    chunk_overlap=50,
    length_function=token_len,   # ← tokens, not chars
)

# ── Memgraph helpers ──────────────────────────────────────────────────────────

def fetch_unchunked_docs(limit: int) -> list[dict]:
    rows = mg.execute_and_fetch(
        f"""
        MATCH (d:Document)
        WHERE NOT (d)-[:HAS_CHUNK]->()
          AND (d.contentAr IS NOT NULL OR d.contentEn IS NOT NULL)
        RETURN d.id AS id, d.contentAr AS ar, d.contentEn AS en
        LIMIT {limit}
        """
    )
    return list(rows)


def batch_insert_chunks(chunk_rows: list[dict]) -> None:
    if not chunk_rows:
        return

    df = pd.DataFrame(chunk_rows)
    df["embedding"] = embed_texts(df["text"].tolist())

    mg.execute(
        """
        UNWIND $rows AS row
        MERGE (c:Chunk {id: row.chunk_id})
        SET c.text      = row.text,
            c.language  = row.lang,
            c.doc_id    = row.doc_id,
            c.index     = row.index,
            c.embedding = row.embedding
        WITH c, row
        MATCH (d:Document {id: row.doc_id})
        MERGE (d)-[:HAS_CHUNK {language: row.lang}]->(c)
        """,
        parameters={"rows": df.to_dict("records")},
    )

# ── Chunking logic ────────────────────────────────────────────────────────────

def build_chunk_rows(doc: dict) -> list[dict]:
    rows  = []
    doc_id = doc["id"]

    ar = (doc.get("ar") or "").strip()
    en = (doc.get("en") or "").strip()

    # prefer Arabic; fall back to English if Arabic is absent
    lang, content = ("ar", ar) if ar else ("en", en)

    if not content:
        return rows

    for i, text in enumerate(splitter.split_text(content)):
        
        rows.append({
            "chunk_id": f"{doc_id}-{lang}-{i}",
            "text":     text,
            "lang":     lang,
            "doc_id":   doc_id,
            "index":    i,
        })
    return rows

# ── Run loop ──────────────────────────────────────────────────────────────────

def run_loop(interval: int, batch_size: int) -> None:
    logger.info(f"Chunker Worker started — interval={interval}s, batch={batch_size}")

    while True:
        try:
            docs = fetch_unchunked_docs(batch_size)

            if not docs:
                logger.info("No unchunked documents found. Sleeping...")
            else:
                logger.info(f"Chunking {len(docs)} documents...")

                chunk_rows: list[dict] = []
                failed = 0

                for doc in docs:
                    try:
                        chunk_rows.extend(build_chunk_rows(doc))
                    except Exception as e:
                        logger.warning(f"[{doc['id']}] chunking failed: {e}")
                        failed += 1

                batch_insert_chunks(chunk_rows)

                logger.success(
                    f"Inserted {len(chunk_rows)} chunks from {len(docs) - failed} docs "
                    f"({failed} failed)."
                )

        except Exception as e:
            logger.error(f"Worker iteration failed: {e}")

        time.sleep(interval)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Chunker Worker")
    parser.add_argument("--interval",   type=int, default=60,  help="Seconds between batches")
    parser.add_argument("--batch-size", type=int, default=100, help="Docs per batch")
    args = parser.parse_args()

    run_loop(args.interval, args.batch_size)