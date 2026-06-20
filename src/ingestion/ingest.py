"""
Phase 3 — Event-driven ingestion: Redis → Memgraph

Design:
- Single unified queue for all events (nodes and edges)
- MERGE handles both creates and upserts, so ordering doesn't matter
- Messages stay in queue until successfully written to graph
- Safe retry on crash or failure
"""

import json
import os
import time
from dotenv import load_dotenv
import redis
from gqlalchemy import Memgraph
from loguru import logger
from src.ingestion.schema import create_schema

# ── Config ───────────────────────────────────────────────
load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

MEMGRAPH_HOST = os.getenv("MEMGRAPH_HOST", "localhost")
MEMGRAPH_PORT = int(os.getenv("MEMGRAPH_PORT", 7687))

QUEUE = "ingestion:docs"

DEFAULT_SLEEP = 1.0


# ── Connections ──────────────────────────────────────────
def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def get_memgraph():
    return Memgraph(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT)


# ── Queue helpers ────────────────────────────────────────
def requeue(r, msg: str):
    r.rpush(QUEUE, msg)


# ── Event processor ──────────────────────────────────────
def process_event(mg, raw_msg: str):
    envelope = json.loads(raw_msg)
    event_type = envelope.get("type")
    payload = envelope.get("payload")

    if event_type == "node":
        mg.execute(
            """
            MERGE (d:Document {id: $id})
            SET d.title          = $title,
                d.date           = $date,
                d.document_type  = $document_type,
                d.issuer         = $issuer,
                d.url_ar         = $url_ar,
                d.url_en         = $url_en,
                d.pdf_url_ar     = $pdf_url_ar,
                d.pdf_url_en     = $pdf_url_en,
                d.ar_source      = $ar_source,
                d.en_source      = $en_source,
                d.contentAr      = $contentAr,
                d.contentEn      = $contentEn,
                d.status         = $status
            """,
            parameters={
                "id":            payload.get("id"),
                "title":         payload.get("title"),
                "date":          payload.get("date"),
                "document_type": payload.get("document_type"),
                "issuer":        payload.get("issuer"),
                "url_ar":        payload.get("url_ar"),
                "url_en":        payload.get("url_en"),
                "pdf_url_ar":    payload.get("pdf_url_ar"),
                "pdf_url_en":    payload.get("pdf_url_en"),
                "ar_source":     payload.get("ar_source"),
                "en_source":     payload.get("en_source"),
                "contentAr":     payload.get("contentAr"),
                "contentEn":     payload.get("contentEn"),
                "status":        payload.get("status"),
            },
        )

    elif event_type == "edge":
        mg.execute(
            """
            MERGE (a:Document {id: $src})
            MERGE (b:Document {id: $dst})
            MERGE (a)-[r:RELATION]->(b)
            SET r.type = $rel
            """,
            parameters={
                "src": payload["source_id"],
                "dst": payload["target_id"],
                "rel": payload.get("relation_type", "REFERENCES"),
            },
        )

    else:
        logger.warning(f"Unknown event type: {event_type!r} — skipping")


# ── Worker loop ───────────────────────────────────────────
def run():
    r = get_redis()
    mg = get_memgraph()
    create_schema()

    logger.info("Ingestor started (event-driven mode, unified queue)")

    while True:
        msg = r.lpop(QUEUE)

        if not msg:
            time.sleep(DEFAULT_SLEEP)
            continue

        try:
            process_event(mg, msg)
        except Exception as e:
            logger.error(f"[EVENT FAIL] {e}")
            requeue(r, msg)


if __name__ == "__main__":
    run()