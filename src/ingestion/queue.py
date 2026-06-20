"""
Ingestion queue helpers

Design:
- Single unified Redis list for all events
- Messages are removed ONLY when successfully ingested
- Failed messages are requeued (retry later)
- Event type ("node" | "edge") is embedded in the envelope
"""

import dataclasses
import json

from loguru import logger

from src.scraper.models import AmendmentRelation, ScrapedDocument

# ── Redis key ──────────────────────────────────────────────
QUEUE = "ingestion:docs"


# ── PUSHING TO QUEUE ───────────────────────────────────────
def push_to_ingestion_queue(r, obj):
    """
    Push either a node or edge event into the unified Redis queue.
    """

    # ── EDGE ───────────────────────────────────────────────
    if isinstance(obj, AmendmentRelation):
        payload = {
            "type": "edge",
            "payload": {
                "id": obj.id,
                "source_id": obj.source_id,
                "target_id": obj.target_id,
                "relation_type": obj.relation_type,
            },
        }
        logger.info(f"Pushing EDGE to {QUEUE}: {payload}")
        r.rpush(QUEUE, json.dumps(payload))
        return

    # ── NODE ───────────────────────────────────────────────
    if isinstance(obj, ScrapedDocument):
        payload = {
            "type": "node",
            "payload": dataclasses.asdict(obj),
        }
        r.rpush(QUEUE, json.dumps(payload))
        return

    logger.warning(f"Unknown object type passed to queue: {type(obj)}")


# ── HELPERS ────────────────────────────────────────────────

def queue_length(r):
    """Debug helper: inspect queue size"""
    return {"events": r.llen(QUEUE)}


def clear_queue(r):
    """Dangerous: wipes the queue"""
    r.delete(QUEUE)