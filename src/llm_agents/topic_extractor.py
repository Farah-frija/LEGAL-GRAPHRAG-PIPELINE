"""
Phase 4 — LLM-driven topic extraction (Gemini generate_content + batch Memgraph insert).
Continuous background worker: fetches untopiced docs, calls Gemini one-by-one
respecting rate limits, collects successful results, then bulk-inserts into Memgraph.
"""

import argparse
import json
import os
import signal
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gqlalchemy import Memgraph
from loguru import logger
from src.vector_ops.embedder import embed_texts
import redis

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)

EMBEDDED_TOPICS_KEY = "embedded_topics"  # Redis Set
MODEL_NAME          = "gemini-3.1-flash-lite"
RPM_LIMIT           = 30
DELAY_S             = 60 / RPM_LIMIT   # ~2s between calls

_SYSTEM_INSTRUCTION = (
    "You are a legal analyst specializing in Omani law. "
    "Given a legal document excerpt, identify 1 to 7 core legal topics or thematic entities. "
    "Each topic must be a concise English noun phrase "
    "(e.g. 'Omanization', 'Judicial Fees', 'Labor Rights', 'Maritime Law', 'Taxation'). "
    "Return between 1 and 7 topics."
)

_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "topics": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.STRING,
                description="A concise English legal topic noun phrase",
            ),
            min_items=1,
            max_items=7,
        )
    },
    required=["topics"],
)

# ── Clients ───────────────────────────────────────────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)

mg = Memgraph(
    host=os.getenv("MEMGRAPH_HOST", "localhost"),
    port=int(os.getenv("MEMGRAPH_PORT", 7687)),
)

# ── Memgraph helpers ──────────────────────────────────────────────────────────

def fetch_untopiced_docs(limit: int) -> list[dict]:
    rows = mg.execute_and_fetch(
        f"""
        MATCH (d:Document)
        WHERE NOT (d)-[:HAS_TOPIC]->()
          AND (d.contentEn IS NOT NULL OR d.contentAr IS NOT NULL)
        RETURN d.id AS id, d.contentEn AS en, d.contentAr AS ar
        LIMIT {limit}
        """
    )
    return list(rows)


def batch_link_topics(results: dict[str, list[str]]) -> None:
    if not results:
        return

    pairs = [
        {"doc_id": doc_id, "name": topic}
        for doc_id, topics in results.items()
        for topic in topics
    ]

    unique_names = list({p["name"] for p in pairs})

    new_names = [
        name for name in unique_names
        if not redis_client.sismember(EMBEDDED_TOPICS_KEY, name)
    ]

    # Build embedding lookup only for truly new topics
    embedding_map: dict[str, list[float]] = {}
    if new_names:
        embeddings = embed_texts(new_names)

        # Normalize to plain Python lists (embed_texts may return numpy arrays)
        embedding_map = {
            name: emb.tolist() if hasattr(emb, "tolist") else list(emb)
            for name, emb in zip(new_names, embeddings)
        }

        if len(embedding_map) != len(new_names):
            logger.error(
                f"Embedding count mismatch: got {len(embedding_map)}, "
                f"expected {len(new_names)} — skipping Redis cache update"
            )
        else:
            logger.debug(
                f"Embedded {len(embedding_map)} new topics, "
                f"dim={len(next(iter(embedding_map.values())))}"
            )
            # Only cache after confirmed successful embedding
            redis_client.sadd(EMBEDDED_TOPICS_KEY, *new_names)

    topic_params = [
        {
            "name": name,
            "embedding": embedding_map.get(name),  # None for cached topics
        }
        for name in unique_names
    ]

    # Upsert topics — only SET embedding when we actually have a fresh one
    mg.execute(
        """
        UNWIND $topics AS t
        MERGE (topic:Topic {name: t.name})
        WITH topic, t
        WHERE t.embedding IS NOT NULL
        SET topic.embedding = t.embedding
        """,
        parameters={"topics": topic_params},
    )

    # Link documents to topics separately so a missing doc doesn't block topic creation
    mg.execute(
        """
        UNWIND $pairs AS pair
        MATCH (d:Document {id: pair.doc_id})
        MATCH (t:Topic {name: pair.name})
        MERGE (d)-[:HAS_TOPIC]->(t)
        """,
        parameters={"pairs": pairs},
    )


# ── Gemini single call ────────────────────────────────────────────────────────

def extract_topics(doc: dict) -> list[str]:
    """Call Gemini for one document. Returns list of topics, or [] on failure."""
    content = (doc.get("en") or doc.get("ar") or "").strip()
    if not content:
        return []

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=content,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    raw    = response.text.strip()
    parsed = json.loads(raw)
    topics = [t.strip() for t in parsed.get("topics", []) if isinstance(t, str) and t.strip()]
    return topics[:7]


# ── Run loop ──────────────────────────────────────────────────────────────────

def run_loop(interval: int, batch_size: int) -> None:
    logger.info(
        f"Topic Extraction Worker started — model={MODEL_NAME}, "
        f"interval={interval}s, batch={batch_size}, delay={DELAY_S:.2f}s/req"
    )

    while True:
        try:
            docs = fetch_untopiced_docs(batch_size)

            if not docs:
                logger.info("No untopiced documents found. Sleeping...")
            else:
                logger.info(f"Processing {len(docs)} documents (one-by-one, ~{DELAY_S:.1f}s apart)...")

                results: dict[str, list[str]] = {}

                for i, doc in enumerate(docs):
                    doc_id = doc["id"]
                    try:
                        topics = extract_topics(doc)
                        if topics:
                            results[doc_id] = topics
                            logger.debug(f"[{i+1}/{len(docs)}] {doc_id} → {topics}")
                        else:
                            logger.warning(f"[{i+1}/{len(docs)}] {doc_id} → no topics extracted")
                    except Exception as e:
                        logger.warning(f"[{i+1}/{len(docs)}] {doc_id} failed: {e}")

                    # respect rate limit between calls (skip delay after last doc)
                    if i < len(docs) - 1:
                        time.sleep(DELAY_S)

                # bulk insert all successful results
                batch_link_topics(results)

                skipped = len(docs) - len(results)
                logger.success(
                    f"Batch done — linked topics for {len(results)} docs, "
                    f"{skipped} skipped/failed."
                )

        except Exception as e:
            logger.error(f"Worker iteration failed: {e}")

        time.sleep(interval)


# ── Entry point ───────────────────────────────────────────────────────────────

def shutdown(sig, frame):
    logger.info("Shutting down — clearing embedded topics cache...")
    redis_client.delete(EMBEDDED_TOPICS_KEY)
    logger.success("Cache cleared.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Topic Extractor Worker")
    parser.add_argument("--interval",   type=int, default=20, help="Seconds between batches")
    parser.add_argument("--batch-size", type=int, default=1,  help="Docs per batch")
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    run_loop(args.interval, args.batch_size)