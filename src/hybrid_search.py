# src/retrieval/hybrid_search.py
"""
Phase 7 — Hybrid Search with Multi-Stage Reranking.

Pipeline:
  Startup : Load BM25 indexes from Redis + Cross-Encoder model
  Stage 1 : Candidate Generation (K=50)
              - BM25 on EN + AR document content  (sparse)
              - Dense chunk vector search          (dense)
              - Dense topic vector search via query topic avg embedding
              → weighted merge → top K unique chunks
  Stage 2 : Topological Context Expansion
              - Batch Memgraph traversal (one query for all K chunks)
              - Redis cache on doc_id → metadata+topics (TTL=1hr)
  Stage 3 : Cross-Encoder Reranking
              - BAAI/bge-reranker-base scores (query, expanded_chunk)
              - Top final_n → Gemini synthesis

Optimizations:
  - Async query pooling  : Stage 1 searches run concurrently
  - Traversal cache      : Redis TTL=1hr on doc metadata
  - Bilingual query vec  : avg(embed(query_en), embed(query_ar))
  - BM25 null safety     : docs with null content skipped gracefully

Usage:
    python src/retrieval/hybrid_search.py
    python src/retrieval/hybrid_search.py --query "..." --top-k 50 --final-n 5
"""

import argparse
import asyncio
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import login
import numpy as np
import redis
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gqlalchemy import Memgraph
from loguru import logger
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import CrossEncoder

from src.llm_agents.translator import get_both, tokenize_en, tokenize_ar
from src.vector_ops.embedder import embed_texts

load_dotenv()
# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set.")

VECTOR_INDEX_CHUNKS = "chunk_embedding_idx"
VECTOR_INDEX_TOPICS = "topic_embedding_idx"
GEMINI_MODEL        = "gemini-3.1-flash-lite"

REDIS_BM25_EN       = "bm25:index_en"
REDIS_BM25_AR       = "bm25:index_ar"
REDIS_DOC_IDS_EN    = "bm25:doc_ids_en"
REDIS_DOC_IDS_AR    = "bm25:doc_ids_ar"
REDIS_TRAVERSAL_TTL = 3600  # 1 hour

WEIGHT_BM25         = 0.2
WEIGHT_DENSE_CHUNK  = 0.6
WEIGHT_DENSE_TOPIC  = 0.2

CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

_TOPIC_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "topics": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            min_items=1,
            max_items=7,
        )
    },
    required=["topics"],
)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
# ── Clients ───────────────────────────────────────────────────────────────────

logger.info("Connecting to Memgraph...")
mg = Memgraph(
    host=os.getenv("MEMGRAPH_HOST", "localhost"),
    port=int(os.getenv("MEMGRAPH_PORT", 7687)),
)
logger.success(
    f"Memgraph connected @ "
    f"{os.getenv('MEMGRAPH_HOST','localhost')}:"
    f"{os.getenv('MEMGRAPH_PORT',7687)}"
)

logger.info("Connecting to Redis...")
redis_bytes = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=False,
)
redis_str = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)
logger.success("Redis connected")

logger.info("Initializing Gemini client...")
gemini = genai.Client(api_key=GEMINI_API_KEY)
logger.success("Gemini client ready")

console = Console()

# ── Load BM25 indexes at startup ──────────────────────────────────────────────

def load_bm25_indexes() -> tuple[BM25Okapi, list[str], BM25Okapi, list[str]]:
    logger.info("Loading BM25 indexes from Redis...")
    t0 = time.perf_counter()

    bm25_en_bytes  = redis_bytes.get(REDIS_BM25_EN)
    bm25_ar_bytes  = redis_bytes.get(REDIS_BM25_AR)
    doc_ids_en_raw = redis_bytes.get(REDIS_DOC_IDS_EN)
    doc_ids_ar_raw = redis_bytes.get(REDIS_DOC_IDS_AR)

    if not all([bm25_en_bytes, bm25_ar_bytes, doc_ids_en_raw, doc_ids_ar_raw]):
        raise RuntimeError(
            "BM25 indexes not found in Redis. "
            "Run src/retrieval/bm25_indexer.py first."
        )

    bm25_en    = pickle.loads(bm25_en_bytes)
    bm25_ar    = pickle.loads(bm25_ar_bytes)
    doc_ids_en = json.loads(doc_ids_en_raw)
    doc_ids_ar = json.loads(doc_ids_ar_raw)

    logger.success(
        f"BM25 indexes loaded in {time.perf_counter() - t0:.3f}s — "
        f"EN: {len(doc_ids_en)} docs, AR: {len(doc_ids_ar)} docs"
    )
    return bm25_en, doc_ids_en, bm25_ar, doc_ids_ar


def load_cross_encoder() -> CrossEncoder:
    logger.info(f"Loading Cross-Encoder ({CROSS_ENCODER_MODEL})...")
    t0    = time.perf_counter()
    model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    logger.success(f"Cross-Encoder ready in {time.perf_counter() - t0:.2f}s")
    return model


logger.info("Loading models and indexes at startup...")
_bm25_en, _doc_ids_en, _bm25_ar, _doc_ids_ar = load_bm25_indexes()
_cross_encoder = load_cross_encoder()
logger.success("All models and indexes ready")

# ── Query topic extraction ────────────────────────────────────────────────────

def extract_query_topics(query: str) -> list[str]:
    """Extract legal topics from query using Gemini."""
    logger.debug(f"Extracting topics from query: {query[:80]}...")
    t0 = time.perf_counter()

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a legal analyst specializing in Omani law. "
                "Given a legal query, identify 1 to 7 core legal topics. "
                "Each topic must be a concise English noun phrase. "
                "Return between 1 and 7 topics."
            ),
            response_mime_type="application/json",
            response_schema=_TOPIC_SCHEMA,
            temperature=0.1,
        ),
    )

    topics = json.loads(response.text.strip()).get("topics", [])
    logger.debug(
        f"Query topics in {time.perf_counter() - t0:.3f}s: {topics}"
    )
    return topics


def get_query_topic_avg_embedding(query: str) -> list[float] | None:
    """Extract query topics → embed each → return avg vector."""
    topics = extract_query_topics(query)
    if not topics:
        logger.warning("No query topics extracted — skipping topic avg embedding")
        return None

    t0   = time.perf_counter()
    vecs = embed_texts(topics)
    avg  = np.mean(np.array(vecs), axis=0).tolist()
    logger.debug(
        f"Query topic avg embedding in {time.perf_counter() - t0:.3f}s "
        f"from {len(topics)} topics dim={len(avg)}"
    )
    return avg

# ── Stage 1 searches ──────────────────────────────────────────────────────────

def _normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize score array to [0, 1]. Safe against all-zero arrays."""
    max_s = scores.max()
    return scores / max_s if max_s > 0 else scores


def _bm25_search(
    query_en: str,
    query_ar: str,
    top_k: int,
) -> dict[str, float]:
    """
    BM25 sparse search on EN and AR document content.
    Handles null content gracefully — docs missing a language
    simply don't appear in that index.
    Returns {doc_id: normalized_score}.
    """
    logger.debug("BM25 search started...")
    t0 = time.perf_counter()

    tokens_en = tokenize_en(query_en)
    tokens_ar = tokenize_ar(query_ar)

    logger.debug(f"  EN tokens: {tokens_en[:10]}")
    logger.debug(f"  AR tokens: {tokens_ar[:10]}")

    scores_en = _normalize(_bm25_en.get_scores(tokens_en))
    scores_ar = _normalize(_bm25_ar.get_scores(tokens_ar))

    # Map doc_id → best score across EN and AR
    doc_scores: dict[str, float] = {}

    for doc_id, score in zip(_doc_ids_en, scores_en):
        if score > 0:
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), float(score))

    for doc_id, score in zip(_doc_ids_ar, scores_ar):
        if score > 0:
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), float(score))

    # Top K by BM25 score
    top_docs = sorted(
        doc_scores.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    elapsed = time.perf_counter() - t0
    logger.debug(
        f"BM25 done in {elapsed:.3f}s — "
        f"{len(doc_scores)} docs matched, "
        f"top: {top_docs[0] if top_docs else 'none'}"
    )
    return dict(top_docs)


def _dense_chunk_search(
    query_vec: list[float],
    top_k: int,
) -> list[dict]:
    """Dense vector search on chunk embeddings."""
    logger.debug(f"Dense chunk search: top_k={top_k}")
    t0 = time.perf_counter()

    rows = list(mg.execute_and_fetch(
        f"""
        CALL vector_search.search("{VECTOR_INDEX_CHUNKS}", {top_k}, $vec)
        YIELD node, similarity
        RETURN node.id       AS id,
               node.text     AS text,
               node.doc_id   AS doc_id,
               node.language AS lang,
               node.index    AS index,
               similarity    AS score
        ORDER BY score DESC
        """,
        parameters={"vec": query_vec},
    ))

    logger.debug(
        f"Dense chunk search done in {time.perf_counter() - t0:.3f}s — "
        f"{len(rows)} results | "
        f"top score={rows[0]['score']:.4f}" if rows else "no results"
    )
    return rows


def _dense_topic_search(
    topic_vec: list[float],
    top_k: int,
) -> dict[str, float]:
    """
    Dense search on topic embeddings using query topic avg embedding.
    Returns {doc_id: max_topic_similarity_score}.
    """
    logger.debug(f"Dense topic search: top_k={top_k}")
    t0 = time.perf_counter()

    topic_rows = list(mg.execute_and_fetch(
        f"""
        CALL vector_search.search("{VECTOR_INDEX_TOPICS}", {top_k}, $vec)
        YIELD node, similarity
        RETURN node.name AS name, similarity AS score
        ORDER BY score DESC
        """,
        parameters={"vec": topic_vec},
    ))

    if not topic_rows:
        logger.debug("Dense topic search — no topics found")
        return {}

    logger.debug(
        f"Similar topics: {[r['name'] for r in topic_rows[:5]]}"
    )

    topic_score_map = {r["name"]: r["score"] for r in topic_rows}
    topic_names     = list(topic_score_map.keys())

    # Find documents linked to these topics
    doc_rows = list(mg.execute_and_fetch(
        """
        UNWIND $topics AS topic_name
        MATCH (t:Topic {name: topic_name})<-[:HAS_TOPIC]-(d:Document)
        RETURN d.id AS doc_id, topic_name AS topic
        """,
        parameters={"topics": topic_names},
    ))

    doc_scores: dict[str, float] = {}
    for row in doc_rows:
        doc_id = row["doc_id"]
        score  = topic_score_map.get(row["topic"], 0.0)
        doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), score)

    logger.debug(
        f"Dense topic search done in {time.perf_counter() - t0:.3f}s — "
        f"{len(doc_scores)} docs matched via topics"
    )
    return doc_scores


async def _async_stage1(
    query: str,
    query_en: str,
    query_ar: str,
    query_vec: list[float],
    top_k: int,
) -> list[dict]:
    """
    Run BM25 + dense chunk + dense topic searches concurrently via ThreadPoolExecutor.
    Merge scores with weights → return top_k unique candidates.
    """
    logger.info(f"Stage 1 — async candidate generation: top_k={top_k}")
    t0       = time.perf_counter()
    loop     = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    # Get query topic avg embedding first (needed for topic search)
    topic_vec = await loop.run_in_executor(
        executor, get_query_topic_avg_embedding, query
    )
    # Fall back to query_vec if topic extraction failed
    search_topic_vec = topic_vec if topic_vec else query_vec

    # Run all three searches concurrently
    bm25_fut        = loop.run_in_executor(
        executor, _bm25_search, query_en, query_ar, top_k
    )
    dense_chunk_fut = loop.run_in_executor(
        executor, _dense_chunk_search, query_vec, top_k
    )
    dense_topic_fut = loop.run_in_executor(
        executor, _dense_topic_search, search_topic_vec, top_k
    )

    bm25_scores, chunk_results, topic_scores = await asyncio.gather(
        bm25_fut, dense_chunk_fut, dense_topic_fut
    )

    logger.debug(
        f"All searches returned in {time.perf_counter() - t0:.3f}s — "
        f"bm25={len(bm25_scores)} docs, "
        f"chunks={len(chunk_results)}, "
        f"topic_docs={len(topic_scores)}"
    )

    # Normalize dense chunk scores to [0, 1]
    if chunk_results:
        max_chunk = max(r["score"] for r in chunk_results)
        if max_chunk > 0:
            for r in chunk_results:
                r["score"] /= max_chunk

    # Normalize topic scores to [0, 1]
    if topic_scores:
        max_topic = max(topic_scores.values())
        if max_topic > 0:
            topic_scores = {k: v / max_topic for k, v in topic_scores.items()}

    # Merge scores per chunk
    merged: dict[str, dict] = {}
    for chunk in chunk_results:
        cid    = chunk["id"]
        doc_id = chunk["doc_id"]

        bm25_s  = bm25_scores.get(doc_id, 0.0)
        chunk_s = chunk["score"]
        topic_s = topic_scores.get(doc_id, 0.0)
        final_s = (
            WEIGHT_BM25        * bm25_s  +
            WEIGHT_DENSE_CHUNK * chunk_s +
            WEIGHT_DENSE_TOPIC * topic_s
        )

        merged[cid] = {
            **chunk,
            "score":       final_s,
            "bm25_score":  bm25_s,
            "chunk_score": chunk_s,
            "topic_score": topic_s,
        }

    candidates = sorted(
        merged.values(), key=lambda x: x["score"], reverse=True
    )[:top_k]

    elapsed = time.perf_counter() - t0
    if candidates:
        logger.success(
            f"Stage 1 done in {elapsed:.3f}s — "
            f"{len(candidates)} candidates | "
            f"top: doc={candidates[0]['doc_id']} score={candidates[0]['score']:.4f}"
        )
        logger.debug("Score breakdown (top 3):")
        for c in candidates[:3]:
            logger.debug(
                f"  [{c['doc_id']}] final={c['score']:.4f} "
                f"bm25={c['bm25_score']:.4f} "
                f"chunk={c['chunk_score']:.4f} "
                f"topic={c['topic_score']:.4f}"
            )
    else:
        logger.warning(f"Stage 1 done in {elapsed:.3f}s — no candidates found")

    return candidates

# ── Stage 2: Topological Context Expansion ────────────────────────────────────

def _cache_key(doc_id: str) -> str:
    return f"traversal:{doc_id}"


def _fetch_doc_metadata_batch(doc_ids: list[str]) -> dict[str, dict]:
    """Fetch metadata for multiple doc_ids in ONE Memgraph query."""
    logger.debug(f"Fetching metadata for {len(doc_ids)} docs from Memgraph...")
    t0 = time.perf_counter()

    rows = list(mg.execute_and_fetch(
        """
        UNWIND $doc_ids AS doc_id
        MATCH (d:Document {id: doc_id})
        OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
        RETURN
            d.id            AS doc_id,
            d.title         AS title,
            d.issuer        AS issuer,
            d.date          AS date,
            d.document_type AS document_type,
            collect(t.name) AS topics
        """,
        parameters={"doc_ids": doc_ids},
    ))

    result = {
        row["doc_id"]: {
            "title":         row.get("title"),
            "issuer":        row.get("issuer"),
            "date":          row.get("date"),
            "document_type": row.get("document_type"),
            "topics":        row.get("topics") or [],
        }
        for row in rows
    }

    logger.debug(
        f"Memgraph batch fetch done in {time.perf_counter() - t0:.3f}s "
        f"— {len(result)} docs"
    )
    return result


def expand_contexts(candidates: list[dict]) -> list[str]:
    """
    Stage 2: Enrich each chunk with document metadata + topics.
    Redis cache per doc_id (TTL=1hr), batch Memgraph query for misses.
    """
    logger.info(f"Stage 2 — expanding context for {len(candidates)} chunks")
    t0 = time.perf_counter()

    unique_doc_ids = list({c["doc_id"] for c in candidates})
    cache_hits:   dict[str, dict] = {}
    cache_misses: list[str]       = []

    for doc_id in unique_doc_ids:
        cached = redis_str.get(_cache_key(doc_id))
        if cached:
            cache_hits[doc_id] = json.loads(cached)
            logger.debug(f"  Cache hit : doc_id={doc_id}")
        else:
            cache_misses.append(doc_id)
            logger.debug(f"  Cache miss: doc_id={doc_id}")

    logger.info(
        f"Cache — hits: {len(cache_hits)}, misses: {len(cache_misses)}"
    )

    fetched: dict[str, dict] = {}
    if cache_misses:
        fetched = _fetch_doc_metadata_batch(cache_misses)
        pipe    = redis_str.pipeline()
        for doc_id, meta in fetched.items():
            pipe.setex(_cache_key(doc_id), REDIS_TRAVERSAL_TTL, json.dumps(meta))
        pipe.execute()
        logger.debug(f"Cached {len(fetched)} new entries in Redis (TTL={REDIS_TRAVERSAL_TTL}s)")

    all_meta = {**cache_hits, **fetched}

    expanded = []
    for c in candidates:
        meta   = all_meta.get(c["doc_id"], {})
        topics = ", ".join(meta.get("topics") or []) or "—"
        meta_str = (
            f"[Document: {meta.get('title', '?')} | "
            f"Issuer: {meta.get('issuer', '?')} | "
            f"{meta.get('date', '?')} | "
            f"{meta.get('document_type', '?')}]\n"
            f"[Topics: {topics}]\n\n"
        )
        expanded.append(meta_str + c["text"])
        logger.debug(
            f"  Expanded chunk {c['id']} — "
            f"doc={meta.get('title','?')!r} topics=[{topics[:60]}]"
        )

    logger.success(
        f"Stage 2 done in {time.perf_counter() - t0:.3f}s — "
        f"{len(expanded)} contexts expanded"
    )
    return expanded

# ── Stage 3: Cross-Encoder Reranking ─────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[dict],
    expanded: list[str],
    final_n: int,
) -> tuple[list[dict], list[str]]:
    """
    Stage 3: Score (query, expanded_chunk) pairs with Cross-Encoder.
    Returns top final_n candidates and their expanded texts.
    """
    logger.info(
        f"Stage 3 — cross-encoder reranking: "
        f"{len(candidates)} → top {final_n}"
    )
    t0     = time.perf_counter()
    pairs  = [(query, ctx) for ctx in expanded]
    scores = _cross_encoder.predict(pairs)

    logger.debug(
        f"Cross-encoder scores — "
        f"min={scores.min():.4f}, max={scores.max():.4f}, "
        f"mean={scores.mean():.4f}"
    )

    for i, (c, score) in enumerate(zip(candidates, scores)):
        c["rerank_score"] = float(score)
        logger.debug(
            f"  [{i+1:02d}] doc={c['doc_id']} "
            f"hybrid={c['score']:.4f} rerank={c['rerank_score']:.4f}"
        )

    ranked = sorted(
        zip(candidates, expanded),
        key=lambda x: x[0]["rerank_score"],
        reverse=True,
    )

    top_candidates = [c for c, _ in ranked[:final_n]]
    top_expanded   = [e for _, e in ranked[:final_n]]

    logger.success(
        f"Stage 3 done in {time.perf_counter() - t0:.3f}s — "
        f"top: doc={top_candidates[0]['doc_id']} "
        f"rerank={top_candidates[0]['rerank_score']:.4f}"
        if top_candidates else "Stage 3 — no results"
    )
    return top_candidates, top_expanded

# ── LLM Synthesis ─────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = (
    "You are a legal research assistant specializing in Omani law. "
    "Answer the user's question using ONLY the provided legal document excerpts. "
    "Cite document titles and numbers when referencing specific provisions. "
    "If the answer cannot be found in the excerpts, say so clearly."
)


def synthesize(query: str, contexts: list[str]) -> str:
    logger.info(
        f"Synthesis — model={GEMINI_MODEL}, {len(contexts)} excerpts"
    )
    t0 = time.perf_counter()

    context_block = "\n\n---\n\n".join(
        f"Excerpt {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)
    )
    total_chars = sum(len(c) for c in contexts)
    logger.debug(
        f"Context block: {len(contexts)} excerpts, ~{total_chars} chars"
    )

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Question: {query}\n\nExcerpts:\n{context_block}",
        config=types.GenerateContentConfig(
            system_instruction=_SYNTHESIS_SYSTEM,
            temperature=0.1,
        ),
    )
    answer = response.text.strip()

    logger.success(
        f"Synthesis done in {time.perf_counter() - t0:.3f}s — "
        f"{len(answer)} chars"
    )
    logger.debug(f"Answer preview: {answer[:120]}...")
    return answer

# ── Display ───────────────────────────────────────────────────────────────────

def display_candidates(
    candidates: list[dict],
    title: str = "Candidates",
) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("#",       style="bold",    width=3)
    table.add_column("Final",   style="cyan",    width=7)
    table.add_column("Rerank",  style="green",   width=7)
    table.add_column("BM25",    style="yellow",  width=6)
    table.add_column("Chunk",   style="blue",    width=6)
    table.add_column("Topic",   style="magenta", width=6)
    table.add_column("Doc ID",  style="yellow",  width=18)
    table.add_column("Lang",    width=4)
    table.add_column("Excerpt", style="white")

    for i, c in enumerate(candidates):
        table.add_row(
            str(i + 1),
            f"{c.get('score', 0):.4f}",
            f"{c.get('rerank_score', 0):.4f}",
            f"{c.get('bm25_score', 0):.3f}",
            f"{c.get('chunk_score', 0):.3f}",
            f"{c.get('topic_score', 0):.3f}",
            c["doc_id"],
            c.get("lang", "?"),
            c["text"][:150] + "...",
        )
    console.print(table)


def display_topics(doc_ids: list[str]) -> None:
    rows = mg.execute_and_fetch(
        """
        UNWIND $doc_ids AS doc_id
        MATCH (d:Document {id: doc_id})-[:HAS_TOPIC]->(t:Topic)
        RETURN t.name AS name
        """,
        parameters={"doc_ids": doc_ids},
    )
    all_topics = sorted({r["name"] for r in rows})
    logger.info(f"Related topics: {all_topics}")
    if all_topics:
        console.print(Panel(
            " • ".join(all_topics),
            title="[bold magenta]Related Topics",
            border_style="magenta",
        ))

# ── Metrics ───────────────────────────────────────────────────────────────────

def log_metrics(candidates: list[dict], final_n: int) -> None:
    """
    Log MRR, Precision@N, Recall@N.
    Uses rerank_score > 0 as proxy for relevance (no ground truth available).
    """
    threshold = 0.0
    relevant  = [c for c in candidates if c.get("rerank_score", 0) > threshold]
    top_n     = candidates[:final_n]
    top_n_rel = [c for c in top_n if c.get("rerank_score", 0) > threshold]

    mrr = 0.0
    for rank, c in enumerate(candidates, 1):
        if c.get("rerank_score", 0) > threshold:
            mrr = 1.0 / rank
            break

    precision = len(top_n_rel) / final_n   if final_n   else 0.0
    recall    = len(top_n_rel) / len(relevant) if relevant else 0.0

    logger.info("─" * 42)
    logger.info("Search Metrics")
    logger.info("─" * 42)
    logger.info(f"  MRR                : {mrr:.4f}")
    logger.info(f"  Precision@{final_n:<2}       : {precision:.4f}")
    logger.info(f"  Recall@{final_n:<2}          : {recall:.4f}")
    logger.info(f"  Total candidates   : {len(candidates)}")
    logger.info(f"  Relevant (proxy)   : {len(relevant)}")
    logger.info("─" * 42)

# ── Main search flow ──────────────────────────────────────────────────────────

def search(query: str, top_k: int = 50, final_n: int = 5) -> None:
    logger.info(
        f"Search started — query={query!r}, top_k={top_k}, final_n={final_n}"
    )
    t_total = time.perf_counter()

    console.print(Panel(
        f"[bold cyan]{query}",
        title="Hybrid Legal GraphRAG Query",
    ))

    # Translate + embed (bilingual query vector)
    with console.status("[cyan]Translating and embedding query..."):
        t0        = time.perf_counter()
        both      = get_both(query)
        query_en  = both["en"]
        query_ar  = both["ar"]

        vec_en    = np.array(embed_texts([query_en])[0])
        vec_ar    = np.array(embed_texts([query_ar])[0])
        query_vec = np.mean([vec_en, vec_ar], axis=0).tolist()

    logger.info(
        f"Translation + embedding done in {time.perf_counter() - t0:.3f}s — "
        f"EN: {query_en[:60]} | AR: {query_ar[:60]}"
    )
    console.print(
        f"[green]✓ Translation:[/] EN: {query_en[:60]} | AR: {query_ar[:60]}\n"
    )

    # Stage 1 — async candidate generation
    with console.status("[cyan]Stage 1: Candidate generation..."):
        candidates = asyncio.run(_async_stage1(
            query, query_en, query_ar, query_vec, top_k
        ))
    console.print(f"[green]✓ Stage 1:[/] {len(candidates)} candidates\n")

    if not candidates:
        logger.warning("No candidates — aborting")
        console.print("[red]No results found.[/]")
        return

    # Stage 2 — context expansion
    with console.status("[cyan]Stage 2: Context expansion..."):
        expanded = expand_contexts(candidates)
    console.print(
        f"[green]✓ Stage 2:[/] {len(expanded)} contexts expanded\n"
    )

    # Stage 3 — cross-encoder reranking
    with console.status("[cyan]Stage 3: Cross-encoder reranking..."):
        top_candidates, top_expanded = rerank(
            query, candidates, expanded, final_n
        )
    console.print(f"[green]✓ Stage 3:[/] Top {final_n} selected\n")

    display_candidates(top_candidates, title=f"Top {final_n} After Reranking")
    display_topics([c["doc_id"] for c in top_candidates])
    log_metrics(candidates, final_n)

    # Synthesis
    with console.status("[cyan]Synthesizing answer..."):
        answer = synthesize(query, top_expanded)
    console.print(Panel(answer, title="[bold green]Answer", border_style="green"))

    logger.success(
        f"Search complete — total time {time.perf_counter() - t_total:.3f}s"
    )

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid Legal GraphRAG Search")
    ap.add_argument("--query",   type=str, default=None)
    ap.add_argument("--top-k",   type=int, default=50,
                    help="Candidates from Stage 1")
    ap.add_argument("--final-n", type=int, default=5,
                    help="Chunks passed to LLM after reranking")
    args = ap.parse_args()

    logger.info(
        f"Hybrid search client starting — "
        f"top_k={args.top_k}, final_n={args.final_n}"
    )

    if args.query:
        search(args.query, args.top_k, args.final_n)
        return

    console.print(
        "[bold]Hybrid Legal GraphRAG Search[/] — type 'exit' to quit\n"
    )
    while True:
        try:
            query = input("Query > ").strip()
            if query.lower() in ("exit", "quit", "q"):
                logger.info("User exited")
                break
            if query:
                search(query, args.top_k, args.final_n)
                console.print()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break


if __name__ == "__main__":
    main()