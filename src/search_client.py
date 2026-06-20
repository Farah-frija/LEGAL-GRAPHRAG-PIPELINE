"""
Phase 6 — Hybrid Search CLI.

3-stage retrieval:
  Stage 1 : Dense vector search via Memgraph native vector index (top-K)
  Stage 2 : Graph traversal — Chunk → Document → Topics (context expansion)
  Stage 3 : LLM synthesis using Gemini

Usage:
    python src/search_client.py
    python src/search_client.py --query "What are the rules for foreign investment?"
    python src/search_client.py --top-k 5
"""

import argparse
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from gqlalchemy import Memgraph
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.vector_ops.embedder import embed_texts

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────

logger.info("Connecting to Memgraph...")
mg = Memgraph(
    host=os.getenv("MEMGRAPH_HOST", "localhost"),
    port=int(os.getenv("MEMGRAPH_PORT", 7687)),
)
logger.success(f"Memgraph connected @ {os.getenv('MEMGRAPH_HOST', 'localhost')}:{os.getenv('MEMGRAPH_PORT', 7687)}")

logger.info("Initializing Gemini client...")
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
logger.success("Gemini client ready.")

console = Console()

VECTOR_INDEX_CHUNKS = "chunk_embedding_idx"
GEMINI_MODEL        = "gemini-3.1-flash-lite"

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    logger.debug(f"Embedding query ({len(query)} chars): {query[:80]}...")
    t0  = time.perf_counter()
    vec = embed_texts([query])[0]
    logger.debug(f"Query embedded in {time.perf_counter() - t0:.3f}s — dim={len(vec)}")
    return vec

# ── Stage 1: Dense vector search ─────────────────────────────────────────────

def vector_search(query_vec: list[float], top_k: int) -> list[dict]:
    logger.info(f"Stage 1 — vector search: index={VECTOR_INDEX_CHUNKS}, top_k={top_k}")
    t0 = time.perf_counter()

    rows = mg.execute_and_fetch(
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
    )
    results = list(rows)

    elapsed = time.perf_counter() - t0
    logger.success(f"Stage 1 done — {len(results)} candidates in {elapsed:.3f}s")

    if results:
        logger.debug(f"Score range: {results[-1]['score']:.4f} – {results[0]['score']:.4f}")
        for i, r in enumerate(results[:3]):
            logger.debug(f"  [{i+1}] doc_id={r['doc_id']} score={r['score']:.4f} lang={r.get('lang','?')}")

    return results

# ── Stage 2: Topological context expansion ────────────────────────────────────

def expand_context(chunk: dict) -> str:
    """
    Traverse: Chunk → parent Document → linked Topics.
    Prepends structured metadata to the chunk text.
    """
    logger.debug(f"Expanding context for chunk id={chunk['id']} doc_id={chunk['doc_id']}")

    rows = list(mg.execute_and_fetch(
        """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
        RETURN
               d.title         AS title,
               d.date          AS date,
               d.document_type AS document_type,
               d.issuer        AS issuer,
               collect(t.name) AS topics
        """,
        parameters={"doc_id": chunk["doc_id"]},
    ))

    if not rows:
        logger.warning(f"No document found for doc_id={chunk['doc_id']} — returning raw chunk text")
        return chunk["text"]

    row    = rows[0]
    title  = row.get("title")
    topics = ", ".join(row.get("topics") or []) or "—"

    logger.debug(
        f"  doc={title!r} | issuer={row.get('issuer','?')} | "
        f"date={row.get('date','?')} | topics=[{topics}]"
    )

    meta = (
        f"[Document: {title} | Issuer: {row.get('issuer', '?')} | "
        f"{row.get('date', '?')} | {row.get('document_type', '?')}]\n"
        f"[Topics: {topics}]\n\n"
    )
    return meta + chunk["text"]


def display_topics(doc_ids: list[str]) -> None:
    """Fetch and display all unique topics for a list of doc_ids in one query."""
    logger.debug(f"Fetching topics for {len(doc_ids)} doc_ids in one query")

    rows = mg.execute_and_fetch(
        """
        UNWIND $doc_ids AS doc_id
        MATCH (d:Document {id: doc_id})-[:HAS_TOPIC]->(t:Topic)
        RETURN t.name AS name
        """,
        parameters={"doc_ids": doc_ids},
    )
    all_topics = sorted({r["name"] for r in rows})
    logger.info(f"Total unique topics: {len(all_topics)}")

    if all_topics:
        console.print(Panel(
            " • ".join(all_topics),
            title="[bold magenta]Related Topics",
            border_style="magenta",
        ))
    else:
        logger.warning("No topics found for any of the retrieved chunks")

# ── Stage 3: LLM synthesis ────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = (
    "You are a legal research assistant specializing in Omani law. "
    "Answer the user's question using ONLY the provided legal document excerpts. "
    "Cite document titles and numbers when referencing specific provisions. "
    "If the answer cannot be found in the excerpts, say so clearly."
)

def synthesize(query: str, contexts: list[str]) -> str:
    logger.info(f"Stage 3 — synthesizing answer: model={GEMINI_MODEL}, {len(contexts)} excerpts")
    t0 = time.perf_counter()

    context_block = "\n\n---\n\n".join(
        f"Excerpt {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)
    )
    total_chars = sum(len(c) for c in contexts)
    logger.debug(f"Context block: {len(contexts)} excerpts, ~{total_chars} chars total")

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Question: {query}\n\nExcerpts:\n{context_block}",
        config=types.GenerateContentConfig(
            system_instruction=_SYNTHESIS_SYSTEM,
            temperature=0.1,
        ),
    )
    answer = response.text.strip()

    elapsed = time.perf_counter() - t0
    logger.success(f"Stage 3 done — {len(answer)} chars synthesized in {elapsed:.3f}s")
    logger.debug(f"Answer preview: {answer[:120]}...")

    return answer

# ── Display helpers ───────────────────────────────────────────────────────────

def display_candidates(candidates: list[dict]) -> None:
    logger.debug(f"Displaying {len(candidates)} candidates")
    table = Table(title=f"Top {len(candidates)} Matching Chunks", show_lines=True)
    table.add_column("#",       style="bold", width=3)
    table.add_column("Score",   style="cyan", width=7)
    table.add_column("Doc ID",  style="yellow", width=20)
    table.add_column("Lang",    width=5)
    table.add_column("Excerpt", style="white")

    for i, c in enumerate(candidates):
        table.add_row(
            str(i + 1),
            f"{c['score']:.4f}",
            c["doc_id"],
            c.get("lang", "?"),
            c["text"][:200] + "...",
        )
    console.print(table)

# ── Main search flow ──────────────────────────────────────────────────────────

def search(query: str, top_k: int = 5) -> None:
    logger.info(f"Search started — query={query!r}, top_k={top_k}")
    t_total = time.perf_counter()

    console.print(Panel(f"[bold cyan]{query}", title="Legal GraphRAG Query"))

    # Stage 1 — dense retrieval
    with console.status("[cyan]Stage 1: Vector search..."):
        query_vec  = embed_query(query)
        candidates = vector_search(query_vec, top_k)
    console.print(f"[green]✓ Stage 1:[/] {len(candidates)} candidates retrieved\n")

    if not candidates:
        logger.warning("No candidates returned from vector search — aborting")
        console.print("[red]No results found.[/]")
        return

    display_candidates(candidates)

    # Stage 2 — graph traversal
    logger.info(f"Stage 2 — expanding context for {len(candidates)} chunks")
    t2 = time.perf_counter()
    with console.status("[cyan]Stage 2: Graph traversal..."):
        expanded = [expand_context(c) for c in candidates]
        doc_ids  = list({c["doc_id"] for c in candidates})
    logger.success(f"Stage 2 done in {time.perf_counter() - t2:.3f}s")
    console.print(f"[green]✓ Stage 2:[/] Context expanded for {len(candidates)} chunks\n")
    display_topics(doc_ids)

    # Stage 3 — synthesis
    with console.status("[cyan]Stage 3: Synthesizing answer..."):
        answer = synthesize(query, expanded)
    console.print(f"\n[green]✓ Stage 3:[/] Answer synthesized\n")
    console.print(Panel(answer, title="[bold green]Answer", border_style="green"))

    logger.success(f"Search complete — total time {time.perf_counter() - t_total:.3f}s")

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Legal GraphRAG Search Client")
    ap.add_argument("--query",  type=str, default=None, help="Query string")
    ap.add_argument("--top-k",  type=int, default=5,    help="Chunks to retrieve and pass to LLM")
    args = ap.parse_args()

    logger.info(f"Search client starting — top_k={args.top_k}")

    if args.query:
        search(args.query, args.top_k)
        return

    console.print("[bold]Legal GraphRAG Search[/] — type 'exit' to quit\n")
    while True:
        try:
            query = input("Query > ").strip()
            if query.lower() in ("exit", "quit", "q"):
                logger.info("User exited search CLI")
                break
            if query:
                search(query, args.top_k)
                console.print()
        except KeyboardInterrupt:
            logger.info("Search CLI interrupted by user")
            break

if __name__ == "__main__":
    main()