"""
Phase 1+2 — Main crawl orchestrator.

Flow:
  1. Connect to Redis and verify connection.
  2. Fetch taxonomy sitemap → discover all category URLs dynamically.
  3. For each category, paginate listing pages → enqueue doc URLs (with category).
     Handles resume and repagination detection automatically.
  4. For each queued doc URL, parse the document page (Arabic + English),
     convert HTML → Markdown inline, save as JSON to data/sample_output/.
  5. On completion, report done / failed counts.

Usage:
  python src/scraper/crawler.py                          # full run
  python src/scraper/crawler.py --resume                 # resume after crash
  python src/scraper/crawler.py --status                 # print Redis state
  python src/scraper/crawler.py --reset                  # wipe Redis, start fresh
  python src/scraper/crawler.py --retry-failed           # retry failed docs
  python src/scraper/crawler.py --category royal_decree  # one category only
"""
from __future__ import annotations
import argparse
import asyncio
import dataclasses
import json
from pathlib import Path

from loguru import logger
from src.ingestion.queue import push_to_ingestion_queue 
from src.scraper.checkpoint import (
    get_redis,
    get_pending_urls,
    get_url_category,
    mark_done,
    mark_failed,
    done_count,
    failed_count,
    pending_count,
    get_status,
    reset_checkpoint,
    requeue_failed,
)
from src.scraper.config import DATA_OUTPUT_PATH, CONCURRENCY
from src.scraper.parser import parse_document_page
from src.scraper.utils import setup_logging, fetch_categories, discover_all


# ── Phase B: Extraction ───────────────────────────────────────────

async def extract_document(r, url: str, semaphore: asyncio.Semaphore) -> None:
    """
    Scrape one document URL under the concurrency semaphore.

    On success: save JSON → mark_done (atomic Redis pipeline)
    On failure: mark_failed (atomic Redis pipeline)

    Both outcomes remove the URL from pending so the queue
    always converges to empty.
    """
    async with semaphore:
        doc_id = url.rstrip("/").split("/")[-1]

        # Get category from Redis (stored during discovery)
        category_ar = get_url_category(r, url)

        logger.info(f"Extracting [{category_ar}]: {doc_id}")
        try:
            # parse_document_page calls html_to_markdown inline (Phase 2)
            doc = await parse_document_page(url, category_ar)
            
            # ─────────────────────────────
            # 1. HARD FAILURE
            # ─────────────────────────────
            if doc is None:
                mark_failed(r, doc_id, url)
                logger.error(f"Failed (parser returned None): {doc_id}")
                return
             # ─────────────────────────────
            # 2. EMPTY CONTENT (VALID CASE)
            # ─────────────────────────────
            if isinstance(doc, dict) and doc.get("status") == "EMPTY_CONTENT":
                mark_done(r, doc_id, url)
                logger.warning(f"Empty content: {doc_id}")
                return
            # ─────────────────────────────
            # 3. NORMAL DOCUMENT
            # ─────────────────────────────
            # Save to disk for demo purpose 
            out = Path(DATA_OUTPUT_PATH) / f"{doc.id}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(dataclasses.asdict(doc), f, ensure_ascii=False, indent=2)
            # Push to the appropriate ingestion queue
            push_to_ingestion_queue(r, doc)
            # Atomic checkpoint: done + visited + remove from pending
            mark_done(r, doc.id, url)
            logger.success(
                f"Done: {doc.id} | total done: {done_count(r)} "
                f"| pending: {pending_count(r)}"
            )

        except Exception as e:
            logger.error(f"Failed: {doc_id} — {e}")
            # Atomic checkpoint: failed + visited + remove from pending
            mark_failed(r, doc_id, url)


async def run_extraction(r ,max_urls: int | None = None) -> None:
    """
    Drain the pending queue with bounded concurrency.
    Reads all pending URLs from Redis at start.
    Each URL is processed once; mark_done/mark_failed prevents re-processing.
    """
    urls = get_pending_urls(r)

    if max_urls is not None:
        urls = urls[:max_urls]
    if not urls:
        logger.info("No pending URLs to extract.")
        return

    logger.info(f"Extracting {len(urls)} documents (concurrency={CONCURRENCY})")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    await asyncio.gather(*[
        extract_document(r, url, semaphore) for url in urls
    ])


# ── Entry point ───────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    setup_logging()
    r = get_redis()

    # Verify Redis is reachable
    try:
        r.ping()
        logger.info("Redis connection OK")
    except Exception as e:
        logger.error(f"Redis unreachable: {e}")
        logger.error("Run: docker compose up -d redis")
        return

    # ── Utility commands ──────────────────────────────────────────
    if args.status:
        status = get_status(r)
        logger.info("=== Checkpoint Status ===")
        for k, v in status.items():
            logger.info(f"  {k}: {v}")
        return

    if args.reset:
        confirm = input("Wipe ALL Redis checkpoint data? (yes/no): ")
        if confirm.strip().lower() == "yes":
            reset_checkpoint(r)
            logger.info("Checkpoint reset.")
        else:
            logger.info("Reset cancelled.")
        return

    if args.retry_failed:
        requeued = requeue_failed(r)
        logger.info(f"Requeued {requeued} failed documents.")
        if requeued == 0:
            return

    if args.resume:
        s = get_status(r)
        logger.info(
            f"Resuming — done: {s['done']}, "
            f"pending: {s['pending']}, failed: {s['failed']}"
        )
    # at the top of main(), before everything else:
    if args.test_url:
        await test_single(args.test_url, args.test_category)
        return
    # ── Phase A: Discovery ────────────────────────────────────────
    logger.info("=== Phase A: Discovery ===")

    # Discover categories dynamically from taxonomy sitemap
    all_categories = await fetch_categories()

    # Filter to one category if --category flag given
    if args.category:
        all_categories = [
            c for c in all_categories
            if c["category_key"] == args.category
        ]
        if not all_categories:
            logger.error(f"Category '{args.category}' not found in sitemap.")
            return

    # Paginate each category → enqueue doc URLs into Redis
    await discover_all(r, all_categories, max_pages=args.max_pages)

    # ── Phase B: Extraction ───────────────────────────────────────
    logger.info("=== Phase B: Extraction (+ inline Markdown conversion) ===")
    await run_extraction(r, args.max_urls)

    # ── Summary ───────────────────────────────────────────────────
    logger.info("=== Pipeline Summary ===")
    logger.info(f"  Done:    {done_count(r)}")
    logger.info(f"  Failed:  {failed_count(r)}")
    logger.info(f"  Pending: {pending_count(r)}")

    if failed_count(r) > 0:
        logger.warning(
            f"{failed_count(r)} documents failed. "
            "Run with --retry-failed to attempt them again."
        )
    else:
        logger.success("All documents scraped successfully.")
        
async def test_single(url: str, category: str = "test") -> None:
    setup_logging()
    r = get_redis()

    try:
        r.ping()
    except Exception as e:
        logger.error(f"Redis unreachable: {e}")
        return

  
    semaphore = asyncio.Semaphore(1)
    await extract_document(r, url, semaphore)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="qanoon.om scraper — Phase 1 + 2"
    )
    ap.add_argument(
    "--max-urls",
    type=int,
    default=None,
    help="Maximum number of document URLs to process (default: no limit)"
)   
    ap.add_argument(
    "--max-pages",
    type=int,
    default=None,
    help="Maximum number of pages to paginate per category (default: unlimited)"
)
    ap.add_argument("--resume",       action="store_true",
                    help="Resume from last Redis checkpoint")
    ap.add_argument("--reset",        action="store_true",
                    help="Wipe Redis checkpoint and start fresh")
    ap.add_argument("--status",       action="store_true",
                    help="Print current checkpoint state and exit")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Requeue failed documents and retry")
    ap.add_argument("--category",     type=str, default=None,
                    help="Scrape one category only (e.g. royal_decree)")
    
    ap.add_argument("--test-url",      type=str, default=None,
                help="Extract a single document URL directly (skips discovery)")
    ap.add_argument("--test-category", type=str, default="test",
                help="Category to assign to the test URL (default: 'test')")
    asyncio.run(main(ap.parse_args()))