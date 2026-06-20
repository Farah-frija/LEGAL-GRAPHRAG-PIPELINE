"""
Redis-backed checkpoint manager.

Key layout:
  qanoon:visited_urls        SET    every URL fetched (listing + doc)
  qanoon:pending_urls        LIST   doc URLs queued for extraction
  qanoon:url_category        HASH   { doc_url → category_ar }
  qanoon:done_doc_ids        SET    successfully scraped doc IDs
  qanoon:failed_doc_ids      SET    failed after all retries
  qanoon:category_progress   HASH   { category_url → last fully processed page }
  qanoon:last_updated        STRING ISO timestamp of last write
"""
from __future__ import annotations
from datetime import datetime, timezone

import redis as redis_lib

from src.scraper.config import REDIS_HOST, REDIS_PORT, REDIS_DB

_PFX           = "qanoon:"
K_VISITED      = f"{_PFX}visited_urls"
K_PENDING      = f"{_PFX}pending_urls"
K_URL_CAT      = f"{_PFX}url_category"
K_DONE         = f"{_PFX}done_doc_ids"
K_FAILED       = f"{_PFX}failed_doc_ids"
K_CAT_PROGRESS = f"{_PFX}category_progress"
K_UPDATED      = f"{_PFX}last_updated"
K_CAT_STATE = f"{_PFX}category_state"

def get_redis() -> redis_lib.Redis:
    return redis_lib.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
    )


# ── visited_urls ──────────────────────────────────────────────────

def is_visited(r: redis_lib.Redis, url: str) -> bool:
    """O(1) — have we already fetched this URL?"""
    return bool(r.sismember(K_VISITED, url))


def mark_visited(r: redis_lib.Redis, url: str) -> None:
    r.sadd(K_VISITED, url)


def visited_count(r: redis_lib.Redis) -> int:
    return r.scard(K_VISITED)


# ── pending_urls ──────────────────────────────────────────────────

def enqueue_urls(r: redis_lib.Redis,
                  url_cat_pairs: list[tuple[str, str]]) -> int:
    """
    Add (url, category_ar) pairs to the pending queue.
    Skips URLs already visited or already pending — handles
    new document additions automatically with no extra logic.
    Returns number of genuinely new URLs added.
    """
    if not url_cat_pairs:
        return 0

    existing_pending = set(r.lrange(K_PENDING, 0, -1))
    new_pairs = [
        (url, cat) for url, cat in url_cat_pairs
        if not r.sismember(K_VISITED, url)
        and url not in existing_pending
    ]
    if not new_pairs:
        return 0

    pipe = r.pipeline()
    for url, cat in new_pairs:
        pipe.rpush(K_PENDING, url)
        pipe.hset(K_URL_CAT, url, cat)
    pipe.execute()
    return len(new_pairs)


def get_pending_urls(r: redis_lib.Redis) -> list[str]:
    return r.lrange(K_PENDING, 0, -1)


def pending_count(r: redis_lib.Redis) -> int:
    return r.llen(K_PENDING)


def get_url_category(r: redis_lib.Redis, url: str) -> str:
    return r.hget(K_URL_CAT, url) or ""


# ── done_doc_ids ──────────────────────────────────────────────────

def mark_done(r: redis_lib.Redis, doc_id: str, url: str) -> None:
    """Atomic pipeline: done + visited + remove from pending."""
    pipe = r.pipeline()
    pipe.sadd(K_DONE, doc_id)
    pipe.sadd(K_VISITED, url)
    pipe.lrem(K_PENDING, 1, url)
    pipe.set(K_UPDATED, _now())
    pipe.execute()


def is_done(r: redis_lib.Redis, doc_id: str) -> bool:
    return bool(r.sismember(K_DONE, doc_id))


def done_count(r: redis_lib.Redis) -> int:
    return r.scard(K_DONE)


# ── failed_doc_ids ────────────────────────────────────────────────

def mark_failed(r: redis_lib.Redis, doc_id: str, url: str) -> None:
    """Atomic pipeline: failed + visited + remove from pending."""
    pipe = r.pipeline()
    pipe.sadd(K_FAILED, doc_id)
    pipe.sadd(K_VISITED, url)
    pipe.lrem(K_PENDING, 1, url)
    pipe.set(K_UPDATED, _now())
    pipe.execute()


def is_failed(r: redis_lib.Redis, doc_id: str) -> bool:
    return bool(r.sismember(K_FAILED, doc_id))


def failed_count(r: redis_lib.Redis) -> int:
    return r.scard(K_FAILED)


def requeue_failed(r: redis_lib.Redis) -> int:
    """
    Move all failed docs back to pending for a retry pass.
    Clears failed set, removes URLs from visited, re-adds to pending.
    Returns number requeued.
    """
    failed_ids = r.smembers(K_FAILED)
    if not failed_ids:
        return 0

    # Find URLs for failed doc IDs by scanning url_category hash
    all_url_cats = r.hgetall(K_URL_CAT)
    requeued = 0
    pipe = r.pipeline()
    for url, cat in all_url_cats.items():
        doc_id = url.rstrip("/").split("/")[-1]
        if doc_id in failed_ids:
            pipe.srem(K_FAILED,  doc_id)  # remove from failed
            pipe.srem(K_VISITED, url)      # allow re-processing
            pipe.rpush(K_PENDING, url)     # back to queue
            requeued += 1
    pipe.set(K_UPDATED, _now())
    pipe.execute()
    return requeued


# ── category_progress ─────────────────────────────────────────────
def set_category_state(r: redis_lib.Redis, category_url: str, state: str) -> None:
    r.hset(K_CAT_STATE, category_url, state)


def get_category_state(r: redis_lib.Redis, category_url: str) -> str:
    return r.hget(K_CAT_STATE, category_url) or "DISCOVERING"


def is_discovery_done(r: redis_lib.Redis, category_url: str) -> bool:
    return get_category_state(r, category_url) == "DISCOVERY_DONE"
def save_category_progress(r: redis_lib.Redis,
                            category_url: str, page: int) -> None:
    """
    Save the last FULLY processed listing page for this category.
    Called ONLY after all doc URLs from a page are enqueued —
    never mid-page. This is what the backwards walk reads on resume.
    """
    r.hset(K_CAT_PROGRESS, category_url, page)


def get_category_progress(r: redis_lib.Redis,
                           category_url: str) -> int:
    """
    Return last fully processed page number, or 0 if never started.
    0 means first run — start from page 1.
    """
    val = r.hget(K_CAT_PROGRESS, category_url)
    return int(val) if val else 0


# ── Status / reset ────────────────────────────────────────────────

def get_status(r: redis_lib.Redis) -> dict:
    return {
        "visited":      visited_count(r),
        "pending":      pending_count(r),
        "done":         done_count(r),
        "failed":       failed_count(r),
        "last_updated": r.get(K_UPDATED) or "never",
    }


def reset_checkpoint(r: redis_lib.Redis) -> None:
    """Delete ALL qanoon:* keys. Cannot be undone."""
    keys = r.keys(f"{_PFX}*")
    if keys:
        r.delete(*keys)


# ── Internal ──────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()