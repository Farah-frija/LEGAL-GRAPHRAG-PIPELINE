"""
Shared scraper utilities.

Contains:
  - HTTP fetch with anti-bot evasion (httpx primary, Playwright fallback)
  - Taxonomy sitemap parser  (discovers category URLs)
  - Listing page parser      (extracts doc URLs from <h2> titles only)
  - Discovery                (pagination + simple resume + deletion handling)
  - Logging setup
"""
from __future__ import annotations
import asyncio
import logging
import random
import re
import sys
from urllib.parse import unquote

from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import async_playwright
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)
import httpx

from src.scraper.config import (
    USER_AGENTS, MAX_RETRIES, DELAY_MIN, DELAY_MAX,
    SITEMAP_TAXONOMY_URL, CATEGORY_BLACKLIST,
    SLUG_TO_AR, AR_TO_KEY,
)
from src.scraper.checkpoint import (
    get_pending_urls, is_discovery_done, is_visited, mark_visited, enqueue_urls, pending_count,
    is_done, save_category_progress, get_category_progress, set_category_state,
)


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Logging
# ══════════════════════════════════════════════════════════════════

def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        "logs/scraper_{time}.log",
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
    )


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — HTTP fetch with anti-bot evasion
# ══════════════════════════════════════════════════════════════════

class BlockedError(Exception):
    """Raised when bot-detection response is detected."""


def _headers(referer: str = "https://qanoon.om/") -> dict:
    return {
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ar,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection":      "keep-alive",
        "Referer":         referer,
        "DNT":             "1",
    }


def _is_blocked(html: str) -> bool:
    """Detect common bot-blocking response signatures."""
    signals = [
        "cf-browser-verification",  # Cloudflare challenge
        "Just a moment",            # Cloudflare waiting room
        "captcha",                  # Generic CAPTCHA
        "Access Denied",
        "Rate limit",
    ]
    lower = html.lower()
    return any(s.lower() in lower for s in signals)


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((httpx.HTTPError, BlockedError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _fetch_httpx(url: str, referer: str) -> str:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url, headers=_headers(referer))
        resp.raise_for_status()
        html = resp.text
        if _is_blocked(html):
            raise BlockedError(f"Bot detection triggered on {url}")
        await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        return html


async def _fetch_playwright(url: str) -> str:
    """
    Full headless browser fallback.
    Used when httpx gets blocked by JS challenges or CAPTCHAs.
    """
    logger.warning(f"Playwright fallback: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        ctx = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="ar-OM",
        )
        page = await ctx.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        await page.goto(url, wait_until="networkidle", timeout=60_000)
        await asyncio.sleep(random.uniform(2, 4))
        html = await page.content()
        await browser.close()
        return html


async def fetch(url: str, referer: str = "https://qanoon.om/") -> str:
    """
    Unified fetch entry point used by all scraper code.
    Tries httpx first; falls back to Playwright on BlockedError.
    Raises on all other failures after MAX_RETRIES attempts.
    """
    try:
        return await _fetch_httpx(url, referer)
    except BlockedError:
        return await _fetch_playwright(url)


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Taxonomy sitemap parser
# ══════════════════════════════════════════════════════════════════

def _slug_from_url(url: str) -> str:
    return unquote(url).rstrip("/").split("/")[-1]


def _ar_name(slug: str) -> str:
    return SLUG_TO_AR.get(slug, slug.replace("-", " "))


def _cat_key(ar: str) -> str:
    return AR_TO_KEY.get(ar, ar.replace(" ", "_").lower())


async def fetch_categories() -> list[dict]:
    """
    Fetch wp-sitemap-taxonomies-category-1.xml and return all
    valid categories as { url, category_ar, category_key } dicts.

    This replaces the hardcoded CATEGORIES dict — categories are
    discovered dynamically so new ones are picked up automatically.
    """
    logger.info(f"Fetching taxonomy sitemap: {SITEMAP_TAXONOMY_URL}")
    xml  = await fetch(SITEMAP_TAXONOMY_URL)
    soup = BeautifulSoup(xml, "xml")
    print(soup.prettify()[:10])
    cats = []

    for loc in soup.find_all("loc"):
        url  = loc.get_text(strip=True)
        slug = _slug_from_url(url)

        if slug.lower() in CATEGORY_BLACKLIST:
            logger.debug(f"Skipping blacklisted category: {slug}")
            continue

        ar  = _ar_name(slug)
        key = _cat_key(ar)
        cats.append({"url": url, "category_ar": ar, "category_key": key})
        logger.info(f"  Category: {ar} ({key})")

    logger.info(f"Total categories discovered: {len(cats)}")
    return cats


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Listing page helpers
# ══════════════════════════════════════════════════════════════════

def build_page_url(base_url: str, page: int) -> str:
    base = base_url.rstrip("/") + "/"
    return base if page == 1 else f"{base}page/{page}/"


def extract_doc_urls(html: str) -> list[str]:
    """
    Extract individual document URLs from a category listing page.

    IMPORTANT: only reads hrefs from <h2> title links, NOT from
    inline body links. This is critical because each listing page
    renders full document bodies including cross-reference links
    (e.g. references to النظام الأساسي للدولة). Naively extracting
    all /p/YEAR/ hrefs would flood the queue with referenced docs
    that belong to other categories or are already known.

    From real page inspection: every listing item title is an <h2>
    containing a single <a> with the document URL.
    Fragment URLs (#more-XXXXX for truncated posts) are excluded.
    """
    soup = BeautifulSoup(html, "lxml")
    seen, urls = set(), []

    for h2 in soup.find_all("h2"):
        a = h2.find("a", href=True)
        if not a:
            continue
        href: str = a["href"]
        # Must match /p/YEAR/slug/ with no # fragment
        if re.search(r"/p/(19|20)\d{2}/[\w-]+/", href) and "#" not in href:
            clean = href.rstrip("/") + "/"
            if clean not in seen:
                seen.add(clean)
                urls.append(clean)

    return urls


def doc_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Discovery (pagination + simple resume logic)
# ══════════════════════════════════════════════════════════════════

async def _find_safe_restart_page(r, category: dict, last_page: int) -> int:
    """
    Phase-aware restart logic.

    - If DISCOVERY_DONE → use done-based logic (old system)
    - If still DISCOVERING → use pending-based logic (safe fallback)
    """

    cat_url = category["url"]
    cat_ar = category["category_ar"]

    

    for page in range(last_page, 0, -1):
        page_url = build_page_url(cat_url, page)

        try:
            html = await fetch(page_url, referer=cat_url)
        except Exception as e:
            logger.warning(f"[{cat_ar}] page {page} fetch failed: {e}")
            continue

        doc_urls = extract_doc_urls(html)
        doc_ids = [doc_id_from_url(u) for u in doc_urls]

        if not doc_ids:
            continue

        # ─────────────────────────────────────────────
        # CASE 1: DISCOVERY COMPLETE → use DONE logic
        # ─────────────────────────────────────────────
        
          
        pending_set = set(get_pending_urls(r))

            # page is "stable" if all docs are already discovered
        if all(u in pending_set for u in doc_urls):
             logger.info(
                    f"[{cat_ar}] Safe restart (PENDING-based): page {page} → {page + 1}"
                )
             return page + 1

        logger.debug(f"[{cat_ar}] Page {page} unsafe → going further back")

    logger.warning(f"[{cat_ar}] No safe restart point → page 1")
    return 1


async def _paginate_forward(r, category: dict, start_page: int,max_pages: int | None = None) -> None:
    """
    Paginate a category from start_page forward until an empty page.

    After EACH listing page (in order):
      1. Extract doc URLs from <h2> titles only
      2. Enqueue new (url, category_ar) pairs — deduplication is automatic
      3. Mark listing page as visited
      4. Save page progress to Redis

    Step 4 happens ONLY after the full page is enqueued and visited.
    A crash between steps 2–4 is safe: the backwards walk re-fetches
    this page and enqueue_urls deduplicates any already-queued URLs.
    """
    cat_url = category["url"]
    cat_ar  = category["category_ar"]
    page    = start_page
    failure=False
    while True:
        if max_pages is not None and page > max_pages:
            logger.info(f"Reached max_pages={max_pages} for category, stopping.")
            break

        page_url = build_page_url(cat_url, page)

        if is_visited(r, page_url):
            logger.debug(f"[{cat_ar}] page {page} already visited — skip")
            page += 1
            continue

        logger.info(f"[{cat_ar}] listing page {page}")

        try:
            print(page_url)
            html = await fetch(page_url, referer=cat_url)
        except Exception as e:
            logger.error(f"[{cat_ar}] page {page} fetch failed: {e}")
            failure=True
            break

        doc_urls = extract_doc_urls(html)
        
        if not doc_urls:
            logger.info(f"[{cat_ar}] page {page} empty — end of category")
            mark_visited(r, page_url)
            break

        pairs = [(url, cat_ar) for url in doc_urls]
        added = enqueue_urls(r, pairs)
        logger.info(
            f"[{cat_ar}] page {page}: "
            f"{len(doc_urls)} found, {added} new queued"
        )

        mark_visited(r, page_url)
        save_category_progress(r, cat_url, page)

        page += 1
        await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    if not failure:
        set_category_state(r, category["url"], "DISCOVERY_DONE")
async def discover_category(r, category: dict,max_pages: int | None = None) -> None:
    """
    Full discovery for one category with smart resume.

    First run  → start from page 1
    Resume     → walk backwards to find first fully-done page
                 → restart from there + 1
    Additions  → enqueue_urls deduplication catches new docs automatically
    Deletions  → backwards walk finds correct restart point naturally
    """
    cat_ar         = category["category_ar"]
    last_done_page = get_category_progress(r, category["url"])

    if last_done_page == 0:
        logger.info(f"[{cat_ar}] First run — starting from page 1")
        start_page = 1
    else:
        start_page = await _find_safe_restart_page(r, category, last_done_page)

    await _paginate_forward(r, category, start_page,max_pages)


async def discover_all(r, categories, max_pages: int | None = None): 
    for cat in categories:
        logger.info("=" * 50)
        discovery_done = is_discovery_done(r, cat["url"])
        logger.info(
            f"[{cat}] Restart check | "
            f"discovery_done={discovery_done} | last_page={get_category_progress(r, cat["url"])}"
            )
        if not discovery_done:
            
            logger.info(f"Discovering: {cat['category_ar']} ({cat['category_key']})")
        
            await discover_category(r, cat, max_pages)
            
        
            

        # ✅ mark THIS category as fully discovered
        

    logger.info(f"Discovery complete — total pending: {pending_count(r)}")