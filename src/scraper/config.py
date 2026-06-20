"""
Scraper configuration.
All constants, URLs, and environment-driven settings live here.
"""
import os

# ── Sitemap ───────────────────────────────────────────────────────
SITEMAP_TAXONOMY_URL = "https://qanoon.om/wp-sitemap-taxonomies-category-1.xml"

# Categories to ignore during discovery
CATEGORY_BLACKLIST = {"uncategorized"}

# Mapping from URL slug → Arabic display name
SLUG_TO_AR = {
    "مرسوم-سلطاني":                           "مرسوم سلطاني",
    "قرار-وزاري":                              "قرار وزاري",
    "الجريدة-الرسمية":                        "الجريدة الرسمية",
    "أمر-سامي":                               "أمر سامي",
    "قانون-معدل":                              "قانون معدل",
    "اتفاقية-دولية":                          "اتفاقية دولية",
    "قانون-تقليدي":                            "قانون تقليدي",
    "قرارات-اللجنة-العليا-للتعامل-مع-فيروس": "قرارات اللجنة العليا",
    "لائحة-معدلة":                            "لائحة معدلة",
    "فتاوى-قانونية":                          "فتاوى قانونية",
    "تعميم":                                  "تعميم",
}

# Mapping from Arabic display name → normalized English key
AR_TO_KEY = {
    "مرسوم سلطاني":      "royal_decree",
    "قرار وزاري":        "ministerial_decision",
    "الجريدة الرسمية":   "official_gazette",
    "أمر سامي":          "royal_order",
    "قانون معدل":        "amended_law",
    "اتفاقية دولية":     "international_treaty",
    "قانون تقليدي":      "legacy_law",
    "قرارات اللجنة العليا": "supreme_committee_decisions",
    "لائحة معدلة":       "amended_regulation",
    "فتاوى قانونية":     "legal_opinion",
    "تعميم":             "circular",
}

# ── Anti-bot ──────────────────────────────────────────────────────
DELAY_MIN   = float(os.getenv("SCRAPER_DELAY_MIN",   "1.0"))
DELAY_MAX   = float(os.getenv("SCRAPER_DELAY_MAX",   "3.0"))
MAX_RETRIES = int(os.getenv("SCRAPER_MAX_RETRIES",   "5"))
CONCURRENCY = int(os.getenv("SCRAPER_CONCURRENCY",   "3"))

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/124.0.0.0 Safari/537.36",
]

# ── Repagination detection ────────────────────────────────────────
# How many consecutive already-done doc IDs signal we have caught up
CONSECUTIVE_DONE_THRESHOLD = 20

# ── Paths ─────────────────────────────────────────────────────────
DATA_OUTPUT_PATH = os.getenv("DATA_OUTPUT_PATH", "data/sample_output/")
PDF_OUTPUT_PATH  = os.getenv("PDF_OUTPUT_PATH",  "data/pdfs/")

# ── Redis ─────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB",  "0"))