import asyncio
import sys
import json
from pathlib import Path
from src.scraper.parser import parse_document_page
from src.scraper.utils import setup_logging

from src.scraper.models import ScrapedDocument  # important
from src.scraper.parser import AmendmentRelation  # if you defined it there


def _section(label: str, value) -> None:
    print(f"\n── {label} ──")
    print(value if value else f"⚠️  No {label} resolved")


def _as_dict(obj):
    """Normalize output into dict for saving/printing."""
    if isinstance(obj, dict):
        return obj

    if isinstance(obj, ScrapedDocument):
        return obj.__dict__

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return {"raw": str(obj)}


async def main(url: str) -> None:
    setup_logging()
    result = await parse_document_page(url,"المرسوم السلطاني")

    # ── TYPE CHECK ─────────────────────────────────────────────
    if isinstance(result, dict):
        data = result

    elif isinstance(result, ScrapedDocument):
        data = result.__dict__

    else:
        # fallback (e.g. AmendmentRelation or future graph object)
        print("\n⚠️ Non-document object returned:")
        print(type(result))
        print(result)

        data = _as_dict(result)

    # ── SUMMARY ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    _section("doc_id", data.get("doc_id"))
    _section("title", data.get("title"))
    _section("date", data.get("date"))
    _section("issuer", data.get("issuer"))
    # ── CONTENT ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONTENT")
    print("=" * 60)

    _section("content_Ar", data.get("contentAr"))
    _section("content_En", data.get("contentEn"))

    print("\n" + "=" * 60)

    # ── SAVE OUTPUT ────────────────────────────────────────────
    out_dir = Path("data/test_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_id = data.get("doc_id", "amendment_relation")

    json_path = out_dir / f"{doc_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if data.get("content_ar"):
        (out_dir / f"{doc_id}_ar.md").write_text(
            data["content_ar"], encoding="utf-8"
        )

    if data.get("content_en"):
        (out_dir / f"{doc_id}_en.md").write_text(
            data["content_en"], encoding="utf-8"
        )

    print(f"\nSaved to {out_dir}/{doc_id}.json (+ .md files)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.scraper.test_parse <url>")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))