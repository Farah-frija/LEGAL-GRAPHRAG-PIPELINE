"""
Phase 6.2 — Graph Community Detection + LLM Summarization.

Steps:
  1. Run Louvain community detection via Memgraph MAGE on the
     Document-Topic graph, writing community_id onto each node.
  2. For each community, collect its topic names + document titles.
  3. Call Gemini to generate a one-sentence legal sub-field description.
  4. Create a Community node linked to all member nodes.

Usage:
    python src/phase6_community.py
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from gqlalchemy import Memgraph
from loguru import logger

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set.")

GEMINI_MODEL = "gemini-3.1-flash-lite"
RPM_LIMIT    = 30
DELAY_S      = 60 / RPM_LIMIT

gemini = genai.Client(api_key=GEMINI_API_KEY)

mg = Memgraph(
    host=os.getenv("MEMGRAPH_HOST", "localhost"),
    port=int(os.getenv("MEMGRAPH_PORT", 7687)),
)

# ── Step 1: Run Louvain ───────────────────────────────────────────────────────

def run_louvain() -> int:
    logger.info("Running Louvain on Document-Topic subgraph only...")
    t0 = time.perf_counter()

    # Project only Document and Topic nodes via HAS_TOPIC edges
    mg.execute(
    """
    MATCH (d:Document)-[e:HAS_TOPIC]->(t:Topic)
    WITH collect(d) + collect(t) AS nodes, collect(e) AS relationships
    CALL community_detection.get_subgraph(nodes, relationships)
    YIELD node, community_id
    SET node.community_id = community_id
    """
)

    result = list(mg.execute_and_fetch(
        """
        MATCH (t:Topic)
        WHERE t.community_id IS NOT NULL
        RETURN count(DISTINCT t.community_id) AS n_communities
        """
    ))
    n = result[0]["n_communities"] if result else 0

    topic_count = list(mg.execute_and_fetch(
        "MATCH (t:Topic) WHERE t.community_id IS NOT NULL RETURN count(t) AS n"
    ))[0]["n"]
    doc_count = list(mg.execute_and_fetch(
        "MATCH (d:Document) WHERE d.community_id IS NOT NULL RETURN count(d) AS n"
    ))[0]["n"]

    logger.success(
        f"Louvain complete in {time.perf_counter() - t0:.3f}s — "
        f"{n} communities | {topic_count} topics | {doc_count} documents labeled"
    )
    return n


# ── Step 2: Collect community members ────────────────────────────────────────

def fetch_communities() -> dict[int, dict]:
    """
    Returns { community_id: { topics: [...], doc_titles: [...] } }
    """
    topic_rows = mg.execute_and_fetch(
        """
        MATCH (t:Topic)
        WHERE t.community_id IS NOT NULL
        RETURN t.community_id AS cid, collect(t.name) AS topics
        """
    )
    doc_rows = mg.execute_and_fetch(
        """
        MATCH (d:Document)
        WHERE d.community_id IS NOT NULL
        RETURN d.community_id AS cid,
        collect(d.title) AS titles
        """
    )

    communities: dict[int, dict] = {}

    for row in topic_rows:
        cid = row["cid"]
        communities.setdefault(cid, {"topics": [], "doc_titles": []})
        communities[cid]["topics"] = row["topics"]

    for row in doc_rows:
        cid = row["cid"]
        communities.setdefault(cid, {"topics": [], "doc_titles": []})
        communities[cid]["doc_titles"] = row["titles"]

    return communities

# ── Step 3: LLM community summarization ──────────────────────────────────────

_SUMMARY_SYSTEM = (
    "You are a legal taxonomist specializing in Omani law. "
    "Given a list of legal topics and document titles that form a cluster, "
    "write a single concise English noun phrase (max 8 words) that names "
    "the legal sub-field this cluster represents. "
    "Return only the noun phrase, nothing else."
)

def summarize_community(cid: int, topics: list[str], doc_titles: list[str]) -> str:
    topics_str = ", ".join(topics[:20]) or "—"
    titles_str = "\n".join(f"- {t}" for t in doc_titles[:10] if t) or "—"

    prompt = (
        f"Topics: {topics_str}\n\n"
        f"Document titles:\n{titles_str}"
    )
    try:
        response = gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SUMMARY_SYSTEM,
                temperature=0.2,
                max_output_tokens=32,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Community {cid} summarization failed: {e}")
        return f"Community {cid}"

# ── Step 4: Persist Community nodes ──────────────────────────────────────────

def persist_community(cid: int, label: str) -> None:
    """
    Create a Community node and link all member Documents and Topics to it.
    """
    mg.execute(
        """
        MERGE (c:Community {id: $cid})
        SET c.label = $label
        WITH c
        MATCH (n)
        WHERE n.community_id = $cid
        MERGE (n)-[:BELONGS_TO]->(c)
        """,
        parameters={"cid": cid, "label": label},
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    run_louvain()
    communities = fetch_communities()
    logger.info(f"Summarizing {len(communities)} communities via Gemini...")

    for i, (cid, members) in enumerate(communities.items()):
        topics     = members["topics"]
        doc_titles = members["doc_titles"]

        if not topics and not doc_titles:
            continue

        label = summarize_community(cid, topics, doc_titles)
        persist_community(cid, label)
        logger.success(f"[{i+1}/{len(communities)}] Community {cid} → '{label}'")

        if i < len(communities) - 1:
            time.sleep(DELAY_S)

    logger.success("Community detection and labeling complete.")

if __name__ == "__main__":
    main()