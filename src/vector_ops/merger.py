"""
Phase 6 (Bonus) — Topic node deduplication via cosine similarity.

LLMs often produce near-duplicate topics (e.g. "Labor Policies" vs
"Labor Regulations"). This script:
  1. Loads all Topic embeddings.
  2. Computes pairwise cosine similarity.
  3. Merges any pair exceeding a threshold (default 0.88).
     - Redirects all relationships from the duplicate to the canonical node.
     - Deletes the duplicate.

Usage:
    python src/vector_ops/merger.py
    python src/vector_ops/merger.py --threshold 0.90
"""
import argparse
import os

import numpy as np
from gqlalchemy import Memgraph
from loguru import logger
from scipy.spatial.distance import cosine

mg = Memgraph(
    host=os.getenv("MEMGRAPH_HOST", "localhost"),
    port=int(os.getenv("MEMGRAPH_PORT", 7687)),
)


def load_topics() -> list[dict]:
    return list(mg.execute_and_fetch(
        "MATCH (t:Topic) WHERE t.embedding IS NOT NULL "
        "RETURN t.name AS name, t.embedding AS embedding"
    ))


def merge_pair(canonical: str, duplicate: str) -> None:
    """Redirect all rels from duplicate to canonical, then delete duplicate."""
    mg.execute(
        """
        MATCH (dup:Topic {name: $dup})<-[:HAS_TOPIC]-(d:Document)
        MATCH (can:Topic {name: $can})
        MERGE (d)-[:HAS_TOPIC]->(can)
        """,
        parameters={"dup": duplicate, "can": canonical},
    )
    mg.execute(
        "MATCH (t:Topic {name: $name}) DETACH DELETE t",
        parameters={"name": duplicate},
    )
    logger.info(f"Merged '{duplicate}' → '{canonical}'")


def run(threshold: float) -> None:
    topics = load_topics()
    logger.info(f"Loaded {len(topics)} topic embeddings")

    names = [t["name"] for t in topics]
    vecs  = np.array([t["embedding"] for t in topics])
    merged = set()

    for i in range(len(names)):
        if names[i] in merged:
            continue
        for j in range(i + 1, len(names)):
            if names[j] in merged:
                continue
            sim = 1 - cosine(vecs[i], vecs[j])
            if sim >= threshold:
                merge_pair(canonical=names[i], duplicate=names[j])
                merged.add(names[j])

    logger.success(f"Merged {len(merged)} duplicate topics (threshold={threshold})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.88)
    run(ap.parse_args().threshold)
