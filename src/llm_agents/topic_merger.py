"""
Phase 6.1 — Topological Merging via Vector Proximity.

Algorithm:
  1. Load all Topic nodes with embeddings
  2. Build a similarity graph (nodes=topics, edges=sim > threshold)
  3. Find connected components via BFS
  4. For each component → pick canonical (max HAS_TOPIC edges)
  5. Merge all duplicates into canonical
  6. Clean up Redis cache
  7. Report optimization metrics

Usage:
    python src/phase6_topic_merge.py [--threshold 0.88] [--dry-run]
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import redis
from gqlalchemy import Memgraph
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.88
EMBEDDED_TOPICS_KEY  = "embedded_topics"

mg = Memgraph(
    host=os.getenv("MEMGRAPH_HOST", "localhost"),
    port=int(os.getenv("MEMGRAPH_PORT", 7687)),
)
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)

# ── Load topics ───────────────────────────────────────────────────────────────

def load_topics() -> list[dict]:
    """Fetch all Topic nodes that have an embedding."""
    logger.info("Fetching Topic nodes with embeddings from Memgraph...")
    t0   = time.perf_counter()
    rows = list(mg.execute_and_fetch(
        """
        MATCH (t:Topic)
        WHERE t.embedding IS NOT NULL
        RETURN t.name AS name, t.embedding AS embedding
        """
    ))
    logger.success(f"Loaded {len(rows)} topics in {time.perf_counter() - t0:.3f}s")
    return rows

# ── Graph metrics ─────────────────────────────────────────────────────────────

def get_graph_stats() -> dict:
    """Snapshot current Topic node and HAS_TOPIC edge counts."""
    topic_count = list(mg.execute_and_fetch(
        "MATCH (t:Topic) RETURN count(t) AS n"
    ))[0]["n"]

    edge_count = list(mg.execute_and_fetch(
        "MATCH (:Document)-[r:HAS_TOPIC]->(:Topic) RETURN count(r) AS n"
    ))[0]["n"]

    logger.debug(f"Graph stats — topics: {topic_count}, HAS_TOPIC edges: {edge_count}")
    return {"topics": topic_count, "edges": edge_count}


def get_topic_edge_counts(names: list[str]) -> dict[str, int]:
    """For a list of topic names, return how many HAS_TOPIC edges each has."""
    logger.debug(f"Fetching edge counts for {len(names)} topics...")
    rows = mg.execute_and_fetch(
        """
        UNWIND $names AS name
        MATCH (t:Topic {name: name})
        OPTIONAL MATCH (:Document)-[r:HAS_TOPIC]->(t)
        RETURN t.name AS name, count(r) AS edge_count
        """,
        parameters={"names": names},
    )
    counts = {r["name"]: r["edge_count"] for r in rows}
    for name, count in counts.items():
        logger.debug(f"  '{name}' → {count} edges")
    return counts


def log_optimization_report(before: dict, after: dict, merged: int, elapsed: float) -> None:
    topics_removed = before["topics"] - after["topics"]
    edges_removed  = before["edges"]  - after["edges"]
    topics_pct     = (topics_removed / before["topics"] * 100) if before["topics"] else 0
    edges_pct      = (edges_removed  / before["edges"]  * 100) if before["edges"]  else 0

    logger.success("─" * 54)
    logger.success("Optimization Report")
    logger.success("─" * 54)
    logger.success(f"  Total time          : {elapsed:.2f}s")
    logger.success(f"  Merge pairs processed: {merged}")
    logger.success(
        f"  Topic nodes  : {before['topics']:,} → {after['topics']:,} "
        f"  (-{topics_removed:,} / -{topics_pct:.1f}%)"
    )
    logger.success(
        f"  HAS_TOPIC edges : {before['edges']:,} → {after['edges']:,} "
        f"  (-{edges_removed:,} / -{edges_pct:.1f}%)"
    )
    logger.success("─" * 54)

# ── Similarity graph + connected components ───────────────────────────────────

def build_similarity_graph(
    topics: list[dict],
    threshold: float,
) -> dict[str, set[str]]:
    """
    Build adjacency list where edges = cosine similarity > threshold.
    Returns dict: name → set of similar names.
    """
    logger.info(f"Computing {len(topics)}×{len(topics)} cosine similarity matrix...")
    t0 = time.perf_counter()

    names      = [t["name"] for t in topics]
    matrix     = np.array([t["embedding"] for t in topics], dtype=np.float32)
    sim_matrix = cosine_similarity(matrix)

    elapsed = time.perf_counter() - t0
    logger.success(f"Similarity matrix computed in {elapsed:.3f}s")

    adjacency: dict[str, set[str]] = defaultdict(set)
    n          = len(names)
    edge_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                adjacency[names[i]].add(names[j])
                adjacency[names[j]].add(names[i])
                edge_count += 1
                logger.debug(
                    f"  Similar: '{names[i]}' ↔ '{names[j]}' "
                    f"(sim={sim_matrix[i, j]:.4f})"
                )

    logger.info(f"Similarity graph built — {edge_count} edges above threshold {threshold}")
    return adjacency


def find_connected_components(
    adjacency: dict[str, set[str]],
) -> list[set[str]]:
    """BFS over the similarity graph to find connected components."""
    logger.info("Finding connected components via BFS...")
    visited    = set()
    components = []

    for node in adjacency:
        if node in visited:
            continue

        # BFS
        component = set()
        queue     = [node]
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            visited.add(current)
            queue.extend(adjacency[current] - component)

        if len(component) >= 2:
            components.append(component)
            logger.debug(f"  Component found: {component}")

    logger.info(f"Found {len(components)} connected components (clusters to merge)")
    return components


def find_merge_clusters(topics: list[dict], threshold: float) -> list[dict]:
    """
    Full pipeline:
      1. Build similarity graph
      2. Find connected components
      3. For each component pick canonical (max HAS_TOPIC edges)
    Returns list of {"canonical": str, "duplicates": [str]}
    """
    adjacency  = build_similarity_graph(topics, threshold)
    components = find_connected_components(adjacency)

    if not components:
        return []

    clusters = []
    for i, component in enumerate(components):
        members     = list(component)
        edge_counts = get_topic_edge_counts(members)

        # canonical = most referenced; tie-break = alphabetically first
        canonical  = max(members, key=lambda n: (edge_counts.get(n, 0), -ord(n[0])))
        duplicates = [n for n in members if n != canonical]

        logger.info(
            f"  Cluster {i+1}/{len(components)}: "
            f"canonical='{canonical}' ({edge_counts.get(canonical, 0)} edges) | "
            f"duplicates={duplicates}"
        )
        clusters.append({"canonical": canonical, "duplicates": duplicates})

    return clusters

# ── Memgraph merge ────────────────────────────────────────────────────────────

def merge_topic_pair(canonical: str, duplicate: str) -> None:
    """
    Redirect all HAS_TOPIC edges from duplicate → canonical, then delete duplicate.
    """

    mg.execute(
        """
        MATCH (d:Document)-[r:HAS_TOPIC]->(dup:Topic {name: $dup})
        MATCH (can:Topic {name: $can})
        MERGE (d)-[:HAS_TOPIC]->(can)
        DELETE r
        """,
        parameters={"dup": duplicate, "can": canonical},
    )

    logger.debug(f"  Deleting duplicate node: '{duplicate}'")
    mg.execute(
        "MATCH (t:Topic {name: $dup}) DELETE t",
        parameters={"dup": duplicate},
    )

# ── Redis cleanup ─────────────────────────────────────────────────────────────

def remove_from_redis(name: str) -> None:
    result = redis_client.srem(EMBEDDED_TOPICS_KEY, name)
    if result:
        logger.debug(f"  Removed '{name}' from Redis cache")
    else:
        logger.debug(f"  '{name}' was not in Redis cache (already absent)")

# ── Main ──────────────────────────────────────────────────────────────────────

def run(threshold: float, dry_run: bool) -> None:
    t_total = time.perf_counter()
    logger.info(f"Topic merge started — threshold={threshold}, dry_run={dry_run}")

    topics = load_topics()

    if len(topics) < 2:
        logger.info("Not enough topics to compare. Exiting.")
        return

    clusters = find_merge_clusters(topics, threshold)
    logger.info(f"Total clusters to merge: {len(clusters)}")

    if not clusters:
        logger.success("No merges needed — graph is already clean.")
        return

    if dry_run:
        logger.info("[DRY RUN] Would perform the following merges:")
        for i, c in enumerate(clusters):
            logger.info(
                f"  [{i+1}] canonical='{c['canonical']}' ← "
                f"duplicates={c['duplicates']}"
            )
        logger.info("[DRY RUN] No changes made.")
        return

    # snapshot before
    before = get_graph_stats()
    logger.info(
        f"Before — topics: {before['topics']:,}, "
        f"HAS_TOPIC edges: {before['edges']:,}"
    )

    merged = 0
    for i, cluster in enumerate(clusters):
        canonical  = cluster["canonical"]
        duplicates = cluster["duplicates"]

        logger.info(
            f"[{i+1}/{len(clusters)}] Merging {len(duplicates)} duplicate(s) "
            f"into '{canonical}'"
        )

        for duplicate in duplicates:
            logger.info(f"  '{duplicate}' → '{canonical}'")
            merge_topic_pair(canonical, duplicate)
            remove_from_redis(duplicate)
            merged += 1

    # snapshot after
    after   = get_graph_stats()
    elapsed = time.perf_counter() - t_total
    logger.info(
        f"After  — topics: {after['topics']:,}, "
        f"HAS_TOPIC edges: {after['edges']:,}"
    )

    log_optimization_report(before, after, merged, elapsed)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Topic deduplication via cosine similarity")
    ap.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                    help="Cosine similarity threshold (default: 0.88)")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Print merge candidates without modifying the graph")
    args = ap.parse_args()

    run(args.threshold, args.dry_run)