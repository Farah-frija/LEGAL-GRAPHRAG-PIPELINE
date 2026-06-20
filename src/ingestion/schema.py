"""
Phase 3 — Graph schema setup.
Creates uniqueness constraints and indexes in Memgraph.

Run ONCE before starting the ingestor:
    python src/ingestion/schema.py

Safe to re-run — every statement is wrapped in a try/except
so already-existing constraints/indexes are silently skipped.

Node labels and unique keys:
    :Document  →  id          (URL slug, e.g. "d123")
    :Topic     →  name        (normalised label from LLM)
    :Chunk     →  id          (f"{doc_id}_{chunk_index}")

Relationship types (all between Document nodes):
   
    [:AMENDS]      sourced from AmendmentRelation records
    [:REPEALS]     detected from title / content keywords
"""

import os

from gqlalchemy import Memgraph
from loguru import logger

mg = Memgraph(
    host=os.getenv("MEMGRAPH_HOST", "localhost"),
    port=int(os.getenv("MEMGRAPH_PORT", 7687)),
)

_CONSTRAINTS = [
    "CREATE CONSTRAINT ON (d:Document) ASSERT d.id IS UNIQUE;",
    "CREATE CONSTRAINT ON (t:Topic)    ASSERT t.name IS UNIQUE;",
    "CREATE CONSTRAINT ON (c:Chunk)    ASSERT c.id IS UNIQUE;",
]

_INDEXES = [
    # Lookup indexes used during ingestion and search
    "CREATE INDEX ON :Document(id);",
    "CREATE INDEX ON :Document(date);",
    "CREATE INDEX ON :Document(category);",
    "CREATE INDEX ON :Document(status);",
    "CREATE INDEX ON :Topic(name);",
    "CREATE INDEX ON :Chunk(doc_id);",

    # Vector indexes 
     "CREATE VECTOR INDEX chunk_embedding_idx ON :Chunk(embedding) "
     "WITH CONFIG {'dimension': 384, 'capacity': 1000000};",
     "CREATE VECTOR INDEX topic_embedding_idx ON :Topic(embedding) "
     "WITH CONFIG {'dimension': 384, 'capacity': 50000};",
]


def create_schema() -> None:
    logger.info("Setting up Memgraph schema…")

    for q in _CONSTRAINTS + _INDEXES:
        try:
            mg.execute(q)
            logger.success(f"OK  : {q[:80]}")
        except Exception as e:
            # Memgraph raises if constraint/index already exists — safe to skip
            logger.warning(f"SKIP: {q[:80]}  ({e})")

    logger.info("Schema setup complete.")


if __name__ == "__main__":
    create_schema()