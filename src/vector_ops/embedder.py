"""
Shared embedding module — paraphrase-multilingual-MiniLM-L12-v2

- Free, self-hosted, no API key
- 50+ languages including Arabic
- 512-token context window  →  fits 400-token chunks with headroom
- 384-dim dense vectors
- ~90MB download

Import and call embed_texts() from any worker.
The model is loaded once at import time (singleton).
"""

import os
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from loguru import logger
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM   = 384
BATCH_SIZE  = 64  # safe for CPU; bump to 128+ with GPU

logger.info(f"Loading embedding model: {EMBED_MODEL} ...")
_model = SentenceTransformer(EMBED_MODEL)
logger.info("Embedding model ready.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings. Returns list of 384-dim float vectors,
    in the same order as input. Empty strings are replaced with a space.
    """
    if not texts:
        return []

    safe = [t if t.strip() else " " for t in texts]
    vecs = _model.encode(safe, batch_size=BATCH_SIZE, show_progress_bar=False)
    return [v.tolist() for v in vecs]