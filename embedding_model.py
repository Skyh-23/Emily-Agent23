"""
Shared Embedding Model — Singleton Loader
==========================================
Both RAG engine and Memory system share the same SentenceTransformer
instance to avoid loading the model twice (~2GB RAM saved).
"""

import logging
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model_instance = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns a shared SentenceTransformer instance (lazy singleton).
    First call loads the model, subsequent calls return the cached instance.
    """
    global _model_instance
    if _model_instance is None:
        logger.info("Loading embedding model: %s (first load, may take a moment)...", EMBEDDING_MODEL)
        print(f"🔄  Loading embedding model '{EMBEDDING_MODEL}'...")

        # Resolve to local cache path — bypasses is_base_mistral() network call
        local_path = snapshot_download(EMBEDDING_MODEL, local_files_only=True)
        _model_instance = SentenceTransformer(local_path)

        print(f"✅  Embedding model loaded: {EMBEDDING_MODEL}")
        logger.info("Embedding model loaded: %s", EMBEDDING_MODEL)
    return _model_instance


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a vector."""
    model = get_embedding_model()
    return model.encode(text).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts into vectors (batched, faster)."""
    model = get_embedding_model()
    return model.encode(texts).tolist()


if __name__ == "__main__":
    # Quick test
    vec = embed_text("Hello, I am Emily, your AI assistant.")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Vector dimensions: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")