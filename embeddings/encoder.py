"""Singleton Sentence-Transformer encoder with cosine similarity helper."""

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_encoder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode(texts: list[str] | str) -> np.ndarray:
    """Return L2-normalised embeddings (shape: [n, dim] or [dim] for single string)."""
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    vecs = get_encoder().encode(texts, normalize_embeddings=True)
    return vecs[0] if single else vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D unit vectors."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return float(np.dot(a, b))  # already unit-norm after encode()
