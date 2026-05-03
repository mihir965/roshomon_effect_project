"""Singleton Sentence-Transformer encoder with cosine similarity helper."""

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None
_text_cache: dict[str, np.ndarray] = {}
_CACHE_LIMIT = 4096  # bound size to avoid unbounded growth in long sessions


def get_encoder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _encode_one(text: str) -> np.ndarray:
    cached = _text_cache.get(text)
    if cached is not None:
        return cached
    vec = get_encoder().encode([text], normalize_embeddings=True)[0]
    if len(_text_cache) >= _CACHE_LIMIT:
        _text_cache.pop(next(iter(_text_cache)))
    _text_cache[text] = vec
    return vec


def encode(texts: list[str] | str) -> np.ndarray:
    """Return L2-normalised embeddings (shape: [n, dim] or [dim] for single string).

    In-memory cache deduplicates repeat encodings within a session — golden
    reasoning is encoded once per question even though RAS is computed
    per-run × per-model.
    """
    if isinstance(texts, str):
        return _encode_one(texts)

    # Encode any uncached strings in one batch, fill in cached ones from memory.
    missing_idx = [i for i, t in enumerate(texts) if t not in _text_cache]
    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        new_vecs = get_encoder().encode(missing_texts, normalize_embeddings=True)
        for i, vec in zip(missing_idx, new_vecs):
            if len(_text_cache) >= _CACHE_LIMIT:
                _text_cache.pop(next(iter(_text_cache)))
            _text_cache[texts[i]] = vec
    return np.stack([_text_cache[t] for t in texts])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D unit vectors."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return float(np.dot(a, b))  # already unit-norm after encode()
