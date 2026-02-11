"""Embedding utilities for entity text. Used for vector index and retrieval."""

import os
from typing import List, Optional
from src.models.entity import Entity
from src.models.chunk import Chunk

# OpenAI (when OPENAI_API_KEY is set)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536

# Fallback: sentence-transformers (no API key)
SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
SENTENCE_TRANSFORMERS_DIMENSION = 384

_embedder = None
_embedding_dimension: Optional[int] = None


def _use_openai() -> bool:
    """Use OpenAI embeddings when OPENAI_API_KEY is set."""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _get_embedder():
    """Lazy-load the active embedder (OpenAI or sentence-transformers)."""
    global _embedder, _embedding_dimension
    if _embedder is not None:
        return _embedder
    if _use_openai():
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. Install with: uv add openai"
            )
        _embedder = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        _embedding_dimension = OPENAI_EMBEDDING_DIMENSION
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required when OPENAI_API_KEY is not set. "
            "Install with: uv add sentence-transformers"
        )
    _embedder = SentenceTransformer(SENTENCE_TRANSFORMERS_MODEL)
    _embedding_dimension = SENTENCE_TRANSFORMERS_DIMENSION
    return _embedder


def entity_to_text(entity: Entity) -> str:
    """Build a single searchable text from an entity (name, type, description, other props)."""
    parts = [entity.name, entity.type]
    if entity.description:
        parts.append(entity.description)
    props = entity.get_all_properties()
    for key in ("id", "name", "type", "description", "metadata"):
        props.pop(key, None)
    for k, v in sorted(props.items()):
        if v is not None and not isinstance(v, (dict, list)):
            parts.append(f"{k}: {v}")
    return " | ".join(str(p) for p in parts)


def _embed_text_openai(client, text: str) -> List[float]:
    """Embed using OpenAI API."""
    resp = client.embeddings.create(
        input=text.strip(),
        model=OPENAI_EMBEDDING_MODEL,
    )
    return resp.data[0].embedding


def _embed_text_sentence_transformers(model, text: str) -> List[float]:
    """Embed using sentence-transformers."""
    vec = model.encode(text.strip(), convert_to_numpy=True)
    return vec.tolist()


def embed_text(text: str) -> List[float]:
    """Embed a string; returns a list of floats. Dimension depends on backend (OpenAI or sentence-transformers)."""
    if not text or not text.strip():
        dim = get_embedding_dimension()
        return [0.0] * dim
    backend = _get_embedder()
    if _use_openai():
        return _embed_text_openai(backend, text)
    return _embed_text_sentence_transformers(backend, text)


def embed_entity(entity: Entity) -> List[float]:
    """Embed an entity (its text representation)."""
    return embed_text(entity_to_text(entity))


def embed_chunk(chunk: Chunk) -> List[float]:
    """Embed a chunk (its text content) for vector search."""
    return embed_text(chunk.text)


def get_embedding_dimension() -> int:
    """Return the embedding dimension of the active backend (OpenAI or sentence-transformers)."""
    global _embedding_dimension
    if _embedding_dimension is not None:
        return _embedding_dimension
    _get_embedder()
    return _embedding_dimension
