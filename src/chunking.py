"""Chunk document text into segments for document/chunk graph structure."""

from typing import List
from src.models.chunk import Chunk


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Chunk]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full document text.
        document_id: Document ID to attach to each chunk.
        chunk_size: Target size in characters per chunk.
        overlap: Overlap in characters between consecutive chunks.

    Returns:
        List of Chunk with id chunk_0, chunk_1, ... in order.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    step = max(1, chunk_size - overlap)
    chunks: List[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end]
        if segment.strip():
            chunks.append(
                Chunk(
                    id=f"chunk_{index}",
                    text=segment,
                    index=index,
                    document_id=document_id,
                )
            )
            index += 1
        start += step
    return chunks
