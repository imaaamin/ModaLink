"""Chunk model for text segments linked to a document."""

from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A segment of document text, ordered and linked to the document and next chunk."""

    id: str = Field(description="Unique chunk identifier (e.g. chunk_0, chunk_1)")
    text: str = Field(description="Chunk text content")
    index: int = Field(description="Zero-based order in the document")
    document_id: str = Field(description="ID of the source document")
