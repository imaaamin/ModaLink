"""Document model for the source document node in the graph."""

from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Source document node: docId, title, source, publishedDate."""

    doc_id: str = Field(description="Unique document identifier (alias: docId)")
    title: Optional[str] = Field(default=None, description="Document title")
    source: Optional[str] = Field(default=None, description="Source path or URI")
    published_date: Optional[str] = Field(default=None, description="Publication date (alias: publishedDate)")

    def for_neo4j(self) -> dict:
        """Property dict for Neo4j (snake_case keys; exporter can use docId, publishedDate if desired)."""
        return {
            "docId": self.doc_id,
            "title": self.title or "",
            "source": self.source or "",
            "publishedDate": self.published_date or "",
        }
