"""Document graph model for the complete graph structure."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .entity import Entity
from .relation import Relation
from .document import Document
from .chunk import Chunk


class DocumentGraph(BaseModel):
    """Represents the complete graph of entities and relations from a document.

    Invocation is unchanged: run the extractor to get a DocumentGraph with entities and relations.
    Relations can have extra attributes (e.g. start_date, end_date, occurred_on, role) stored on
    the relation, not as separate entities. Access them via relation.get_all_properties() or
    relation.<attr> (e.g. relation.start_date) when present.
    Optional document and chunks: document node (docId, title, source, publishedDate), chunks
    linked Document->first chunk and chunk->next chunk; entities linked to chunks via source_chunk_id.
    """

    entities: List[Entity] = Field(default_factory=list, description="List of extracted entities")
    relations: List[Relation] = Field(default_factory=list, description="List of extracted relations")
    document_id: Optional[str] = Field(default=None, description="Identifier for the source document")
    document: Optional[Document] = Field(default=None, description="Source document node (docId, title, source, publishedDate)")
    chunks: List[Chunk] = Field(default_factory=list, description="Ordered text chunks; Document->first chunk, chunk->next chunk")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional document metadata")

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by its ID."""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None

    def get_relations_for_entity(self, entity_id: str) -> List[Relation]:
        """Get all relations involving a specific entity."""
        return [
            rel for rel in self.relations
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        ]

    def get_relations_with_attribute(self, attr_name: str) -> List[Relation]:
        """Get relations that have a given attribute (e.g. 'start_date', 'end_date', 'role')."""
        return [
            rel for rel in self.relations
            if rel.get_all_properties().get(attr_name) is not None
        ]
