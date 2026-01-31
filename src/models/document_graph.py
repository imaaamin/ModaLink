"""Document graph model for the complete graph structure."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .entity import Entity
from .relation import Relation


class DocumentGraph(BaseModel):
    """Represents the complete graph of entities and relations from a document."""
    
    entities: List[Entity] = Field(default_factory=list, description="List of extracted entities")
    relations: List[Relation] = Field(default_factory=list, description="List of extracted relations")
    document_id: Optional[str] = Field(default=None, description="Identifier for the source document")
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
