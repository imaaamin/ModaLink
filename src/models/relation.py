"""Relation model for extracted relations between entities."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Relation(BaseModel):
    """Represents a relation between two entities."""
    
    id: str = Field(description="Unique identifier for the relation")
    source_entity_id: str = Field(description="ID of the source entity")
    target_entity_id: str = Field(description="ID of the target entity")
    relation_type: str = Field(description="Type of relation (e.g., WORKS_FOR, LOCATED_IN, OWNS)")
    description: Optional[str] = Field(default=None, description="Description or context of the relation")
    confidence: Optional[float] = Field(default=None, description="Confidence score (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    # Allow additional properties dynamically
    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "example": {
                "id": "relation_1",
                "source_entity_id": "entity_1",
                "target_entity_id": "entity_2",
                "relation_type": "WORKS_FOR",
                "description": "John Doe works for TechCorp",
                "confidence": 0.95,
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "role": "CEO",
                "metadata": {}
            }
        }
    }
    
    def get_all_properties(self) -> Dict[str, Any]:
        """Get all properties including additional fields as a dictionary."""
        # Get all fields from the model, including extra fields
        model_dict = self.model_dump(exclude_none=False, exclude_unset=False)
        
        # Start with core fields
        props = {
            "id": model_dict.get("id"),
            "source_entity_id": model_dict.get("source_entity_id"),
            "target_entity_id": model_dict.get("target_entity_id"),
            "relation_type": model_dict.get("relation_type"),
            "description": model_dict.get("description"),
            "confidence": model_dict.get("confidence"),
        }
        
        # Add metadata fields
        if model_dict.get("metadata"):
            props.update(model_dict["metadata"])
        
        # Add any additional fields (extra fields from Pydantic)
        for key, value in model_dict.items():
            if key not in ["id", "source_entity_id", "target_entity_id", "relation_type", "description", "confidence", "metadata"] and value is not None:
                props[key] = value
        
        return props
