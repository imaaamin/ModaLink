"""Entity model for extracted entities."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, model_validator


class Entity(BaseModel):
    """Represents an entity extracted from a document."""
    
    id: str = Field(description="Unique identifier for the entity")
    name: str = Field(description="Name or label of the entity")
    type: str = Field(description="Type/category of the entity (e.g., PERSON, ORGANIZATION, LOCATION)")
    description: Optional[str] = Field(default=None, description="Additional description or context")
    source_chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks where this entity is mentioned (for MENTIONED_IN links)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode="before")
    @classmethod
    def _source_chunk_id_to_ids(cls, data: Any) -> Any:
        """Backward compat: accept old source_chunk_id and set source_chunk_ids."""
        if isinstance(data, dict) and data.get("source_chunk_id") and not data.get("source_chunk_ids"):
            data = {**data, "source_chunk_ids": [data["source_chunk_id"]]}
        return data
    
    # Allow additional properties dynamically
    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "example": {
                "id": "entity_1",
                "name": "John Doe",
                "type": "PERSON",
                "description": "CEO of TechCorp",
                "email": "john@example.com",
                "phone": "+1234567890",
                "location": "New York",
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
            "name": model_dict.get("name"),
            "type": model_dict.get("type"),
            "description": model_dict.get("description"),
        }
        
        # Add metadata fields
        if model_dict.get("metadata"):
            props.update(model_dict["metadata"])
        
        # Add any additional fields (extra fields from Pydantic)
        for key, value in model_dict.items():
            if key not in ["id", "name", "type", "description", "source_chunk_ids", "metadata"] and value is not None:
                props[key] = value
        
        return props
