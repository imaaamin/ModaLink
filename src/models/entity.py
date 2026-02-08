"""Entity model for extracted entities."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Represents an entity extracted from a document."""
    
    id: str = Field(description="Unique identifier for the entity")
    name: str = Field(description="Name or label of the entity")
    type: str = Field(description="Type/category of the entity (e.g., PERSON, ORGANIZATION, LOCATION)")
    description: Optional[str] = Field(default=None, description="Additional description or context")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
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
            if key not in ["id", "name", "type", "description", "metadata"] and value is not None:
                props[key] = value
        
        return props
