"""Relation extraction module using LangChain and LLMs."""

import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.models.entity import Entity
from src.models.relation import Relation
import json


class RelationExtractor:
    """Extracts relations between entities from document text using LLM."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.0):
        """
        Initialize the relation extractor.
        
        Args:
            model_name: Name of the Groq LLM model to use (e.g., "openai/gpt-oss-120b", "llama-3.1-70b-versatile", "mixtral-8x7b-32768")
            temperature: Temperature for the LLM
        """
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.parser = JsonOutputParser(pydantic_object=Relation)
        
    def extract_relations(
        self, 
        text: str, 
        entities: List[Entity]
    ) -> List[Relation]:
        """
        Extract relations between entities from text. Dynamically identifies relation types.
        
        Args:
            text: Text content to extract relations from
            entities: List of entities to find relations between
        
        Returns:
            List of extracted relations with dynamically identified relation types
        """
        # Create entity mapping for reference
        entities_info = [
            f"ID: {e.id}, Name: {e.name}, Type: {e.type}"
            for e in entities
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert relation extraction system. Extract all relations between the given entities from the document text.

Available entities:
{entities_info}

For each relation, provide:
- id: A unique identifier (e.g., "relation_1", "relation_2")
- source_entity_id: The ID of the source entity
- target_entity_id: The ID of the target entity
- relation_type: A descriptive type of the relation (e.g., "WORKS_FOR", "LOCATED_IN", "OWNS", "PART_OF", "FOUNDED", "MANAGES", "MENTIONED_IN", "REFERENCED_IN", "OCCURRED_ON", "PUBLISHED_BY", etc.)
- description: A brief description of the relation as it appears in the text
- confidence: A confidence score between 0 and 1 (optional)

Additionally, extract ALL available properties for each relation based on the text context. Common properties include:
- start_date, end_date: For temporal relationships
- role, position, title: For employment/role relationships
- amount, value, price: For financial relationships
- location, address: For location-based relationships
- duration, frequency: For time-based relationships
- method, reason, purpose: For action relationships
- And any other relevant properties mentioned in the text

Include these properties directly in the relation object (not just in metadata). Extract as many properties as are available in the text.

IMPORTANT: Extract ALL possible relations, including:
- Direct relationships (e.g., A works for B, A owns B)
- Contextual relationships (e.g., entities mentioned in the same section, entities related to the same topic)
- Temporal relationships (e.g., dates associated with entities, events occurring on dates)
- Reference relationships (e.g., publications referencing entities, entities mentioned in documents)

Try to connect as many entities as possible based on the document context. Even weak or indirect relationships are valuable.
Identify the relation type based on the context in the document - do not use a predefined list.
Return a JSON array of relations with all their properties."""),
            ("human", "Extract all relations between the given entities from the following text. Include all available properties for each relation (dates, roles, amounts, etc.):\n\n{text}\n\nReturn only valid JSON array of relations with all their properties.")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({
                "text": text,
                "entities_info": "\n".join(entities_info)
            })
            
            # Handle both list and dict responses
            if isinstance(result, dict) and "relations" in result:
                relations_data = result["relations"]
            elif isinstance(result, list):
                relations_data = result
            else:
                relations_data = [result] if result else []
            
            relations = []
            for i, relation_data in enumerate(relations_data):
                try:
                    # Validate entity IDs exist
                    source_id = relation_data.get("source_entity_id")
                    target_id = relation_data.get("target_entity_id")
                    
                    if not source_id or not target_id:
                        continue
                    
                    # Check if entities exist
                    entity_ids = {e.id for e in entities}
                    if source_id not in entity_ids or target_id not in entity_ids:
                        continue
                    
                    # Ensure id is present
                    if "id" not in relation_data:
                        relation_data["id"] = f"relation_{i+1}"
                    
                    relation = Relation(**relation_data)
                    relations.append(relation)
                except Exception as e:
                    print(f"Error parsing relation {i}: {e}")
                    continue
            
            return relations
            
        except Exception as e:
            print(f"Error extracting relations: {e}")
            return []
