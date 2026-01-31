"""Entity extraction module using LangChain and LLMs."""

import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.models.entity import Entity
import json


class EntityExtractor:
    """Extracts entities from document text using LLM."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.0):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: Name of the Groq LLM model to use (e.g., "openai/gpt-oss-120b", "llama-3.1-70b-versatile", "mixtral-8x7b-32768")
            temperature: Temperature for the LLM
        """
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.parser = JsonOutputParser(pydantic_object=Entity)
        
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text. First extracts all entities, then categorizes them into types.
        
        Args:
            text: Text content to extract entities from
        
        Returns:
            List of extracted entities with dynamically assigned types
        """
        # Step 1: Extract all entities from the document without predefined types
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert entity extraction system. Extract all meaningful entities from the given text.

For each entity, provide:
- id: A unique identifier (e.g., "entity_1", "entity_2")
- name: The exact name or phrase as it appears in the text
- description: A brief description or context about the entity

Additionally, extract ALL available properties for each entity based on the text context. Common properties include:
- For PERSON: email, phone, title, role, age, birth_date, location, address, etc.
- For ORGANIZATION: website, industry, founded_date, location, address, size, revenue, etc.
- For LOCATION: country, city, coordinates, population, etc.
- For DATE: year, month, day, format, etc.
- For PRODUCT: price, category, brand, model, etc.
- For EVENT: start_date, end_date, location, participants, etc.
- For DOCUMENT: author, publication_date, version, pages, etc.
- And any other relevant properties mentioned in the text

Include these properties directly in the entity object (not just in metadata). Extract as many properties as are available in the text.
Return a JSON array of entities. Each entity should be unique - do not duplicate entities with the same name."""),
            ("human", "Extract all entities from the following text. Include all available properties for each entity:\n\n{text}\n\nReturn only valid JSON array of entities with all their properties.")
        ])
        
        extract_chain = extract_prompt | self.llm | JsonOutputParser()
        
        try:
            # Extract entities first
            extract_result = extract_chain.invoke({"text": text})
            
            # Handle both list and dict responses
            if isinstance(extract_result, dict) and "entities" in extract_result:
                entities_data = extract_result["entities"]
            elif isinstance(extract_result, list):
                entities_data = extract_result
            else:
                entities_data = [extract_result] if extract_result else []
            
            if not entities_data:
                print("Warning: No entities extracted in first step")
                return []
            
            print(f"Extracted {len(entities_data)} entities in first step, now categorizing...")
            
            # Step 2: Categorize entities into types
            categorize_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert entity categorization system. Categorize the extracted entities into appropriate types.

Given a list of entities, assign each one a type that best describes it. Common types include:
PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PRODUCT, EVENT, TECHNOLOGY, CONCEPT, etc.

For each entity, provide:
- id: The original entity id
- type: A category/type that best describes this entity
- name: The original entity name
- description: The original description
- ALL original properties: Preserve all properties that were extracted (email, phone, location, dates, etc.)

IMPORTANT: Keep all properties from the original entity extraction. Do not remove any properties - only add the type field.
Return a JSON array with the same entities but with type assigned to each, preserving all other properties."""),
                ("human", "Categorize these entities into types. Preserve all properties from the original extraction:\n\n{entities_json}\n\nReturn only valid JSON array of entities with types assigned, keeping all original properties.")
            ])
            
            categorize_chain = categorize_prompt | self.llm | JsonOutputParser()
            
            # Convert entities to JSON string for categorization
            entities_json = json.dumps(entities_data, indent=2)
            
            categorize_result = categorize_chain.invoke({"entities_json": entities_json})
            
            # Handle categorization result
            if isinstance(categorize_result, dict) and "entities" in categorize_result:
                categorized_entities = categorize_result["entities"]
            elif isinstance(categorize_result, list):
                categorized_entities = categorize_result
            else:
                categorized_entities = [categorize_result] if categorize_result else []
            
            if not categorized_entities:
                print("Warning: No entities returned from categorization step")
                # Fallback: use original entities with UNKNOWN type
                categorized_entities = []
                for entity_data in entities_data:
                    entity_data_copy = entity_data.copy()
                    entity_data_copy["type"] = "UNKNOWN"
                    categorized_entities.append(entity_data_copy)
            
            # Parse categorized entities into Entity objects
            entities = []
            for i, entity_data in enumerate(categorized_entities):
                try:
                    # Ensure id is present
                    if "id" not in entity_data:
                        entity_data["id"] = f"entity_{i+1}"
                    
                    # Ensure type is present (fallback to UNKNOWN if not categorized)
                    if "type" not in entity_data:
                        entity_data["type"] = "UNKNOWN"
                    
                    entity = Entity(**entity_data)
                    entities.append(entity)
                except Exception as e:
                    print(f"Error parsing entity {i}: {e}")
                    continue
            
            return entities
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []
