"""Helper functions for integration tests."""

import json
from typing import List, Tuple, Dict, Any
from src.neo4j_exporter import Neo4jExporter


def check_all_entities(json_path: str, neo4j_config: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Check which entities from JSON are in Neo4j.
    
    Args:
        json_path: Path to the JSON graph file
        neo4j_config: Neo4j configuration dictionary
        
    Returns:
        Tuple of (missing_entity_ids, extra_entity_ids)
    """
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entity_ids = {e['id'] for e in data.get('entities', [])}
    
    # Check Neo4j
    with Neo4jExporter(
        uri=neo4j_config["uri"],
        user=neo4j_config["user"],
        password=neo4j_config["password"],
        database=neo4j_config["database"]
    ) as exporter:
        with exporter.driver.session(database=neo4j_config["database"]) as session:
            # Get all entity IDs from Neo4j
            query = "MATCH (n) WHERE n.id IS NOT NULL RETURN n.id as id, labels(n) as labels, n.name as name"
            result = session.run(query)
            neo4j_ids = {}
            for record in result:
                neo4j_ids[record["id"]] = {
                    "labels": record["labels"],
                    "name": record["name"]
                }
            
            # Find missing entities
            missing = sorted(list(entity_ids - set(neo4j_ids.keys())))
            
            # Find extra entities in Neo4j
            extra = sorted(list(set(neo4j_ids.keys()) - entity_ids))
            
            return missing, extra


def check_all_relations(json_path: str, neo4j_config: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Check which relations from JSON are in Neo4j.
    
    Args:
        json_path: Path to the JSON graph file
        neo4j_config: Neo4j configuration dictionary
        
    Returns:
        Tuple of (missing_relation_ids, extra_relation_ids)
    """
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    relation_ids = {r['id'] for r in data.get('relations', [])}
    
    # Check Neo4j
    with Neo4jExporter(
        uri=neo4j_config["uri"],
        user=neo4j_config["user"],
        password=neo4j_config["password"],
        database=neo4j_config["database"]
    ) as exporter:
        with exporter.driver.session(database=neo4j_config["database"]) as session:
            # Get all relation IDs from Neo4j
            query = """
            MATCH (source)-[r]->(target)
            WHERE source.id IS NOT NULL AND target.id IS NOT NULL AND r.id IS NOT NULL
            RETURN r.id as rel_id, source.id as source_id, target.id as target_id, 
                   type(r) as rel_type, keys(r) as properties
            """
            result = session.run(query)
            neo4j_relations = set()
            neo4j_relation_details = {}
            
            for record in result:
                rel_id = record.get("rel_id")
                if rel_id:
                    neo4j_relations.add(rel_id)
                    neo4j_relation_details[rel_id] = {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "rel_type": record["rel_type"]
                    }
            
            # Find missing relations
            missing = sorted(list(relation_ids - neo4j_relations))
            
            # Find extra relations in Neo4j
            extra = sorted(list(neo4j_relations - relation_ids))
            
            return missing, extra


def get_entity_from_neo4j(entity_id: str, neo4j_config: Dict[str, str]) -> Dict[str, Any]:
    """
    Get an entity from Neo4j by ID.
    
    Args:
        entity_id: The entity ID to look up
        neo4j_config: Neo4j configuration dictionary
        
    Returns:
        Dictionary with entity properties, or None if not found
    """
    with Neo4jExporter(
        uri=neo4j_config["uri"],
        user=neo4j_config["user"],
        password=neo4j_config["password"],
        database=neo4j_config["database"]
    ) as exporter:
        with exporter.driver.session(database=neo4j_config["database"]) as session:
            query = """
            MATCH (n {id: $entity_id})
            RETURN n, labels(n) as labels
            """
            result = session.run(query, entity_id=entity_id)
            record = result.single()
            
            if record:
                node = record["n"]
                return {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "type": node.get("type"),
                    "labels": record["labels"],
                    "properties": dict(node)
                }
            return None
