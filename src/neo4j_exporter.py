"""Neo4j exporter for document graphs."""

import os
from typing import Optional, Dict, Any
from src.models.document_graph import DocumentGraph
from src.models.entity import Entity
from src.models.relation import Relation


class Neo4jExporter:
    """Exports document graphs to Neo4j database."""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = None,
        database: str = "neo4j"
    ):
        """
        Initialize the Neo4j exporter.
        
        Args:
            uri: Neo4j connection URI (default: bolt://localhost:7687)
            user: Neo4j username (default: neo4j)
            password: Neo4j password (if None, reads from NEO4J_PASSWORD env var)
            database: Database name (default: neo4j)
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j driver is required. Install it with: uv add neo4j"
            )
        
        self.uri = uri
        self.user = user
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database
        
        if not self.password:
            raise ValueError(
                "Neo4j password is required. Set NEO4J_PASSWORD environment variable "
                "or pass it to the constructor."
            )
        
        self.driver = GraphDatabase.driver(uri, auth=(user, self.password))
    
    @staticmethod
    def _sanitize_property_name(name: str) -> str:
        """
        Sanitize a property name to be a valid Neo4j identifier.
        
        Args:
            name: Property name to sanitize
            
        Returns:
            Sanitized property name
        """
        # Replace invalid characters with underscores
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized
        # Ensure it's not empty
        if not sanitized:
            sanitized = "_property"
        return sanitized
    
    @staticmethod
    def _sanitize_label(label: str) -> str:
        """
        Sanitize a label name to be a valid Neo4j identifier.
        
        Args:
            label: The label name to sanitize
            
        Returns:
            Sanitized label name
        """
        # Replace spaces and hyphens with underscores
        # Remove any other special characters that aren't valid in Neo4j identifiers
        import re
        # Keep only alphanumeric and underscores, replace others with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label)
        # Remove leading/trailing underscores and multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = '_' + sanitized
        # If empty after sanitization, use a default
        if not sanitized:
            sanitized = 'ENTITY'
        return sanitized
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def export_graph(
        self,
        document_graph: DocumentGraph,
        clear_existing: bool = False,
        merge_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Export a document graph to Neo4j.
        
        Args:
            document_graph: The document graph to export
            clear_existing: Whether to clear existing nodes/relationships before importing
            merge_duplicates: Whether to merge nodes with the same name and type
        
        Returns:
            Dictionary with export statistics
        """
        with self.driver.session(database=self.database) as session:
            stats = {
                "entities_created": 0,
                "relations_created": 0,
                "entities_merged": 0,
                "errors": []
            }
            
            try:
                # Clear existing data if requested
                if clear_existing:
                    session.run("MATCH (n) DETACH DELETE n")
                    print("Cleared existing graph data.")
                
                # Create or merge entities
                for entity in document_graph.entities:
                    try:
                        # Get all properties from entity (including additional fields)
                        all_props = entity.get_all_properties()
                        
                        # Core fields that are always set
                        core_fields = {"id", "name", "type", "description"}
                        
                        # Extract additional properties (everything except core fields)
                        additional_props = {
                            k: v for k, v in all_props.items()
                            if k not in core_fields and v is not None
                        }
                        
                        # Sanitize label name to be a valid Neo4j identifier
                        label = self._sanitize_label(entity.type)
                        
                        if merge_duplicates:
                            # Use MERGE to avoid duplicates based on id
                            # Build SET clause for additional properties
                            set_clauses = [
                                "e.name = $name",
                                "e.type = $type",
                                "e.description = $description",
                                "e.created = timestamp()"
                            ]
                            match_clauses = [
                                "e.name = $name",
                                "e.type = $type",
                                "e.description = $description",
                                "e.updated = timestamp()"
                            ]
                            
                            # Add additional properties to SET clauses
                            if additional_props:
                                for prop_key, prop_value in additional_props.items():
                                    sanitized_key = Neo4jExporter._sanitize_property_name(prop_key)
                                    # Use sanitized key for property name, original key for parameter
                                    set_clauses.append(f"e.`{sanitized_key}` = ${prop_key}")
                                    match_clauses.append(f"e.`{sanitized_key}` = ${prop_key}")
                            
                            query = f"""
                            MERGE (e:`{label}` {{id: $id}})
                            ON CREATE SET {', '.join(set_clauses)}
                            ON MATCH SET {', '.join(match_clauses)}
                            RETURN e
                            """
                            
                            params = {
                                "id": entity.id,
                                "name": entity.name,
                                "type": entity.type,
                                "description": entity.description or ""
                            }
                            params.update(additional_props)
                            
                            result = session.run(query, params)
                            record = result.single()
                            if record:
                                stats["entities_created"] += 1
                        else:
                            # Use CREATE to always create new nodes
                            # Build property dictionary
                            props_dict = {
                                "id": entity.id,
                                "name": entity.name,
                                "type": entity.type,
                                "description": entity.description or ""
                            }
                            props_dict.update(additional_props)
                            
                            # Build CREATE query with all properties
                            props_str = ", ".join([f"`{Neo4jExporter._sanitize_property_name(k)}`: ${k}" for k in props_dict.keys()])
                            query = f"""
                            CREATE (e:`{label}` {{{props_str}}})
                            """
                            
                            session.run(query, props_dict)
                            stats["entities_created"] += 1
                    
                    except Exception as e:
                        stats["errors"].append(f"Error creating entity {entity.id}: {str(e)}")
                        continue
                
                # Create relationships
                for relation in document_graph.relations:
                    try:
                        # Get all properties from relation (including additional fields)
                        all_props = relation.get_all_properties()
                        
                        # Core fields that are always set
                        core_fields = {"id", "source_entity_id", "target_entity_id", "relation_type", "description", "confidence"}
                        
                        # Extract additional properties (everything except core fields)
                        additional_props = {
                            k: v for k, v in all_props.items()
                            if k not in core_fields and v is not None
                        }
                        
                        # Get relation type - sanitize it for Neo4j
                        rel_type = self._sanitize_label(relation.relation_type)
                        
                        # Build SET clauses for additional properties
                        set_clauses = [
                            "r.description = $description",
                            "r.confidence = $confidence",
                            "r.created = timestamp()"
                        ]
                        match_clauses = [
                            "r.description = $description",
                            "r.confidence = $confidence",
                            "r.updated = timestamp()"
                        ]
                        
                        # Add additional properties to SET clauses
                        if additional_props:
                            for prop_key, prop_value in additional_props.items():
                                sanitized_key = Neo4jExporter._sanitize_property_name(prop_key)
                                # Use sanitized key for property name, original key for parameter
                                set_clauses.append(f"r.`{sanitized_key}` = ${prop_key}")
                                match_clauses.append(f"r.`{sanitized_key}` = ${prop_key}")
                        
                        # Find source and target nodes by id (they can have any label now)
                        # Create relationship with properties
                        query = f"""
                        MATCH (source {{id: $source_id}})
                        MATCH (target {{id: $target_id}})
                        MERGE (source)-[r:`{rel_type}`]->(target)
                        ON CREATE SET {', '.join(set_clauses)}
                        ON MATCH SET {', '.join(match_clauses)}
                        """
                        
                        params = {
                            "source_id": relation.source_entity_id,
                            "target_id": relation.target_entity_id,
                            "description": relation.description or "",
                            "confidence": relation.confidence
                        }
                        params.update(additional_props)
                        
                        session.run(query, params)
                        stats["relations_created"] += 1
                    
                    except Exception as e:
                        stats["errors"].append(
                            f"Error creating relation {relation.id}: {str(e)}"
                        )
                        continue
                
                return stats
            
            except Exception as e:
                stats["errors"].append(f"Export error: {str(e)}")
                raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Neo4j database.
        
        Returns:
            Dictionary with database statistics
        """
        with self.driver.session(database=self.database) as session:
            # Count nodes by label (entity type)
            node_query = """
            MATCH (n)
            WITH labels(n) as labelList, n
            UNWIND labelList as label
            RETURN label, count(n) as count
            ORDER BY count DESC
            """
            
            # Count relationships by type
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
            """
            
            node_results = session.run(node_query)
            rel_results = session.run(rel_query)
            
            node_stats = {record["label"]: record["count"] for record in node_results}
            rel_stats = {record["type"]: record["count"] for record in rel_results}
            
            # Total counts
            total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            return {
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "nodes_by_label": node_stats,
                "relationships_by_type": rel_stats
            }
