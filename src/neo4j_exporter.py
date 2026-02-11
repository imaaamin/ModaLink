"""Neo4j exporter for document graphs."""

import os
import json
from typing import Optional, Dict, Any, Callable, List
from src.models.document_graph import DocumentGraph
from src.models.document import Document
from src.models.chunk import Chunk
from src.models.entity import Entity
from src.models.relation import Relation

# Vector index name and property used for entity embeddings (Neo4j 5.13+)
ENTITY_EMBEDDING_INDEX_NAME = "entity_embedding"
EMBEDDING_PROPERTY = "embedding"
ENTITY_LABEL_FOR_INDEX = "Entity"


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
    def _sanitize_identifier(name: str, fallback: str = "_property") -> str:
        """
        Sanitize a string to be a safe Neo4j identifier (label, property name, or parameter name).

        Only allows ASCII letters, digits, and underscores. This prevents injection
        via unicode characters, backticks, or other special characters from LLM output.

        Args:
            name: The identifier to sanitize
            fallback: Default value if sanitization produces an empty string

        Returns:
            Sanitized identifier containing only [a-zA-Z0-9_]
        """
        import re
        # Strip to ASCII alphanumeric and underscores only
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Collapse consecutive underscores and strip leading/trailing
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        # Ensure it starts with a letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        # Fallback if empty
        if not sanitized:
            sanitized = fallback
        return sanitized

    @classmethod
    def _sanitize_property_name(cls, name: str) -> str:
        """Sanitize a property name to be a valid Neo4j identifier."""
        return cls._sanitize_identifier(name, fallback="_property")

    @classmethod
    def _sanitize_label(cls, label: str) -> str:
        """Sanitize a label name to be a valid Neo4j identifier."""
        return cls._sanitize_identifier(label, fallback="ENTITY")
    
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
    
    _SOURCE_TAG = "document_extractor"

    def export_graph(
        self,
        document_graph: DocumentGraph,
        clear_existing: bool = False,
        merge_duplicates: bool = True,
        embedder: Optional[Callable[[Entity], List[float]]] = None,
        embedding_dimension: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Export a document graph to Neo4j.

        Args:
            document_graph: The document graph to export
            clear_existing: Whether to clear nodes/relationships created by this tool before importing
            merge_duplicates: Whether to merge nodes with the same name and type
            embedder: Optional callable Entity -> list[float] to store vector embeddings on nodes (for Neo4j 5.13+ vector index).
            embedding_dimension: Required when embedder is set; used for vector index creation (e.g. 384 for all-MiniLM-L6-v2).
        
        Returns:
            Dictionary with export statistics
        """
        if embedder is not None and embedding_dimension is None:
            raise ValueError("embedding_dimension is required when embedder is provided.")
        use_embeddings = embedder is not None

        with self.driver.session(database=self.database) as session:
            stats = {
                "entities_created": 0,
                "relations_created": 0,
                "entities_merged": 0,
                "errors": []
            }

            try:
                # Clear only data created by this tool (scoped by _source tag)
                if clear_existing:
                    session.run("MATCH (n) DETACH DELETE n")
                    # Drop vector index so we don't leave a stale index when re-importing without embeddings
                    if not use_embeddings:
                        try:
                            session.run(f"DROP INDEX {ENTITY_EMBEDDING_INDEX_NAME} IF EXISTS")
                        except Exception:
                            pass
                    print("Cleared existing graph data.")
                
                # Create Document node and Chunk chain (Document)->first chunk, chunk->next chunk
                if document_graph.document and document_graph.chunks:
                    doc = document_graph.document
                    doc_props = doc.for_neo4j()
                    session.run(
                        """
                        MERGE (d:Document {docId: $docId})
                        SET d.title = $title, d.source = $source, d.publishedDate = $publishedDate
                        """,
                        **doc_props,
                    )
                    for ch in document_graph.chunks:
                        session.run(
                            """
                            MERGE (c:Chunk {id: $id})
                            SET c.text = $text, c.index = $index, c.document_id = $document_id
                            """,
                            id=ch.id,
                            text=ch.text,
                            index=ch.index,
                            document_id=ch.document_id,
                        )
                    # (Document)-[:FIRST_CHUNK]->(first Chunk)
                    if document_graph.chunks:
                        first_id = document_graph.chunks[0].id
                        session.run(
                            """
                            MATCH (d:Document {docId: $docId})
                            MATCH (c:Chunk {id: $chunk_id})
                            MERGE (d)-[:FIRST_CHUNK]->(c)
                            """,
                            docId=doc_props["docId"],
                            chunk_id=first_id,
                        )
                    # (Chunk)-[:NEXT_CHUNK]->(next Chunk)
                    for i in range(len(document_graph.chunks) - 1):
                        session.run(
                            """
                            MATCH (a:Chunk {id: $from_id})
                            MATCH (b:Chunk {id: $to_id})
                            MERGE (a)-[:NEXT_CHUNK]->(b)
                            """,
                            from_id=document_graph.chunks[i].id,
                            to_id=document_graph.chunks[i + 1].id,
                        )
                
                # Create or merge entities
                for entity in document_graph.entities:
                    try:
                        # Get all properties from entity (including additional fields)
                        all_props = entity.get_all_properties()
                        
                        # Core fields that are always set
                        core_fields = {"id", "name", "type", "description"}
                        
                        # Extract additional properties (everything except core fields)
                        # Neo4j can handle lists of primitives, but not nested maps or arrays of maps
                        additional_props = {}
                        for k, v in all_props.items():
                            if k not in core_fields and v is not None:
                                # Handle different types
                                if isinstance(v, (str, int, float, bool)):
                                    additional_props[k] = v
                                elif isinstance(v, list):
                                    # Check if list contains only primitives
                                    if all(isinstance(item, (str, int, float, bool)) for item in v):
                                        additional_props[k] = v
                                    else:
                                        # List contains complex types - convert to JSON string
                                        additional_props[k] = json.dumps(v)
                                elif isinstance(v, dict):
                                    # Dict with nested structure - convert to JSON string
                                    additional_props[k] = json.dumps(v)
                                else:
                                    # Convert other types to string
                                    additional_props[k] = str(v)
                        
                        if use_embeddings:
                            emb = embedder(entity)
                            if len(emb) != embedding_dimension:
                                raise ValueError(
                                    f"Embedder returned dimension {len(emb)}, expected {embedding_dimension}"
                                )
                            # Ensure list of Python floats for Neo4j driver serialization
                            additional_props[EMBEDDING_PROPERTY] = [float(x) for x in emb]
                        
                        # Sanitize label name to be a valid Neo4j identifier
                        label = self._sanitize_label(entity.type)
                        # When using embeddings, add Entity label so one vector index covers all nodes
                        labels_clause = f"`{label}`:`{ENTITY_LABEL_FOR_INDEX}`" if use_embeddings else f"`{label}`"
                        
                        if merge_duplicates:
                            # Use MERGE to avoid duplicates based on id
                            # Build SET clause for additional properties
                            set_clauses = [
                                "e.name = $name",
                                "e.type = $type",
                                "e.description = $description",
                                "e._source = $source",
                                "e.created = timestamp()"
                            ]
                            match_clauses = [
                                "e.name = $name",
                                "e.type = $type",
                                "e.description = $description",
                                "e._source = $source",
                                "e.updated = timestamp()"
                            ]
                            
                            # Add additional properties to SET clauses (skip embedding; set explicitly below)
                            if additional_props:
                                for prop_key, prop_value in additional_props.items():
                                    if prop_key == EMBEDDING_PROPERTY:
                                        continue
                                    sanitized_key = Neo4jExporter._sanitize_property_name(prop_key)
                                    set_clauses.append(f"e.`{sanitized_key}` = ${prop_key}")
                                    match_clauses.append(f"e.`{sanitized_key}` = ${prop_key}")
                            
                            # Set embedding via a dedicated param so it is always written when use_embeddings
                            if use_embeddings and EMBEDDING_PROPERTY in additional_props:
                                set_clauses.append(f"e.`{EMBEDDING_PROPERTY}` = $embedding_vec")
                                match_clauses.append(f"e.`{EMBEDDING_PROPERTY}` = $embedding_vec")
                            
                            query = f"""
                            MERGE (e:{labels_clause} {{id: $id}})
                            ON CREATE SET {', '.join(set_clauses)}
                            ON MATCH SET {', '.join(match_clauses)}
                            RETURN e
                            """

                            params = {
                                "id": entity.id,
                                "name": entity.name,
                                "type": entity.type,
                                "description": entity.description or "",
                                "source": self._SOURCE_TAG
                            }
                            params.update(additional_props)
                            if use_embeddings and EMBEDDING_PROPERTY in additional_props:
                                params["embedding_vec"] = additional_props[EMBEDDING_PROPERTY]
                            
                            result = session.run(query, params)
                            record = result.single()
                            if record:
                                stats["entities_created"] += 1
                        else:
                            # Use CREATE to always create new nodes
                            props_dict = {
                                "id": entity.id,
                                "name": entity.name,
                                "type": entity.type,
                                "description": entity.description or "",
                                "_source": self._SOURCE_TAG
                            }
                            props_dict.update(additional_props)
                            # Embedding via dedicated param so it is always written
                            if use_embeddings and EMBEDDING_PROPERTY in props_dict:
                                props_dict["embedding_vec"] = props_dict.pop(EMBEDDING_PROPERTY)
                            props_str = ", ".join(
                                [f"`{Neo4jExporter._sanitize_property_name(k)}`: ${k}" for k in props_dict.keys()]
                            )
                            if use_embeddings and "embedding_vec" in props_dict:
                                props_str = props_str.replace(
                                    "`embedding_vec`: $embedding_vec",
                                    f"`{EMBEDDING_PROPERTY}`: $embedding_vec",
                                )
                            query = f"""
                            CREATE (e:{labels_clause} {{{props_str}}})
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
                        # Neo4j can handle lists of primitives, but not nested maps or arrays of maps
                        additional_props = {}
                        for k, v in all_props.items():
                            if k not in core_fields and v is not None:
                                # Handle different types
                                if isinstance(v, (str, int, float, bool)):
                                    additional_props[k] = v
                                elif isinstance(v, list):
                                    # Check if list contains only primitives
                                    if all(isinstance(item, (str, int, float, bool)) for item in v):
                                        additional_props[k] = v
                                    else:
                                        # List contains complex types - convert to JSON string
                                        additional_props[k] = json.dumps(v)
                                elif isinstance(v, dict):
                                    # Dict with nested structure - convert to JSON string
                                    additional_props[k] = json.dumps(v)
                                else:
                                    # Convert other types to string
                                    additional_props[k] = str(v)
                        
                        # Get relation type - sanitize it for Neo4j
                        rel_type = self._sanitize_label(relation.relation_type)
                        
                        # Build SET clauses for additional properties
                        set_clauses = [
                            "r.id = $rel_id",
                            "r.description = $description",
                            "r.confidence = $confidence",
                            "r._source = $source",
                            "r.created = timestamp()"
                        ]
                        match_clauses = [
                            "r.id = $rel_id",
                            "r.description = $description",
                            "r.confidence = $confidence",
                            "r._source = $source",
                            "r.updated = timestamp()"
                        ]
                        
                        # Add additional properties to SET clauses
                        sanitized_additional = {}
                        if additional_props:
                            for prop_key, prop_value in additional_props.items():
                                sanitized_key = Neo4jExporter._sanitize_property_name(prop_key)
                                set_clauses.append(f"r.`{sanitized_key}` = $`{sanitized_key}`")
                                match_clauses.append(f"r.`{sanitized_key}` = $`{sanitized_key}`")
                                sanitized_additional[sanitized_key] = prop_value

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
                            "rel_id": relation.id,
                            "description": relation.description or "",
                            "confidence": relation.confidence,
                            "source": self._SOURCE_TAG
                        }
                        params.update(sanitized_additional)
                        
                        session.run(query, params)
                        stats["relations_created"] += 1
                    
                    except Exception as e:
                        stats["errors"].append(
                            f"Error creating relation {relation.id}: {str(e)}"
                        )
                        continue
                
                # (Entity)-[:MENTIONED_IN]->(Chunk) for each chunk in entity.source_chunk_ids
                for entity in document_graph.entities:
                    chunk_ids = getattr(entity, "source_chunk_ids", None) or []
                    if not chunk_ids:
                        continue
                    for chunk_id in chunk_ids:
                        try:
                            session.run(
                                """
                                MATCH (e) WHERE e.id = $entity_id AND NOT 'Chunk' IN labels(e)
                                MATCH (c:Chunk {id: $chunk_id})
                                MERGE (e)-[:MENTIONED_IN]->(c)
                                """,
                                entity_id=entity.id,
                                chunk_id=chunk_id,
                            )
                        except Exception as e:
                            stats["errors"].append(f"Error linking entity {entity.id} to chunk {chunk_id}: {str(e)}")
                
                if use_embeddings and embedding_dimension is not None:
                    self._create_entity_vector_index(session, embedding_dimension, stats)
                
                return stats
            
            except Exception as e:
                stats["errors"].append(f"Export error: {str(e)}")
                raise
    
    def _create_entity_vector_index(
        self, session, embedding_dimension: int, stats: Dict[str, Any]
    ) -> None:
        """Create the vector index for Entity.embedding (Neo4j 5.13+). No-op on failure (e.g. older server)."""
        try:
            session.run(f"DROP INDEX {ENTITY_EMBEDDING_INDEX_NAME} IF EXISTS")
        except Exception:
            pass
        try:
            session.run(
                f"""
                CREATE VECTOR INDEX {ENTITY_EMBEDDING_INDEX_NAME}
                FOR (n:{ENTITY_LABEL_FOR_INDEX}) ON (n.`{EMBEDDING_PROPERTY}`)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dim=embedding_dimension,
            )
        except Exception as e:
            stats["errors"].append(f"Could not create vector index (Neo4j 5.13+ required): {e}")

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
