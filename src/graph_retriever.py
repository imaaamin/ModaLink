"""Retrieve relevant graph context for a user query using vector similarity and Neo4j."""

import os
from typing import List, Optional, Callable, Dict, Any
from src.neo4j_exporter import (
    Neo4jExporter,
    ENTITY_EMBEDDING_INDEX_NAME,
)
from src import embeddings as emb


def _default_embedder(text: str) -> List[float]:
    return emb.embed_text(text)


class GraphRetriever:
    """Retrieves relevant entities and their relations from Neo4j via vector search for LLM context."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: Optional[str] = None,
        database: str = "neo4j",
        embedder: Optional[Callable[[str], List[float]]] = None,
    ):
        self._exporter = Neo4jExporter(
            uri=uri,
            user=user,
            password=password or os.getenv("NEO4J_PASSWORD"),
            database=database,
        )
        self._embedder = embedder or _default_embedder

    def close(self) -> None:
        self._exporter.close()

    def __enter__(self) -> "GraphRetriever":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        expand_hop: int = 1,
        include_score: bool = True,
    ) -> str:
        """
        Run vector similarity search for the query, optionally expand to neighbors, and return a text context for the LLM.
        
        Args:
            query: User question or search text.
            top_k: Number of top similar nodes to fetch from the vector index.
            expand_hop: How many hops of relationships to include (1 = direct neighbors).
            include_score: Whether to include similarity score in the context.
        
        Returns:
            A single string with entity and relation descriptions suitable as LLM context.
        """
        query_vector = self._embedder(query)
        with self._exporter.driver.session(database=self._exporter.database) as session:
            # Vector similarity search: CALL db.index.vector.queryNodes('indexName', k, vector)
            result = session.run(
                "CALL db.index.vector.queryNodes($index_name, $k, $query_vector) YIELD node, score RETURN node, score",
                index_name=ENTITY_EMBEDDING_INDEX_NAME,
                k=top_k,
                query_vector=query_vector,
            )
            rows = list(result)
        
        if not rows:
            return "No relevant entities found for this query."
        
        entity_ids = [r["node"].get("id") for r in rows if r["node"].get("id")]
        
        with self._exporter.driver.session(database=self._exporter.database) as session:
            if entity_ids and expand_hop >= 1:
                # Fetch nodes and their 1-hop relations/neighbors for context
                subgraph = session.run(
                    """
                    UNWIND $ids AS id
                    MATCH (n {id: id})
                    OPTIONAL MATCH (n)-[r]-(m)
                    RETURN n, type(r) AS relType, startNode(r).id AS startId, endNode(r).id AS endId,
                           m.name AS otherName, m.type AS otherType
                    """,
                    ids=entity_ids,
                )
                return _format_context_from_subgraph(
                    rows, list(subgraph), include_score=include_score
                )
            return _format_context_from_vector_results(rows, include_score=include_score)


def _format_context_from_vector_results(
    rows: list, include_score: bool = True
) -> str:
    """Format vector search results only (no expansion) as context text."""
    lines = []
    for r in rows:
        node = r["node"]
        score = r.get("score")
        name = node.get("name", "?")
        etype = node.get("type", "?")
        desc = node.get("description") or ""
        part = f"- Entity: {name} (type: {etype})"
        if desc:
            part += f". {desc}"
        if include_score and score is not None:
            part += f" [similarity: {score:.3f}]"
        lines.append(part)
    return "\n".join(lines)


def _format_context_from_subgraph(
    vector_rows: list,
    subgraph_rows: list,
    include_score: bool = True,
) -> str:
    """Format vector results plus 1-hop relations as context text."""
    # Build entity descriptions from vector results (with score)
    seen_entities = {}
    for r in vector_rows:
        node = r["node"]
        eid = node.get("id")
        if eid and eid not in seen_entities:
            seen_entities[eid] = {
                "name": node.get("name", "?"),
                "type": node.get("type", "?"),
                "description": node.get("description") or "",
                "score": r.get("score"),
            }
    
    # Build relation set (avoid duplicates)
    relations = []
    for r in subgraph_rows:
        rel_type = r.get("relType")
        start_id = r.get("startId")
        end_id = r.get("endId")
        other_name = r.get("otherName")
        other_type = r.get("otherType")
        if not rel_type:
            continue
        other = f" ({(other_name or '?')}, {(other_type or '?')})" if (other_name is not None or other_type is not None) else ""
        relations.append((start_id, rel_type, end_id, other))
    seen_rel = set()
    unique_rels = []
    for start, typ, end, other in relations:
        key = (start, typ, end)
        if key not in seen_rel:
            seen_rel.add(key)
            unique_rels.append((start, typ, end, other))
    
    lines = ["## Relevant entities (from vector search)"]
    for eid, info in seen_entities.items():
        part = f"- {info['name']} (type: {info['type']})"
        if info["description"]:
            part += f". {info['description']}"
        if include_score and info.get("score") is not None:
            part += f" [similarity: {info['score']:.3f}]"
        lines.append(part)
    
    if unique_rels:
        lines.append("")
        lines.append("## Relationships")
        for start_id, rel_type, end_id, other in unique_rels:
            start_name = seen_entities.get(start_id, {}).get("name", start_id or "?")
            end_name = seen_entities.get(end_id, {}).get("name", end_id) if (end_id and end_id in seen_entities) else (other.strip() or end_id or "?")
            lines.append(f"- {start_name} --[{rel_type}]--> {end_name}")
    
    return "\n".join(lines)
