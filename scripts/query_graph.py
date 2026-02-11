"""Run a natural-language query against the graph and show the top 10 most related nodes (vector search)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.graph_retriever import GraphRetriever
from src.neo4j_exporter import ENTITY_EMBEDDING_INDEX_NAME

load_dotenv()


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "payment terms and conditions"
    top_k = 10

    if not os.getenv("NEO4J_PASSWORD"):
        print("Error: NEO4J_PASSWORD not set. Set it in .env or environment.")
        sys.exit(1)

    print(f"Query: {query!r}")
    print(f"Top {top_k} most related nodes (vector similarity)")
    print("-" * 60)

    # Cypher equivalent (run in Neo4j Browser with $query_vector bound to your embedded query)
    print("\nCypher (run in Neo4j Browser; bind $query_vector to the embedding list):")
    print(f"  CALL db.index.vector.queryNodes('{ENTITY_EMBEDDING_INDEX_NAME}', {top_k}, $query_vector)")
    print("  YIELD node, score")
    print("  RETURN node.id AS id, node.name AS name, node.type AS type, score")
    print("  ORDER BY score DESC")
    print()

    with GraphRetriever(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    ) as retriever:
        with retriever._exporter.driver.session(database=retriever._exporter.database) as session:
            query_vector = retriever._embedder(query)
            result = session.run(
                "CALL db.index.vector.queryNodes($index_name, $k, $query_vector) YIELD node, score "
                "RETURN node, score ORDER BY score DESC",
                index_name=ENTITY_EMBEDDING_INDEX_NAME,
                k=top_k,
                query_vector=query_vector,
            )
            rows = list(result)

    if not rows:
        print("No nodes found. Ensure the graph was exported with embeddings (no --no-embed).")
        sys.exit(0)

    print("Results:")
    for i, r in enumerate(rows, 1):
        node = r["node"]
        score = r["score"]
        nid = node.get("id", "?")
        name = node.get("name", "?")
        ntype = node.get("type", "?")
        print(f"  {i:2}. [{ntype}] {name} (id={nid})  score={score:.4f}")

    print()
    print("Formatted context for LLM:")
    print("-" * 60)
    with GraphRetriever(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    ) as r2:
        context = r2.retrieve(query, top_k=top_k, expand_hop=1, include_score=True)
    # Avoid UnicodeEncodeError on Windows console (cp1252)
    try:
        print(context)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(context.encode(enc, errors="replace").decode(enc))


if __name__ == "__main__":
    main()
