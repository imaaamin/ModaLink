"""Script to export a graph JSON file to Neo4j."""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.neo4j_exporter import Neo4jExporter
from src.models.document_graph import DocumentGraph
from src.models.document import Document
from src.models.chunk import Chunk
from src.models.entity import Entity
from src.models.relation import Relation
from src import embeddings as emb


def main():
    """Main function to export graph to Neo4j."""
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python export_to_neo4j.py <graph_json_file> [--clear] [--no-embed]")
        print("\nOptions:")
        print("  --clear    Clear existing graph data before importing")
        print("  --no-embed Skip vector embeddings (by default embeddings are stored for semantic search, Neo4j 5.13+)")
        print("\nExample:")
        print('  python export_to_neo4j.py "outputs/Legal _ Uber_graph.json"')
        print('  python export_to_neo4j.py "outputs/Legal _ Uber_graph.json" --clear')
        sys.exit(1)
    
    json_path = sys.argv[1]
    clear_existing = "--clear" in sys.argv
    use_embeddings = "--no-embed" not in sys.argv
    
    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    # Load Neo4j configuration
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    if not neo4j_password:
        print("Error: NEO4J_PASSWORD not found in environment variables.")
        print("Please set it in your .env file or environment.")
        sys.exit(1)
    
    # Load graph from JSON
    print(f"Loading graph from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Reconstruct DocumentGraph
    entities = [Entity(**e) for e in data['entities']]
    relations = [Relation(**r) for r in data['relations']]
    document = None
    if data.get('document'):
        document = Document(**data['document'])
    chunks = [Chunk(**c) for c in data.get('chunks', [])]
    
    graph = DocumentGraph(
        entities=entities,
        relations=relations,
        document_id=data.get('document_id'),
        document=document,
        chunks=chunks,
        metadata=data.get('metadata', {}),
    )
    
    print(f"Loaded {len(graph.entities)} entities and {len(graph.relations)} relations")
    if graph.document:
        print(f"  Document: {graph.document.doc_id} ({len(graph.chunks)} chunks)")
    
    # Export to Neo4j
    try:
        print(f"\nConnecting to Neo4j at {neo4j_uri}...")
        with Neo4jExporter(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database
        ) as exporter:
            if clear_existing:
                print("⚠ Clearing existing graph data...")
            
            if use_embeddings:
                print("Computing embeddings and exporting graph to Neo4j...")
            else:
                print("Exporting graph to Neo4j...")
            stats = exporter.export_graph(
                graph,
                clear_existing=clear_existing,
                merge_duplicates=True,
                embedder=emb.embed_entity if use_embeddings else None,
                embedding_dimension=emb.get_embedding_dimension() if use_embeddings else None,
                chunk_embedder=emb.embed_chunk if use_embeddings else None,
            )
            
            print(f"\n✓ Export complete!")
            print(f"  - Entities created: {stats['entities_created']}")
            print(f"  - Relations created: {stats['relations_created']}")
            
            if stats['errors']:
                print(f"\n⚠ Errors encountered: {len(stats['errors'])}")
                for error in stats['errors'][:10]:
                    print(f"  - {error}")
            
            # Show database statistics
            print("\nDatabase Statistics:")
            db_stats = exporter.get_statistics()
            print(f"  - Total nodes: {db_stats['total_nodes']}")
            print(f"  - Total relationships: {db_stats['total_relationships']}")
            
            if db_stats['nodes_by_label']:
                print("\n  Nodes by label:")
                for label, count in list(db_stats['nodes_by_label'].items())[:10]:
                    print(f"    - {label}: {count}")
            
            if db_stats['relationships_by_type']:
                print("\n  Relationships by type:")
                for rel_type, count in list(db_stats['relationships_by_type'].items())[:10]:
                    print(f"    - {rel_type}: {count}")
    
    except ImportError:
        print("Error: neo4j package not installed.")
        print("Install with: uv add neo4j")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
