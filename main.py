"""Main entry point for the document extractor."""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.agents.extraction_graph import DocumentExtractionGraph

# Add scripts to path for graph_visualizer
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from graph_visualizer import GraphVisualizer
import json


def main():
    """Main function to run the document extractor."""
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GROQ_API_KEY"):
        print("Error: No LLM API key found.")
        print("Please set GOOGLE_API_KEY or GROQ_API_KEY in your .env file.")
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract entities and relations from documents using LangGraph"
    )
    parser.add_argument(
        "document_path",
        help="Path to the document file to process"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing Neo4j data (default: clear graph first, then import new extraction)"
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip storing vector embeddings on Neo4j nodes (by default embeddings are stored for semantic search, Neo4j 5.13+)"
    )
    
    args = parser.parse_args()
    document_path = args.document_path
    
    if not Path(document_path).exists():
        print(f"Error: Document not found at {document_path}")
        sys.exit(1)
    
    print(f"Processing document: {document_path}")
    print("-" * 50)
    
    # Initialize the extraction graph
    extractor = DocumentExtractionGraph(
        temperature=0.0
    )
    
    # Extract entities and relations
    try:
        print("Extracting entities and relations...")
        graph = extractor.extract(document_path)
        
        # Display results
        print(f"\n✓ Extraction complete!")
        print(f"  - Entities found: {len(graph.entities)}")
        print(f"  - Relations found: {len(graph.relations)}")
        
        # Show statistics
        visualizer = GraphVisualizer()
        stats = visualizer.get_statistics(graph)
        
        print("\nEntity Types:")
        for entity_type, count in stats["entity_types"].items():
            print(f"  - {entity_type}: {count}")
        
        print("\nRelation Types:")
        for relation_type, count in stats["relation_types"].items():
            print(f"  - {relation_type}: {count}")
        
        # Export results
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        doc_name = Path(document_path).stem
        json_path = output_dir / f"{doc_name}_graph.json"
        graphml_path = output_dir / f"{doc_name}_graph.graphml"
        
        visualizer.export_to_json(graph, str(json_path))
        visualizer.export_to_graphml(graph, str(graphml_path))
        
        print(f"\n✓ Results exported:")
        print(f"  - JSON: {json_path}")
        print(f"  - GraphML: {graphml_path}")
        
        # Export to Neo4j if configured
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if neo4j_uri and neo4j_password:
            try:
                from src.neo4j_exporter import Neo4jExporter
                from src import embeddings as emb
                
                print("\nExporting to Neo4j...")
                clear_first = not args.merge
                use_embeddings = not getattr(args, "no_embed", False)
                if clear_first:
                    print("Clearing existing graph, then importing new extraction...")
                if use_embeddings:
                    print("Computing embeddings for vector search...")
                    # Validate embedder once so failures surface before export
                    try:
                        emb.get_embedding_dimension()
                        emb.embed_text("test")
                    except Exception as e:
                        print(f"\n⚠ Embedding setup failed: {e}")
                        print("  Fix the error above or run with --no-embed to skip embeddings.")
                        raise
                with Neo4jExporter(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                    database=os.getenv("NEO4J_DATABASE", "neo4j")
                ) as exporter:
                    stats = exporter.export_graph(
                        graph,
                        clear_existing=clear_first,
                        merge_duplicates=True,
                        embedder=emb.embed_entity if use_embeddings else None,
                        embedding_dimension=emb.get_embedding_dimension() if use_embeddings else None,
                    )
                    
                    print(f"✓ Neo4j export complete!")
                    print(f"  - Entities created: {stats['entities_created']}")
                    print(f"  - Relations created: {stats['relations_created']}")
                    if stats['errors']:
                        print(f"  - Errors: {len(stats['errors'])}")
                        for error in stats['errors'][:5]:
                            print(f"    - {error}")
                    
                    # Show database statistics
                    db_stats = exporter.get_statistics()
                    print(f"\nNeo4j Database Statistics:")
                    print(f"  - Total nodes: {db_stats['total_nodes']}")
                    print(f"  - Total relationships: {db_stats['total_relationships']}")
            except ImportError:
                print("\n⚠ Neo4j export skipped (neo4j package not installed)")
                print("  Install with: uv add neo4j")
            except Exception as e:
                print(f"\n⚠ Neo4j export failed: {e}")
        
        # Print sample entities and relations
        print("\nSample Entities:")
        for entity in graph.entities[:5]:
            print(f"  - [{entity.type}] {entity.name} (ID: {entity.id})")
            if entity.description:
                print(f"    Description: {entity.description}")
        
        if len(graph.entities) > 5:
            print(f"  ... and {len(graph.entities) - 5} more entities")
        
        print("\nSample Relations:")
        for relation in graph.relations[:5]:
            source = graph.get_entity_by_id(relation.source_entity_id)
            target = graph.get_entity_by_id(relation.target_entity_id)
            if source and target:
                print(f"  - {source.name} --[{relation.relation_type}]--> {target.name}")
                if relation.description:
                    print(f"    Description: {relation.description}")
                # Show relation attributes (e.g. start_date, end_date, role)
                attrs = relation.get_all_properties()
                extra = {k: v for k, v in attrs.items() if k not in (
                    "id", "source_entity_id", "target_entity_id", "relation_type", "description", "confidence"
                ) and v is not None}
                if extra:
                    print(f"    Attributes: {extra}")
        
        if len(graph.relations) > 5:
            print(f"  ... and {len(graph.relations) - 5} more relations")
        
    except ValueError as e:
        if "API key" in str(e) or "API_KEY" in str(e):
            print(f"\n✗ {e}")
            sys.exit(1)
        raise
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "invalid_api_key" in msg or "Invalid API Key" in msg:
            print(f"\n✗ API key is invalid or rejected.")
            print("  Check GOOGLE_API_KEY or GROQ_API_KEY in your .env file.")
            sys.exit(1)
        print(f"\n✗ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
