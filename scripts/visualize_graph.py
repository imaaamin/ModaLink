"""Script to visualize a GraphML file or JSON graph file."""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from graph_visualizer import GraphVisualizer
import json
from src.models.document_graph import DocumentGraph

load_dotenv()


def visualize_from_json(json_path: str, output_image: str = None):
    """Visualize graph from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Reconstruct DocumentGraph from JSON
    from src.models.entity import Entity
    from src.models.relation import Relation
    from src.models.document import Document
    from src.models.chunk import Chunk
    
    entities = [Entity(**e) for e in data['entities']]
    relations = [Relation(**r) for r in data['relations']]
    document = Document(**data['document']) if data.get('document') else None
    chunks = [Chunk(**c) for c in data.get('chunks', [])]
    
    graph = DocumentGraph(
        entities=entities,
        relations=relations,
        document_id=data.get('document_id'),
        document=document,
        chunks=chunks,
        metadata=data.get('metadata', {}),
    )
    
    visualizer = GraphVisualizer()
    visualizer.visualize(graph, file_path=output_image, show=True)


def visualize_from_graphml(graphml_path: str, output_image: str = None):
    """Visualize graph from GraphML file."""
    import networkx as nx
    
    # Read GraphML file
    G = nx.read_graphml(graphml_path)
    
    # Convert to DocumentGraph format
    from src.models.entity import Entity
    from src.models.relation import Relation
    
    entities = []
    relations = []
    
    # Extract entities from nodes
    for node_id, attrs in G.nodes(data=True):
        entities.append(Entity(
            id=node_id,
            name=attrs.get('name', node_id),
            type=attrs.get('type', 'UNKNOWN'),
            description=attrs.get('description'),
            metadata={k: v for k, v in attrs.items() if k not in ['name', 'type', 'description']}
        ))
    
    # Extract relations from edges
    for i, (source, target, attrs) in enumerate(G.edges(data=True)):
        relations.append(Relation(
            id=f"relation_{i+1}",
            source_entity_id=source,
            target_entity_id=target,
            relation_type=attrs.get('relation_type', 'RELATED_TO'),
            description=attrs.get('description'),
            confidence=attrs.get('confidence'),
            metadata={k: v for k, v in attrs.items() if k not in ['relation_type', 'description', 'confidence']}
        ))
    
    graph = DocumentGraph(
        entities=entities,
        relations=relations,
        document_id=Path(graphml_path).stem
    )
    
    visualizer = GraphVisualizer()
    visualizer.visualize(graph, file_path=output_image, show=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_graph.py <graph_file> [output_image]")
        print("\nExamples:")
        print("  python visualize_graph.py outputs/Legal _ Uber_graph.json")
        print("  python visualize_graph.py outputs/Legal _ Uber_graph.graphml")
        print("  python visualize_graph.py outputs/Legal _ Uber_graph.json graph.png")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        if file_path.endswith('.json'):
            visualize_from_json(file_path, output_image)
        elif file_path.endswith('.graphml'):
            visualize_from_graphml(file_path, output_image)
        else:
            print("Error: Unsupported file format. Use .json or .graphml files.")
            sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall matplotlib with: uv add matplotlib")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
