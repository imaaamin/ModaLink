"""Graph visualization utilities for entity-relation graphs."""

import json
import sys
from pathlib import Path
from typing import Any, List, Optional

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from src.models.document_graph import DocumentGraph
from src.models.entity import Entity
from src.models.relation import Relation


def _graphml_safe_value(value: Any) -> Any:
    """Convert a value to a type supported by GraphML (string, int, float, bool)."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class GraphVisualizer:
    """Visualizes entity-relation graphs using NetworkX."""
    
    def __init__(self):
        """Initialize the graph visualizer."""
        pass
    
    def to_networkx(self, document_graph: DocumentGraph) -> nx.DiGraph:
        """
        Convert a DocumentGraph to a NetworkX directed graph.
        
        Args:
            document_graph: The document graph to convert
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes (entities); use only GraphML-safe types (no list/dict)
        for entity in document_graph.entities:
            node_attrs = {
                "name": _graphml_safe_value(entity.name),
                "type": _graphml_safe_value(entity.type),
            }
            if entity.description is not None:
                node_attrs["description"] = _graphml_safe_value(entity.description)
            for key, value in entity.metadata.items():
                if value is not None:
                    node_attrs[key] = _graphml_safe_value(value)
            G.add_node(entity.id, **node_attrs)
        
        # Add edges (relations); use only GraphML-safe types (no list/dict)
        for relation in document_graph.relations:
            all_props = relation.get_all_properties()
            edge_attrs = {}
            for key, value in all_props.items():
                if value is None or key in ("source_entity_id", "target_entity_id"):
                    continue
                edge_attrs[key] = _graphml_safe_value(value)
            if "relation_type" not in edge_attrs:
                edge_attrs["relation_type"] = relation.relation_type

            G.add_edge(
                relation.source_entity_id,
                relation.target_entity_id,
                **edge_attrs
            )
        
        return G
    
    def get_statistics(self, document_graph: DocumentGraph) -> dict:
        """
        Get statistics about the document graph.
        
        Args:
            document_graph: The document graph to analyze
            
        Returns:
            Dictionary with graph statistics
        """
        G = self.to_networkx(document_graph)
        
        # Entity type distribution
        entity_types = {}
        for entity in document_graph.entities:
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        # Relation type distribution
        relation_types = {}
        for relation in document_graph.relations:
            relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1
        
        return {
            "total_entities": len(document_graph.entities),
            "total_relations": len(document_graph.relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "graph_density": nx.density(G),
            "is_connected": nx.is_weakly_connected(G) if len(G.nodes) > 0 else False,
            "num_connected_components": nx.number_weakly_connected_components(G) if len(G.nodes) > 0 else 0,
        }
    
    def export_to_json(self, document_graph: DocumentGraph, file_path: str):
        """
        Export the document graph to a JSON file.
        
        Args:
            document_graph: The document graph to export
            file_path: Path to save the JSON file
        """
        import json
        
        data = {
            "document_id": document_graph.document_id,
            "entities": [entity.model_dump() for entity in document_graph.entities],
            "relations": [relation.model_dump() for relation in document_graph.relations],
            "metadata": document_graph.metadata
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def export_to_graphml(self, document_graph: DocumentGraph, file_path: str, include_layout: bool = True):
        """
        Export the document graph to GraphML format.
        
        Args:
            document_graph: The document graph to export
            file_path: Path to save the GraphML file
            include_layout: Whether to include node positions for better visualization (default: True)
        """
        G = self.to_networkx(document_graph)
        
        # If layout is requested, compute positions and add them as node attributes
        # This helps visualization tools like yEd display the graph properly
        if include_layout and len(G.nodes) > 0:
            try:
                # Use spring layout for better node distribution
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                
                # Add position attributes that yEd and other tools can use
                for node_id, (x, y) in pos.items():
                    # Normalize coordinates to a reasonable range (0-1000)
                    G.nodes[node_id]['x'] = float(x * 500 + 500)  # Scale and center
                    G.nodes[node_id]['y'] = float(y * 500 + 500)
            except Exception as e:
                print(f"Warning: Could not compute layout: {e}")
        
        nx.write_graphml(G, file_path)
    
    def visualize(self, document_graph: DocumentGraph, file_path: Optional[str] = None, show: bool = True):
        """
        Visualize the document graph using matplotlib.
        
        Args:
            document_graph: The document graph to visualize
            file_path: Optional path to save the visualization image
            show: Whether to display the plot (default: True)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. Install it with: uv add matplotlib"
            )
        
        G = self.to_networkx(document_graph)
        
        if len(G.nodes) == 0:
            print("No nodes to visualize.")
            return
        
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Color nodes by entity type
        entity_type_colors = {}
        colors = plt.cm.Set3(range(len(set(n[1].get('type', 'UNKNOWN') for n in G.nodes(data=True)))))
        for i, entity_type in enumerate(set(n[1].get('type', 'UNKNOWN') for n in G.nodes(data=True))):
            entity_type_colors[entity_type] = colors[i]
        
        node_colors = [entity_type_colors.get(n[1].get('type', 'UNKNOWN'), 'gray') for n in G.nodes(data=True)]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.5,
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            width=1.5
        )
        
        # Draw edge labels (relation types)
        edge_labels = {}
        for source, target, attrs in G.edges(data=True):
            relation_type = attrs.get('relation_type', 'RELATED_TO')
            # Truncate long relation type names for better readability
            label = relation_type.replace('_', ' ')[:15]  # Replace underscores and truncate
            edge_labels[(source, target)] = label
        
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color='darkblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )
        
        # Draw labels (entity names)
        labels = {n[0]: n[1].get('name', n[0])[:20] for n in G.nodes(data=True)}  # Truncate long names
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Create legend
        legend_elements = [
            mpatches.Patch(facecolor=entity_type_colors.get(etype, 'gray'), label=etype)
            for etype in sorted(set(n[1].get('type', 'UNKNOWN') for n in G.nodes(data=True)))
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.title(f"Entity-Relation Graph\n{len(G.nodes)} entities, {len(G.edges)} relations", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if file_path:
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {file_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
