"""Integration tests for document extraction and Neo4j export."""

import json
import pytest
import sys
from pathlib import Path

# Add scripts to path for graph_visualizer
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from graph_visualizer import GraphVisualizer

from src.agents.extraction_graph import DocumentExtractionGraph
from src.neo4j_exporter import Neo4jExporter
from tests.helpers import check_all_entities, check_all_relations


class TestExtractionIntegration:
    """Integration tests for the full extraction pipeline."""
    
    def test_extract_entities_and_relations_from_pdf(self, test_pdf_path, temp_output_dir):
        """Test that entities and relations can be extracted from a PDF."""
        # Initialize extractor
        extractor = DocumentExtractionGraph(
            model_name="openai/gpt-oss-120b",
            temperature=0.0
        )
        
        # Extract entities and relations
        graph = extractor.extract(test_pdf_path)
        
        # Verify extraction results
        assert len(graph.entities) > 0, "Should extract at least one entity"
        assert len(graph.relations) > 0, "Should extract at least one relation"
        
        # Verify entity structure
        for entity in graph.entities:
            assert hasattr(entity, 'id'), "Entity should have id"
            assert hasattr(entity, 'name'), "Entity should have name"
            assert hasattr(entity, 'type'), "Entity should have type"
            assert entity.id, "Entity id should not be empty"
            assert entity.name, "Entity name should not be empty"
            assert entity.type, "Entity type should not be empty"
        
        # Verify relation structure
        for relation in graph.relations:
            assert hasattr(relation, 'id'), "Relation should have id"
            assert hasattr(relation, 'source_entity_id'), "Relation should have source_entity_id"
            assert hasattr(relation, 'target_entity_id'), "Relation should have target_entity_id"
            assert hasattr(relation, 'relation_type'), "Relation should have relation_type"
            assert relation.id, "Relation id should not be empty"
            assert relation.source_entity_id, "Relation source_entity_id should not be empty"
            assert relation.target_entity_id, "Relation target_entity_id should not be empty"
            assert relation.relation_type, "Relation type should not be empty"
            
            # Verify source and target entities exist
            source = graph.get_entity_by_id(relation.source_entity_id)
            target = graph.get_entity_by_id(relation.target_entity_id)
            assert source is not None, f"Source entity {relation.source_entity_id} should exist"
            assert target is not None, f"Target entity {relation.target_entity_id} should exist"
        
        # Export to JSON
        json_path = Path(temp_output_dir) / "test_graph.json"
        visualizer = GraphVisualizer()
        visualizer.export_to_json(graph, str(json_path))
        
        # Verify JSON file was created
        assert json_path.exists(), "JSON file should be created"
        
        # Verify JSON content
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert 'entities' in json_data, "JSON should contain entities"
        assert 'relations' in json_data, "JSON should contain relations"
        assert len(json_data['entities']) == len(graph.entities), "JSON entities count should match"
        assert len(json_data['relations']) == len(graph.relations), "JSON relations count should match"
    
    def test_neo4j_export_matches_json(self, test_pdf_path, clean_neo4j_db, temp_output_dir):
        """Test that Neo4j export matches the JSON output."""
        # Initialize extractor
        extractor = DocumentExtractionGraph(
            model_name="openai/gpt-oss-120b",
            temperature=0.0
        )
        
        # Extract entities and relations
        graph = extractor.extract(test_pdf_path)
        
        # Export to JSON
        json_path = Path(temp_output_dir) / "test_graph.json"
        visualizer = GraphVisualizer()
        visualizer.export_to_json(graph, str(json_path))
        
        # Export to Neo4j
        with Neo4jExporter(
            uri=clean_neo4j_db["uri"],
            user=clean_neo4j_db["user"],
            password=clean_neo4j_db["password"],
            database=clean_neo4j_db["database"]
        ) as exporter:
            stats = exporter.export_graph(graph, clear_existing=True, merge_duplicates=True)
            
            # Verify export statistics
            assert stats['entities_created'] > 0, "Should create entities in Neo4j"
            assert stats['relations_created'] > 0, "Should create relations in Neo4j"
            assert len(stats['errors']) == 0, f"Should have no errors, but got: {stats['errors']}"
        
        # Check that all entities from JSON are in Neo4j
        missing_entities, extra_entities = check_all_entities(
            json_path=str(json_path),
            neo4j_config=clean_neo4j_db
        )
        
        assert len(missing_entities) == 0, f"All entities should be in Neo4j. Missing: {missing_entities}"
        
        # Check that all relations from JSON are in Neo4j
        missing_relations, extra_relations = check_all_relations(
            json_path=str(json_path),
            neo4j_config=clean_neo4j_db
        )
        
        assert len(missing_relations) == 0, f"All relations should be in Neo4j. Missing: {missing_relations}"
    
    def test_entity_properties_exported_to_neo4j(self, test_pdf_path, clean_neo4j_db):
        """Test that entity properties are correctly exported to Neo4j."""
        # Initialize extractor
        extractor = DocumentExtractionGraph(
            model_name="openai/gpt-oss-120b",
            temperature=0.0
        )
        
        # Extract entities and relations
        graph = extractor.extract(test_pdf_path)
        
        # Export to Neo4j
        with Neo4jExporter(
            uri=clean_neo4j_db["uri"],
            user=clean_neo4j_db["user"],
            password=clean_neo4j_db["password"],
            database=clean_neo4j_db["database"]
        ) as exporter:
            exporter.export_graph(graph, clear_existing=True, merge_duplicates=True)
            
            # Check a few entities have their properties
            with exporter.driver.session(database=clean_neo4j_db["database"]) as session:
                # Get an entity with additional properties
                query = """
                MATCH (n)
                WHERE n.id IS NOT NULL
                RETURN n.id as id, n.name as name, keys(n) as properties
                LIMIT 10
                """
                result = session.run(query)
                
                entities_checked = 0
                for record in result:
                    entity_id = record["id"]
                    properties = record["properties"]
                    
                    # Find the entity in the graph
                    entity = graph.get_entity_by_id(entity_id)
                    if entity:
                        # Check that core properties are present
                        assert "id" in properties, f"Entity {entity_id} should have 'id' property"
                        assert "name" in properties, f"Entity {entity_id} should have 'name' property"
                        assert "type" in properties, f"Entity {entity_id} should have 'type' property"
                        
                        # Check that additional properties from entity are in Neo4j
                        all_props = entity.get_all_properties()
                        core_fields = {"id", "name", "type", "description"}
                        additional_props = {k: v for k, v in all_props.items() 
                                           if k not in core_fields and v is not None}
                        
                        # Verify additional properties are exported (may be JSON strings for complex types)
                        for prop_key in additional_props.keys():
                            # Property name might be sanitized
                            sanitized_key = Neo4jExporter._sanitize_property_name(prop_key)
                            assert sanitized_key in properties or prop_key in properties, \
                                f"Entity {entity_id} should have property '{prop_key}' (or '{sanitized_key}')"
                        
                        entities_checked += 1
                        if entities_checked >= 5:
                            break
                
                assert entities_checked > 0, "Should check at least one entity"
    
    def test_relation_properties_exported_to_neo4j(self, test_pdf_path, clean_neo4j_db):
        """Test that relation properties are correctly exported to Neo4j."""
        # Initialize extractor
        extractor = DocumentExtractionGraph(
            model_name="openai/gpt-oss-120b",
            temperature=0.0
        )
        
        # Extract entities and relations
        graph = extractor.extract(test_pdf_path)
        
        # Export to Neo4j
        with Neo4jExporter(
            uri=clean_neo4j_db["uri"],
            user=clean_neo4j_db["user"],
            password=clean_neo4j_db["password"],
            database=clean_neo4j_db["database"]
        ) as exporter:
            exporter.export_graph(graph, clear_existing=True, merge_duplicates=True)
            
            # Check relations have their properties
            with exporter.driver.session(database=clean_neo4j_db["database"]) as session:
                query = """
                MATCH ()-[r]->()
                WHERE r.id IS NOT NULL
                RETURN type(r) as rel_type, keys(r) as properties, r.id as id
                LIMIT 10
                """
                result = session.run(query)
                
                relations_checked = 0
                for record in result:
                    rel_id = record.get("id")
                    properties = record["properties"]
                    
                    # Find the relation in the graph
                    relation = None
                    if rel_id:
                        relation = next((r for r in graph.relations if r.id == rel_id), None)
                    
                    if relation:
                        # Check that core properties are present
                        assert "description" in properties or relation.description is None, \
                            f"Relation {rel_id} should have 'description' property"
                        
                        # Check that additional properties from relation are in Neo4j
                        all_props = relation.get_all_properties()
                        core_fields = {"id", "source_entity_id", "target_entity_id", 
                                      "relation_type", "description", "confidence"}
                        additional_props = {k: v for k, v in all_props.items() 
                                           if k not in core_fields and v is not None}
                        
                        # Verify additional properties are exported
                        for prop_key in additional_props.keys():
                            sanitized_key = Neo4jExporter._sanitize_property_name(prop_key)
                            assert sanitized_key in properties or prop_key in properties, \
                                f"Relation {rel_id} should have property '{prop_key}' (or '{sanitized_key}')"
                        
                        relations_checked += 1
                        if relations_checked >= 5:
                            break
                
                assert relations_checked > 0, "Should check at least one relation"
