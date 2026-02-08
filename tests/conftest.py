"""Pytest configuration and fixtures for integration tests."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
from src.neo4j_exporter import Neo4jExporter

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def test_pdf_path():
    """Path to the test PDF fixture."""
    pdf_path = Path(__file__).parent / "fixtures" / "LegalUber.pdf"
    if not pdf_path.exists():
        # Try alternative locations for CI
        alt_paths = [
            Path(__file__).parent.parent / "LegalUber.pdf",
            Path("LegalUber.pdf"),
            Path("tests/fixtures/LegalUber.pdf"),
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                return str(alt_path)
        pytest.skip(f"Test PDF not found at {pdf_path} or alternative locations")
    return str(pdf_path)


@pytest.fixture(scope="session")
def neo4j_config():
    """Neo4j configuration from environment variables."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    if not password:
        pytest.skip("NEO4J_PASSWORD not set. Skipping Neo4j tests.")
    
    return {
        "uri": uri,
        "user": user,
        "password": password,
        "database": database
    }


@pytest.fixture(scope="function")
def clean_neo4j_db(neo4j_config):
    """Fixture that clears Neo4j database before and after each test."""
    # Clear before test
    with Neo4jExporter(
        uri=neo4j_config["uri"],
        user=neo4j_config["user"],
        password=neo4j_config["password"],
        database=neo4j_config["database"]
    ) as exporter:
        with exporter.driver.session(database=neo4j_config["database"]) as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    yield neo4j_config
    
    # Clear after test
    with Neo4jExporter(
        uri=neo4j_config["uri"],
        user=neo4j_config["user"],
        password=neo4j_config["password"],
        database=neo4j_config["database"]
    ) as exporter:
        with exporter.driver.session(database=neo4j_config["database"]) as session:
            session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture(scope="function")
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="test_extraction_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
