# Integration Tests

This directory contains integration tests for the document extraction and Neo4j export functionality.

## Test Structure

- `conftest.py`: Pytest fixtures for Neo4j configuration and test setup
- `test_extraction_integration.py`: Main integration tests
- `helpers.py`: Helper functions for comparing JSON and Neo4j data
- `fixtures/`: Test fixtures (PDF documents)

## Running Tests

### Prerequisites

1. Ensure Neo4j is running locally
2. Set up environment variables in `.env`:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   NEO4J_DATABASE=neo4j
   GROQ_API_KEY=your_groq_api_key
   ```

### Run All Tests

```bash
uv run pytest tests/
```

### Run Specific Test

```bash
uv run pytest tests/test_extraction_integration.py::TestExtractionIntegration::test_extract_entities_and_relations_from_pdf
```

### Run with Coverage

```bash
uv run pytest tests/ --cov=src --cov-report=html
```

## Test Fixtures

- `fixtures/LegalUber.pdf`: Test PDF document for extraction tests (Uber Terms and Conditions)

## Test Coverage

The integration tests verify:

1. **Entity and Relation Extraction**: Entities and relations are correctly extracted from PDFs
2. **JSON Export**: Extracted data is correctly exported to JSON format
3. **Neo4j Export**: All entities and relations from JSON are correctly exported to Neo4j
4. **Property Export**: Additional entity and relation properties are correctly exported to Neo4j
5. **Data Consistency**: JSON and Neo4j data match

## Notes

- Tests use a clean Neo4j database (cleared before and after each test)
- Tests require a valid Groq API key for LLM calls
- Tests may take several minutes to run due to LLM API calls
