# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocumentExtractor is a multi-modal document data extraction system that uses LangGraph to extract entities and relations from documents, building knowledge graphs. It supports PDF, Word, images (OCR via Tesseract), and plain text. Extracted graphs can be exported to JSON, GraphML, and Neo4j.

## Commands

```bash
# Install dependencies
uv sync

# Run extraction on a document
uv run python main.py <document_path> [--merge]

# Run all tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run a single test
uv run pytest tests/test_extraction_integration.py::TestDocumentExtraction::test_name -v

# Skip slow/integration tests
uv run pytest tests/ -m "not slow"
uv run pytest tests/ -m "not integration"
```

## Environment Variables

Requires a `.env` file (see `.env.example`):
- `GOOGLE_API_KEY` - Google Gemini API key (primary). Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- `GROQ_API_KEY` - Groq API key (fallback, used if `GOOGLE_API_KEY` is not set)
- At least one of the above is required.
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` - Required for Neo4j export

## Architecture

### LangGraph Extraction Pipeline (`src/agents/extraction_graph.py`)

The core workflow is a 4-node LangGraph `StateGraph` with linear flow:

```
load_document → extract_entities → extract_relations → build_graph → END
```

State is passed as `ExtractionState` (TypedDict) through each node. Entry point: `DocumentExtractionGraph.extract(document_path)` returns a `DocumentGraph`.

### Two-Step Entity Extraction (`src/agents/entity_extractor.py`)

Entity extraction uses two sequential LLM calls:
1. **Extract** - LLM identifies all entities with properties (no predefined types)
2. **Categorize** - LLM assigns dynamic types while preserving properties

DATE/TIME entities are filtered out; they become relation attributes instead.

### Relation Extraction (`src/agents/relation_extractor.py`)

Runs after entity extraction. Identifies relationships between entities with dynamic relation types. Captures temporal/contextual attributes (start_date, end_date, role, amount) as relation properties. Validates entity IDs exist before creating relations.

### Data Models (`src/models/`)

All models use Pydantic with `model_config = ConfigDict(extra="allow")` for dynamic properties:
- **Entity** - `id`, `name`, `type`, `description`, `metadata` + arbitrary extra fields
- **Relation** - `id`, `source_entity_id`, `target_entity_id`, `relation_type`, `description`, `confidence`, `metadata` + extra fields
- **DocumentGraph** - Container of entities/relations with query methods (`get_entity_by_id`, `get_relations_for_entity`, `get_relations_with_attribute`)

### Document Processing (`src/document_processor.py`)

Multi-format with fallback chain: Docling (preferred) → format-specific library (pypdf, python-docx) → Tesseract OCR for images.

### Neo4j Export (`src/neo4j_exporter.py`)

Exports `DocumentGraph` to Neo4j. Entity types become node labels, relation types become relationship names. Supports merge-on-id for deduplication. Complex types (lists, dicts) are serialized to JSON strings for Neo4j compatibility.

### LLM Provider (`src/model_provider.py`)

`create_llm(model_name, temperature)` factory that auto-selects the provider based on available API keys: `GOOGLE_API_KEY` → Gemini (default model: `gemini-2.0-flash`), `GROQ_API_KEY` → Groq (default model: `openai/gpt-oss-120b`). Both extractors and the extraction graph use this factory. Pass `model_name` to override the default for the active provider.

## LLM Configuration

LLM responses are expected as JSON; the extractors handle markdown code block stripping and JSON parsing. Both extractors accept `model_name` and `temperature` params.

## Testing

Tests are integration tests that require an LLM API key (`GOOGLE_API_KEY` or `GROQ_API_KEY`) and a running Neo4j instance. Test fixture: `tests/fixtures/LegalUber.pdf`. CI runs Neo4j 5 Community Edition as a Docker service. Markers: `@pytest.mark.integration`, `@pytest.mark.slow`.
