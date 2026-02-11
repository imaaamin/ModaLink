# Document Extractor

A multi-modal document data extractor using LangGraph for extracting entities and their relations from documents. The system uses advanced document processing with docling and dynamic entity/relation extraction to build knowledge graphs from documents.

## Features

- **Dynamic Entity Extraction**: Two-step process that first extracts all entities, then dynamically categorizes them into types based on document content (no hardcoded entity types)
- **Dynamic Relation Extraction**: Extracts relationships between entities with dynamically identified relation types based on document context
- **Graph Structure**: Builds a graph representation of entities and their relations using NetworkX
- **High-Quality Text Extraction**: Uses docling for superior text extraction from PDFs and Word documents
- **Multi-format Support**: Supports PDF, Word documents, images (with OCR), and plain text
- **LangGraph Workflow**: Uses LangGraph for orchestrated, agentic extraction pipeline
- **Neo4j Integration**: Direct export to Neo4j graph database with proper node labels and relationships

## Installation

This project uses `uv` for Python package management.

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API key(s)
```

The system supports two LLM providers with automatic fallback:
- **Google Gemini** (primary): Set `GOOGLE_API_KEY` — get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Groq** (fallback): Set `GROQ_API_KEY` — get one at [console.groq.com](https://console.groq.com)

At least one API key is required. If both are set, Gemini is used.

## Usage

### Basic Usage

```bash
uv run python main.py <document_path> [--merge]
```

Examples:
```bash
# Extract and import to Neo4j (clears existing graph, then adds new extraction)
uv run python main.py documents/sample.pdf

# Extract and merge with existing Neo4j data (do not clear first)
uv run python main.py documents/sample.pdf --merge
```

**Options:**
- `--merge`: Merge with existing Neo4j data instead of clearing the graph first (default: clear then import)

### Programmatic Usage

```python
from dotenv import load_dotenv
from src.agents.extraction_graph import DocumentExtractionGraph
import sys
from pathlib import Path

# Add scripts to path for graph_visualizer
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from graph_visualizer import GraphVisualizer

load_dotenv()

# Initialize extractor (auto-selects provider based on available API key)
extractor = DocumentExtractionGraph()

# Extract entities and relations
graph = extractor.extract("path/to/document.pdf")

# Visualize and export
visualizer = GraphVisualizer()
stats = visualizer.get_statistics(graph)
visualizer.export_to_json(graph, "output.json")
visualizer.export_to_graphml(graph, "output.graphml")
```

### Working with relation attributes

Invoking the graph is the same: `graph = extractor.extract(path)` returns a `DocumentGraph` with `entities` and `relations`. Dates and times are stored as **attributes on relations** (e.g. `start_date`, `end_date`, `occurred_on`), not as DATE/TIME entities.

- **Access all properties on a relation** (including temporal and other attributes):
  ```python
  for rel in graph.relations:
      props = rel.get_all_properties()  # id, relation_type, description, start_date, end_date, role, ...
      if "start_date" in props:
          print(rel.relation_type, props["start_date"])
  ```
- **Query relations that have a given attribute**:
  ```python
  dated_relations = graph.get_relations_with_attribute("start_date")
  ```
- **JSON / Neo4j**: Relation attributes are included in `export_to_json` (via `model_dump()`) and exported to Neo4j as relationship properties. In Cypher you can filter by them, e.g. `MATCH (a)-[r:WORKS_FOR]->(b) WHERE r.start_date IS NOT NULL RETURN a, r, b`.

## Project Structure

```
DocumentExtractor/
├── src/
│   ├── __init__.py
│   ├── agents/                    # Agent modules for extraction
│   │   ├── __init__.py
│   │   ├── entity_extractor.py   # Dynamic entity extraction using LLM (extract + categorize)
│   │   ├── relation_extractor.py # Dynamic relation extraction using LLM
│   │   └── extraction_graph.py   # LangGraph workflow orchestration
│   ├── models/                    # Data models
│   │   ├── __init__.py
│   │   ├── entity.py             # Entity model
│   │   ├── relation.py           # Relation model
│   │   └── document_graph.py     # DocumentGraph model
│   ├── model_provider.py         # LLM provider factory (Gemini primary, Groq fallback)
│   ├── document_processor.py     # Document text extraction using docling
│   ├── embeddings.py             # Entity/text embeddings (OpenAI or sentence-transformers)
│   ├── graph_retriever.py        # Vector-based retrieval over Neo4j for LLM context
│   └── neo4j_exporter.py         # Neo4j database export with Cypher queries
├── scripts/
│   ├── __init__.py
│   ├── graph_visualizer.py       # Graph visualization and export utilities (JSON, GraphML)
│   ├── visualize_graph.py        # Graph visualization script with edge labels
│   ├── export_to_neo4j.py        # Standalone Neo4j export script
│   ├── query_graph.py            # Vector search: query string → top 10 nodes + Cypher
│   └── fetch_pr_comments.py      # Utility to fetch GitHub PR comments
│   └── export_to_neo4j.py        # Standalone Neo4j export script
├── main.py                        # CLI entry point (clears Neo4j by default; use --merge to keep existing)
├── pyproject.toml                 # Project dependencies (uv)
└── README.md
```

## Output

The extractor generates:
- **JSON file**: Complete graph data in JSON format
- **GraphML file**: Graph structure in GraphML format (can be opened in tools like yEd, Gephi)
- **Neo4j database** (optional): Exports directly to Neo4j graph database

### Neo4j Export

The extractor can automatically export graphs to a Neo4j database. Configure your Neo4j connection in `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

The extractor will automatically export to Neo4j if these environment variables are set.

You can also export an existing JSON file to Neo4j:

```bash
uv run python scripts/export_to_neo4j.py outputs/your_file_graph.json
uv run python scripts/export_to_neo4j.py outputs/your_file_graph.json --clear  # Clear existing data first
uv run python scripts/export_to_neo4j.py outputs/your_file_graph.json --clear  # Embeddings are included by default
```

**Neo4j Graph Structure:**
- **Nodes**: Each entity becomes a node labeled with its entity type only (e.g., `:PERSON`, `:ORGANIZATION`, `:LOCATION`)
- **Relationships**: Each relation becomes a directed relationship with the relation type as the relationship type (e.g., `:WORKS_FOR`, `:LOCATED_IN`)
- **Properties**: Entity names, types, descriptions, and metadata are stored as node properties
- **Node Identification**: Nodes are identified by their `id` property for merging duplicates

**Note:** Neo4j Explore view only shows connected nodes by default. To see all nodes (including isolated ones), run:
```cypher
MATCH (n) RETURN n LIMIT 500
```

### Vector embeddings and semantic search (Neo4j 5.13+)

**Vector embeddings are enabled by default** when exporting to Neo4j. Each node gets an `embedding` property and the `entity_embedding` vector index is created for similarity search.

- **CLI:** `uv run python main.py document.pdf` or `uv run python scripts/export_to_neo4j.py graph.json --clear` (use `--no-embed` to skip embeddings)
- **Programmatic:** pass `embedder=emb.embed_entity` and `embedding_dimension=emb.get_embedding_dimension()` to `exporter.export_graph(...)` (or omit both to skip).

Embeddings use **OpenAI** when `OPENAI_API_KEY` is set (model `text-embedding-3-small`, 1536 dimensions), otherwise **sentence-transformers** (model `all-MiniLM-L6-v2`, 384 dimensions, no API key). Each node gets an `embedding` property and an extra label `Entity`; a vector index `entity_embedding` is created so you can run similarity search.

**Graph retriever (context for LLM):** use `GraphRetriever` to turn a user query into relevant graph context:

```python
from src.graph_retriever import GraphRetriever

with GraphRetriever() as retriever:
    context = retriever.retrieve("What are the payment terms?", top_k=10, expand_hop=1)
# Use context as part of your LLM prompt
print(context)
```

This embeds the query, runs `db.index.vector.queryNodes('entity_embedding', k, query_vector)` in Neo4j, then loads 1-hop relationships for the top-k nodes and returns a single text (entities + relationships) suitable as context for an LLM.

## Viewing GraphML Files

### Option 1: Python Visualization (Recommended)

Install matplotlib and visualize directly:
```bash
uv add matplotlib
uv run python scripts/visualize_graph.py outputs/your_file_graph.json
# or
uv run python scripts/visualize_graph.py outputs/your_file_graph.graphml
```

You can also save the visualization as an image:
```bash
uv run python scripts/visualize_graph.py outputs/your_file_graph.json graph.png
```

### Utility Scripts

**Export to Neo4j:**
```bash
uv run python scripts/export_to_neo4j.py outputs/your_file_graph.json [--clear] [--no-embed]
```

**Query graph (vector search, top 10 related nodes):**
```bash
uv run python scripts/query_graph.py "your natural language query"
# Example: uv run python scripts/query_graph.py "payment terms"
# Prints the Cypher used and formatted context for an LLM.
```

**Fetch GitHub PR Comments:**
```bash
uv run python scripts/fetch_pr_comments.py <owner> <repo> <pr_number> [output_file]
```

Example:
```bash
uv run python scripts/fetch_pr_comments.py microsoft vscode 12345
uv run python scripts/fetch_pr_comments.py microsoft vscode 12345 comments.json
```

### Option 2: External Tools

**yEd Graph Editor** (Free, Recommended):
- Download: https://www.yworks.com/products/yed
- Open: File → Open → Select your `.graphml` file
- Best for interactive exploration and editing

**Gephi** (Free, Open Source):
- Download: https://gephi.org/
- Open: File → Open → Select your `.graphml` file
- Best for network analysis and statistics

**Cytoscape** (Free, Open Source):
- Download: https://cytoscape.org/
- Open: File → Import → Network → File → Select your `.graphml` file
- Best for biological/network visualization

## Supported Document Formats

- **PDF** (`.pdf`) - Uses docling for high-quality text extraction
- **Word Documents** (`.docx`, `.doc`) - Uses docling for structured extraction
- **Images** (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`) - OCR with Tesseract
- **Plain Text** (`.txt`)

## Dynamic Entity and Relation Extraction

The system uses a **dynamic extraction approach** that adapts to document content:

### Entity Extraction Process

1. **Step 1 - Extract**: Identifies all meaningful entities from the document without predefined types
2. **Step 2 - Categorize**: Dynamically assigns entity types based on document context

**Common entity types** (dynamically identified):
- PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PRODUCT, EVENT, TECHNOLOGY
- Plus document-specific types like: STANDARD, REGULATION, FRAMEWORK, DOMAIN, ROLE, etc.

### Relation Extraction Process

- Extracts relationships between entities with **dynamically identified relation types**
- Identifies contextual, temporal, and reference relationships
- No predefined relation types - adapts to document content

**Common relation types** (dynamically identified):
- Direct relationships: WORKS_FOR, LOCATED_IN, OWNS, PART_OF, FOUNDED, MANAGES
- Contextual relationships: REFERENCED_IN, DEFINED_IN, MENTIONED_IN
- Temporal relationships: OCCURRED_ON, HAS_REVISION_DATE
- And many more based on document context

## Requirements

- Python 3.10+
- An LLM API key: `GOOGLE_API_KEY` (Gemini) or `GROQ_API_KEY` (Groq)
- `uv` package manager

### LLM Provider Selection

The system uses `src/model_provider.py` to auto-select the LLM provider based on available API keys:

| Provider | Env Variable | Default Model |
|----------|-------------|---------------|
| Google Gemini (primary) | `GOOGLE_API_KEY` | `gemini-2.0-flash` |
| Groq (fallback) | `GROQ_API_KEY` | `openai/gpt-oss-120b` |

You can override the model when initializing the extractor:
```python
extractor = DocumentExtractionGraph(model_name="gemini-2.5-pro-preview-05-06")
```

### Document Processing with Docling

The system uses **docling** for high-quality text extraction from PDFs and Word documents. Docling provides:
- Better handling of complex document layouts
- Improved text extraction quality
- Structured document parsing

Docling is automatically used when available. If docling extraction fails, the system falls back to pypdf/python-docx.

### Optional: OCR Support

For image OCR support, you need to install Tesseract OCR on your system:

- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

The Python wrapper (`pytesseract`) is included in the dependencies and will be installed automatically.

## How It Works

### Extraction Pipeline

1. **Document Loading**: Extracts text using docling (or fallback methods)
2. **Entity Extraction**: 
   - First extracts all entities from the document
   - Then categorizes them into appropriate types dynamically
3. **Relation Extraction**: 
   - Identifies relationships between extracted entities
   - Dynamically determines relation types based on context
4. **Graph Building**: Constructs a DocumentGraph with entities and relations
5. **Export**: Exports to JSON, GraphML, and optionally Neo4j

### Key Design Decisions

- **No Hardcoded Types**: Entity and relation types are determined dynamically based on document content
- **Two-Step Entity Extraction**: Separates entity identification from categorization for better accuracy
- **Context-Aware Relations**: Extracts contextual, temporal, and reference relationships, not just direct ones
- **Neo4j Integration**: Direct export with proper labels and relationships for graph database analysis

### Code Organization

The codebase is organized into logical modules:

- **`src/agents/`**: Contains the agent modules responsible for extraction:
  - `entity_extractor.py`: Handles entity extraction and categorization
  - `relation_extractor.py`: Handles relation extraction between entities
  - `extraction_graph.py`: Orchestrates the LangGraph workflow

- **`src/models/`**: Contains the data models:
  - `entity.py`: Entity model with id, name, type, description, metadata
  - `relation.py`: Relation model with source, target, type, description, confidence
  - `document_graph.py`: DocumentGraph model containing entities and relations

- **`src/`**: Core utilities:
  - `model_provider.py`: LLM provider factory (Gemini primary, Groq fallback)
  - `document_processor.py`: Document text extraction (docling, pypdf, python-docx, OCR)
  - `neo4j_exporter.py`: Neo4j database export functionality

- **`scripts/`**: Utility scripts:
  - `graph_visualizer.py`: Graph visualization and export utilities (JSON, GraphML)
  - `visualize_graph.py`: Graph visualization script with edge labels
  - `export_to_neo4j.py`: Standalone script to export JSON graphs to Neo4j

## Testing

Integration tests are available in the `tests/` directory. These tests verify:

- Entity and relation extraction from documents
- JSON export functionality
- Neo4j export and data consistency
- Property export for entities and relations

### Running Tests Locally

```bash
# Install test dependencies
uv sync

# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### CI/CD

Tests run automatically on GitHub Actions for every push and pull request. The CI workflow:
- Sets up a Neo4j instance using Docker (Neo4j 5 Community Edition)
- Runs all integration tests
- Verifies entity and relation extraction
- Validates Neo4j export functionality

**GitHub Actions Setup:**
1. Add `GOOGLE_API_KEY` or `GROQ_API_KEY` as a repository secret in GitHub Settings → Secrets and variables → Actions
2. The workflow automatically sets up Neo4j in a Docker container
3. Tests run against the containerized Neo4j instance

See `tests/README.md` for more details on test setup and requirements.

## License

MIT
