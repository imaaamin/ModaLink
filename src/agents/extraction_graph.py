"""LangGraph workflow for document entity and relation extraction."""

from pathlib import Path
from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END
from src.models.document_graph import DocumentGraph
from src.models.document import Document
from src.models.chunk import Chunk
from src.models.entity import Entity
from src.models.relation import Relation
from src.document_processor import DocumentProcessor
from src.chunking import chunk_text
from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor
import os


class ExtractionState(TypedDict, total=False):
    """State for the extraction workflow."""
    document_path: str
    text: str
    document: Optional[Document]
    chunks: List[Chunk]
    entities: List[Entity]
    relations: List[Relation]
    graph: DocumentGraph
    error: str


class DocumentExtractionGraph:
    """LangGraph workflow for extracting entities and relations from documents."""
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.0
    ):
        """
        Initialize the extraction graph.

        Args:
            model_name: Model name override. If None, auto-selects based on available API key.
            temperature: Temperature for the LLM
        """
        self.processor = DocumentProcessor(use_docling=True)  # Use docling for better text extraction
        self.entity_extractor = EntityExtractor(model_name=model_name, temperature=temperature)
        self.relation_extractor = RelationExtractor(model_name=model_name, temperature=temperature)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ExtractionState)
        
        # Add nodes
        workflow.add_node("load_document", self._load_document)
        workflow.add_node("extract_entities", self._extract_entities)
        workflow.add_node("extract_relations", self._extract_relations)
        workflow.add_node("build_graph", self._build_graph_structure)
        
        # Define the flow
        workflow.set_entry_point("load_document")
        workflow.add_edge("load_document", "extract_entities")
        workflow.add_edge("extract_entities", "extract_relations")
        workflow.add_edge("extract_relations", "build_graph")
        workflow.add_edge("build_graph", END)
        
        return workflow.compile()
    
    def _load_document(self, state: ExtractionState) -> ExtractionState:
        """Load and process the document; build Document node and chunks."""
        try:
            doc_path = state["document_path"]
            print(f"Loading document: {doc_path}")
            text = self.processor.extract_text(doc_path)
            state["text"] = text
            state["error"] = ""
            state["document"] = None
            state["chunks"] = []
            print(f"Document loaded. Text length: {len(text)} characters")
            if not text or not text.strip():
                state["error"] = "Document text extraction returned empty result"
                return state
            path = Path(doc_path)
            doc_id = path.stem or doc_path
            title = path.name
            state["document"] = Document(
                doc_id=doc_id,
                title=title,
                source=doc_path,
                published_date=None,
            )
            state["chunks"] = chunk_text(text, document_id=doc_id, chunk_size=1000, overlap=200)
            print(f"Chunked into {len(state['chunks'])} chunks")
        except Exception as e:
            state["error"] = f"Error loading document: {str(e)}"
            import traceback
            traceback.print_exc()
        return state
    
    def _extract_entities(self, state: ExtractionState) -> ExtractionState:
        """Extract entities in one LLM call: send full text with chunk separators so the LLM can fill source_chunk_ids for each entity."""
        if state.get("error"):
            return state
        
        chunks: List[Chunk] = state.get("chunks") or []
        if not chunks:
            state["error"] = "No chunks available. Cannot extract entities."
            state["entities"] = []
            return state
        
        try:
            # Build single text with clear chunk separators so the LLM can assign source_chunk_ids
            separator = "\n\n--- CHUNK {id} ---\n\n"
            combined_text = "".join(separator.format(id=ch.id) + ch.text for ch in chunks)
            chunk_ids = [ch.id for ch in chunks]
            print(f"Extracting entities from document (1 call, {len(chunks)} chunks, {len(combined_text)} chars)...")
            entities = self.entity_extractor.extract_entities(combined_text, chunk_ids=chunk_ids)
            state["entities"] = entities
            print(f"Extracted {len(entities)} entities")
        except Exception as e:
            state["error"] = f"Error extracting entities: {str(e)}"
            state["entities"] = []
            import traceback
            traceback.print_exc()
        
        return state
    
    def _extract_relations(self, state: ExtractionState) -> ExtractionState:
        """Extract relations between entities."""
        if state.get("error") or not state.get("entities"):
            return state
        
        try:
            relations = self.relation_extractor.extract_relations(
                state["text"],
                state["entities"]
            )
            state["relations"] = relations
        except Exception as e:
            state["error"] = f"Error extracting relations: {str(e)}"
            state["relations"] = []
        
        return state
    
    def _build_graph_structure(self, state: ExtractionState) -> ExtractionState:
        """Build the final document graph structure (document, chunks, entities, relations). source_chunk_ids set by LLM from chunk markers in text."""
        if state.get("error"):
            return state
        
        try:
            graph = DocumentGraph(
                entities=state.get("entities", []),
                relations=state.get("relations", []),
                document_id=state["document"].doc_id if state.get("document") else state.get("document_path", ""),
                document=state.get("document"),
                chunks=state.get("chunks") or [],
                metadata={
                    "text_length": len(state.get("text", "")),
                    "num_entities": len(state.get("entities", [])),
                    "num_relations": len(state.get("relations", [])),
                    "num_chunks": len(state.get("chunks") or []),
                }
            )
            state["graph"] = graph
        except Exception as e:
            state["error"] = f"Error building graph: {str(e)}"
        
        return state
    
    def extract(self, document_path: str) -> DocumentGraph:
        """
        Extract entities and relations from a document.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            DocumentGraph containing entities and relations
        """
        initial_state: ExtractionState = {
            "document_path": document_path,
            "text": "",
            "entities": [],
            "relations": [],
            "graph": None,
            "error": ""
        }
        
        result = self.graph.invoke(initial_state)
        
        if result.get("error"):
            raise Exception(result["error"])
        
        return result["graph"]
