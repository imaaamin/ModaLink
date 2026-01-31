"""LangGraph workflow for document entity and relation extraction."""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from src.models.document_graph import DocumentGraph
from src.models.entity import Entity
from src.models.relation import Relation
from src.document_processor import DocumentProcessor
from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor
import os


class ExtractionState(TypedDict):
    """State for the extraction workflow."""
    document_path: str
    text: str
    entities: List[Entity]
    relations: List[Relation]
    graph: DocumentGraph
    error: str


class DocumentExtractionGraph:
    """LangGraph workflow for extracting entities and relations from documents."""
    
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.0
    ):
        """
        Initialize the extraction graph.
        
        Args:
            model_name: Name of the Groq LLM model to use (e.g., "openai/gpt-oss-120b", "llama-3.1-70b-versatile", "mixtral-8x7b-32768")
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
        """Load and process the document."""
        try:
            print(f"Loading document: {state['document_path']}")
            text = self.processor.extract_text(state["document_path"])
            state["text"] = text
            state["error"] = ""
            print(f"Document loaded. Text length: {len(text)} characters")
            if not text or not text.strip():
                state["error"] = "Document text extraction returned empty result"
        except Exception as e:
            state["error"] = f"Error loading document: {str(e)}"
            import traceback
            traceback.print_exc()
        return state
    
    def _extract_entities(self, state: ExtractionState) -> ExtractionState:
        """Extract entities from the document text."""
        if state.get("error"):
            return state
        
        # Check if text is empty
        if not state.get("text") or not state["text"].strip():
            state["error"] = "Document text is empty. Cannot extract entities."
            state["entities"] = []
            return state
        
        try:
            print(f"Extracting entities from text (length: {len(state['text'])} characters)...")
            entities = self.entity_extractor.extract_entities(state["text"])
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
        """Build the final document graph structure."""
        if state.get("error"):
            return state
        
        try:
            graph = DocumentGraph(
                entities=state.get("entities", []),
                relations=state.get("relations", []),
                document_id=state.get("document_path", ""),
                metadata={
                    "text_length": len(state.get("text", "")),
                    "num_entities": len(state.get("entities", [])),
                    "num_relations": len(state.get("relations", []))
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
