"""Document processing module for extracting text from various document formats."""

import os
from pathlib import Path

# On Windows, disable Hugging Face cache symlinks to avoid WinError 1314
# ("A required privilege is not held by the client"). Using copies instead.
if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
from typing import Optional

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Fallback imports
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class DocumentProcessor:
    """Processes various document formats and extracts text content."""
    
    def __init__(self, use_docling: bool = True):
        """
        Initialize the document processor.
        
        Args:
            use_docling: Whether to use docling for better text extraction (default: True)
        """
        self.use_docling = use_docling and DOCLING_AVAILABLE
        if self.use_docling:
            try:
                self.converter = DocumentConverter()
                print("✓ Docling initialized successfully")
            except OSError as e:
                if getattr(e, "winerror", None) == 1314:
                    print(
                        "✗ Docling failed (Windows symlink privilege). "
                        "Falling back to pypdf. Enable Developer Mode to use docling."
                    )
                    self.use_docling = False
                else:
                    print(f"✗ Failed to initialize docling: {e}")
                    raise
            except Exception as e:
                print(f"✗ Failed to initialize docling: {e}")
                raise
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif extension in ['.docx', '.doc']:
            return self._extract_from_docx(file_path)
        elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            return self._extract_from_image(file_path)
        elif extension == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using docling."""
        if self.use_docling:
            print(f"Extracting text from PDF using docling: {file_path}")
            try:
                # Docling convert() returns a ConversionResult object
                result = self.converter.convert(file_path)
                
                # The result has a 'document' attribute which is a DoclingDocument
                if not hasattr(result, 'document'):
                    raise ValueError("Docling conversion result has no 'document' attribute")
                
                docling_doc = result.document
                
                # DoclingDocument has export_to_text() and export_to_markdown() methods
                if hasattr(docling_doc, 'export_to_text'):
                    text = docling_doc.export_to_text()
                    if text and text.strip():
                        print(f"✓ Docling extracted {len(text)} characters")
                        return text.strip()
                
                # Fallback to markdown if text export doesn't work
                if hasattr(docling_doc, 'export_to_markdown'):
                    text = docling_doc.export_to_markdown()
                    if text and text.strip():
                        print(f"✓ Docling extracted {len(text)} characters (markdown)")
                        return text.strip()
                
                raise ValueError("Docling extraction returned empty text")
                
            except OSError as e:
                if getattr(e, "winerror", None) == 1314:
                    print(
                        "✗ Docling failed (Windows symlink privilege). "
                        "Falling back to pypdf for this document."
                    )
                    self.use_docling = False
                else:
                    print(f"✗ Docling extraction failed: {e}")
                    raise
            except Exception as e:
                print(f"✗ Docling extraction failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Fallback to pypdf if docling is disabled
        if not PYPDF_AVAILABLE:
            raise ValueError("pypdf is required for PDF extraction. Install with: uv add pypdf")
        
        print(f"Extracting text from PDF using pypdf: {file_path}")
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from Word document using docling."""
        if self.use_docling:
            print(f"Extracting text from Word document using docling: {file_path}")
            try:
                # Docling convert() returns a ConversionResult object
                result = self.converter.convert(file_path)
                
                # The result has a 'document' attribute which is a DoclingDocument
                if not hasattr(result, 'document'):
                    raise ValueError("Docling conversion result has no 'document' attribute")
                
                docling_doc = result.document
                
                # DoclingDocument has export_to_text() and export_to_markdown() methods
                if hasattr(docling_doc, 'export_to_text'):
                    text = docling_doc.export_to_text()
                    if text and text.strip():
                        print(f"✓ Docling extracted {len(text)} characters")
                        return text.strip()
                
                # Fallback to markdown if text export doesn't work
                if hasattr(docling_doc, 'export_to_markdown'):
                    text = docling_doc.export_to_markdown()
                    if text and text.strip():
                        print(f"✓ Docling extracted {len(text)} characters (markdown)")
                        return text.strip()
                
                raise ValueError("Docling extraction returned empty text")
                
            except Exception as e:
                print(f"✗ Docling extraction failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Fallback to python-docx if docling is disabled
        if not DOCX_AVAILABLE:
            raise ValueError("python-docx is required for Word document extraction. Install with: uv add python-docx")
        
        print(f"Extracting text from Word document using python-docx: {file_path}")
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def _extract_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        if not TESSERACT_AVAILABLE:
            raise ValueError(
                "OCR support requires pytesseract. Install it with: uv add pytesseract\n"
                "Also ensure Tesseract OCR is installed on your system:\n"
                "  - macOS: brew install tesseract\n"
                "  - Ubuntu: sudo apt-get install tesseract-ocr\n"
                "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from image: {e}")
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
