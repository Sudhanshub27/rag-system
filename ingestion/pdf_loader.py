"""
PDF Document Loader
Uses PyMuPDF (fitz) for fast, layout-aware PDF text extraction.
Falls back to pypdf if fitz is unavailable.
"""

from pathlib import Path
from typing import List

from ingestion.base_loader import BaseLoader
from utils.helpers import normalize_text
from utils.logger import logger
from utils.models import Document


class PDFLoader(BaseLoader):
    """Load PDF files and extract per-page text with metadata."""

    def load(self, source: str) -> List[Document]:
        """
        Extract text from each page of a PDF.

        Args:
            source: Absolute or relative path to the PDF file.

        Returns:
            One Document per page (non-empty pages only).
        """
        path = Path(source).resolve()
        self._validate_file(path)
        logger.info(f"Loading PDF: {path.name}")

        documents = []

        try:
            import fitz  # PyMuPDF

            pdf = fitz.open(str(path))
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                raw_text = page.get_text("text")
                text = normalize_text(raw_text)

                if len(text) < 20:  # Skip effectively empty pages
                    continue

                documents.append(
                    Document(
                        content=text,
                        source=path.name,
                        metadata={
                            "source": path.name,
                            "full_path": str(path),
                            "page": page_num + 1,
                            "total_pages": len(pdf),
                            "file_type": "pdf",
                        },
                    )
                )
            pdf.close()
            logger.info(f"Loaded {len(documents)} pages from {path.name}")

        except ImportError:
            logger.warning("PyMuPDF not found, falling back to pypdf")
            documents = self._load_with_pypdf(path)

        except Exception as e:
            logger.error(f"Failed to load PDF {path.name}: {e}")
            raise RuntimeError(f"PDF loading failed for {path.name}: {e}") from e

        return documents

    def _load_with_pypdf(self, path: Path) -> List[Document]:
        """Fallback PDF loader using pypdf library."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            documents = []
            for page_num, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                text = normalize_text(raw_text)
                if len(text) < 20:
                    continue
                documents.append(
                    Document(
                        content=text,
                        source=path.name,
                        metadata={
                            "source": path.name,
                            "full_path": str(path),
                            "page": page_num + 1,
                            "total_pages": len(reader.pages),
                            "file_type": "pdf",
                        },
                    )
                )
            return documents

        except ImportError as e:
            raise ImportError(
                "Neither PyMuPDF nor pypdf is installed. "
                "Run: pip install pymupdf pypdf"
            ) from e
