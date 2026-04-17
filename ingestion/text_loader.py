"""
Text and Markdown Document Loaders
Plain-text files are loaded as a single Document.
Markdown files are loaded similarly but strip markdown formatting.
"""

import re
from pathlib import Path
from typing import List

from ingestion.base_loader import BaseLoader
from utils.helpers import normalize_text
from utils.logger import logger
from utils.models import Document


class TextLoader(BaseLoader):
    """Load plain .txt files as a single Document."""

    def load(self, source: str) -> List[Document]:
        path = Path(source).resolve()
        self._validate_file(path)
        logger.info(f"Loading text file: {path.name}")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            text = normalize_text(text)
        except Exception as e:
            logger.error(f"Failed to read {path.name}: {e}")
            raise RuntimeError(f"Text loading failed for {path.name}: {e}") from e

        if not text:
            logger.warning(f"Text file {path.name} is empty after normalization")
            return []

        return [
            Document(
                content=text,
                source=path.name,
                metadata={
                    "source": path.name,
                    "full_path": str(path),
                    "page": 1,
                    "file_type": "txt",
                },
            )
        ]


class MarkdownLoader(BaseLoader):
    """Load .md / .markdown files with optional markdown stripping."""

    # Patterns to strip markdown syntax but preserve readable text
    _MD_PATTERNS = [
        (r"^#{1,6}\s+", ""),        # Headings: # ## ###
        (r"\*{1,2}(.+?)\*{1,2}", r"\1"),  # Bold/italic
        (r"`{1,3}[^`]*`{1,3}", ""),  # Inline code / code blocks
        (r"!\[.*?\]\(.*?\)", ""),    # Images
        (r"\[(.+?)\]\(.*?\)", r"\1"),  # Links → keep text
        (r"^\s*[-*+]\s+", ""),       # Unordered list items
        (r"^\s*\d+\.\s+", ""),       # Ordered list items
        (r"^\s*>\s+", ""),           # Blockquotes
        (r"---+|===+|___+", ""),     # Horizontal rules
        (r"\|", " "),                # Table pipes → space
    ]

    def load(self, source: str) -> List[Document]:
        path = Path(source).resolve()
        self._validate_file(path)
        logger.info(f"Loading Markdown file: {path.name}")

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            text = self._strip_markdown(raw)
            text = normalize_text(text)
        except Exception as e:
            logger.error(f"Failed to read {path.name}: {e}")
            raise RuntimeError(f"Markdown loading failed for {path.name}: {e}") from e

        if not text:
            logger.warning(f"Markdown file {path.name} is empty after stripping")
            return []

        return [
            Document(
                content=text,
                source=path.name,
                metadata={
                    "source": path.name,
                    "full_path": str(path),
                    "page": 1,
                    "file_type": "markdown",
                },
            )
        ]

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown syntax while preserving readable content."""
        for pattern, replacement in self._MD_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        return text
