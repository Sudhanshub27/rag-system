"""
Base class for all document loaders.
Each loader handles a specific file type and returns a list of Documents.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from utils.models import Document


class BaseLoader(ABC):
    """Abstract document loader interface."""

    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        Load one or more Documents from a source path or URL.

        Args:
            source: File path (str) or URL.

        Returns:
            List of Document objects.
        """
        ...

    def _validate_file(self, path: Path, max_size_mb: int = 100) -> None:
        """Raise informative errors for missing or oversized files."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(
                f"File {path.name} is {size_mb:.1f} MB, "
                f"exceeding the {max_size_mb} MB limit."
            )
