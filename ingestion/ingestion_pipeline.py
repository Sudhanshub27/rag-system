"""
Document Ingestion Pipeline
Routes files to the appropriate loader, validates inputs, and returns
a flat list of Document objects ready for chunking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Type

from config import ingestion_config
from ingestion.base_loader import BaseLoader
from ingestion.pdf_loader import PDFLoader
from ingestion.text_loader import MarkdownLoader, TextLoader
from utils.logger import logger
from utils.models import Document


class DocumentIngestionPipeline:
    """
    Orchestrates document loading across multiple file types.

    Usage:
        pipeline = DocumentIngestionPipeline()
        docs = pipeline.ingest("path/to/file.pdf")
        docs = pipeline.ingest_directory("path/to/docs/")
    """

    # Registry: extension → loader class
    _LOADER_REGISTRY: Dict[str, Type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
    }

    def __init__(self, extra_loaders: Optional[Dict[str, Type[BaseLoader]]] = None):
        """
        Args:
            extra_loaders: Optional dict of {".ext": LoaderClass} to extend
                           the built-in registry.
        """
        self._loaders: Dict[str, BaseLoader] = {}
        if extra_loaders:
            self._LOADER_REGISTRY.update(extra_loaders)

    def _get_loader(self, extension: str) -> BaseLoader:
        """Return a cached loader instance for the given extension."""
        ext = extension.lower()
        if ext not in self._LOADER_REGISTRY:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                f"Supported types: {list(self._LOADER_REGISTRY.keys())}"
            )
        if ext not in self._loaders:
            self._loaders[ext] = self._LOADER_REGISTRY[ext]()
        return self._loaders[ext]

    def ingest(self, source: str) -> List[Document]:
        """
        Ingest a single file.

        Args:
            source: Path to the document.

        Returns:
            List of Document objects.

        Raises:
            ValueError: For unsupported file types.
            RuntimeError: For loading failures.
        """
        path = Path(source)
        ext = path.suffix.lower()

        logger.info(f"Ingesting document: {path.name}")

        # File size guard
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > ingestion_config.max_file_size_mb:
                raise ValueError(
                    f"File {path.name} ({size_mb:.1f} MB) exceeds "
                    f"the {ingestion_config.max_file_size_mb} MB limit."
                )

        loader = self._get_loader(ext)
        docs = loader.load(source)
        logger.info(f"Ingested {len(docs)} document(s) from {path.name}")
        return docs

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Ingest all supported files in a directory.

        Args:
            directory: Path to the directory.
            recursive: If True, search subdirectories as well.

        Returns:
            Flat list of all Document objects.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        all_docs: List[Document] = []
        failed: List[str] = []

        supported_exts = set(ingestion_config.supported_extensions)
        files = sorted(
            f for f in dir_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_exts
        )

        if not files:
            logger.warning(f"No supported files found in {directory}")
            return []

        logger.info(f"Found {len(files)} file(s) to ingest in {directory}")

        for file in files:
            try:
                docs = self.ingest(str(file))
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to ingest {file.name}: {e}")
                failed.append(str(file))

        if failed:
            logger.warning(f"Failed to load {len(failed)} file(s): {failed}")

        logger.info(
            f"Directory ingestion complete. "
            f"{len(all_docs)} total documents loaded, {len(failed)} failed."
        )
        return all_docs
