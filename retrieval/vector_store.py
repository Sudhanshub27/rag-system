"""
ChromaDB Vector Store
Persists chunk embeddings and metadata; supports add, query, and delete operations.
"""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config import vector_store_config
from utils.logger import logger
from utils.models import Chunk, RetrievedChunk


class ChromaVectorStore:
    """
    Thin wrapper around ChromaDB for storing and querying chunk embeddings.

    The store keeps the collection persistent on disk so that embeddings
    survive process restarts without recomputation.

    Args:
        persist_directory: Where ChromaDB stores its data on disk.
        collection_name:   Name of the collection to use.
    """

    def __init__(
        self,
        persist_directory: str = vector_store_config.persist_directory,
        collection_name: str = vector_store_config.collection_name,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        logger.info(
            f"Initializing ChromaDB at '{persist_directory}' "
            f"(collection: '{collection_name}')"
        )

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine distance
        )
        logger.info(
            f"ChromaDB ready — {self._collection.count()} existing chunk(s)"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        """
        Add chunks and their embeddings to the collection.

        Skips chunks whose IDs already exist (idempotent).

        Args:
            chunks:     List of Chunk objects.
            embeddings: Corresponding embedding vectors.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        # Filter out already-stored chunks
        existing_ids = set(self._collection.get(include=[])["ids"])
        new_chunks = [(c, e) for c, e in zip(chunks, embeddings)
                      if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already in vector store — nothing to add")
            return

        ids = [c.chunk_id for c, _ in new_chunks]
        docs = [c.text for c, _ in new_chunks]
        vecs = [e for _, e in new_chunks]
        metas = [self._build_metadata(c) for c, _ in new_chunks]

        # ChromaDB upsert in batches of 500 (Chroma's internal limit)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=vecs[i : i + batch_size],
                documents=docs[i : i + batch_size],
                metadatas=metas[i : i + batch_size],
            )

        logger.info(f"Added {len(new_chunks)} new chunk(s) to vector store")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-K most similar chunks.

        Args:
            query_embedding: The embedded query vector.
            top_k:           Number of results to return.
            where:           Optional ChromaDB metadata filter.

        Returns:
            List of RetrievedChunk objects sorted by similarity (descending).
        """
        n_results = min(top_k, self._collection.count())
        if n_results == 0:
            logger.warning("Vector store is empty — no results to return")
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        retrieved: List[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance ∈ [0, 2]. Convert to similarity ∈ [-1, 1].
            similarity = 1.0 - (dist / 2.0)
            chunk = Chunk(
                text=doc,
                source=meta.get("source", "unknown"),
                chunk_id=meta.get("chunk_id", ""),
                page=int(meta.get("page", 1)),
                metadata=meta,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=similarity))

        return retrieved

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks belonging to a particular source document.

        Args:
            source: The filename (as stored in metadata["source"]).

        Returns:
            Number of deleted chunks.
        """
        results = self._collection.get(where={"source": source}, include=[])
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunk(s) for source '{source}'")
        return len(ids)

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (destructive!)."""
        logger.warning(f"Resetting collection '{self.collection_name}'")
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_metadata(chunk: Chunk) -> Dict[str, Any]:
        """Flatten chunk metadata to ChromaDB-compatible scalars."""
        meta = {
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
        }
        # Forward any extra scalar metadata from the ingestion pipeline
        for k, v in chunk.metadata.items():
            if isinstance(v, (str, int, float, bool)) and k not in meta:
                meta[k] = v
        return meta
