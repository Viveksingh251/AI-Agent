"""Vector store management for the AI Support Agent RAG pipeline.

Provides a unified interface over FAISS (local) and pgvector (production).
All heavy imports are guarded so the module is importable even when LangChain
/ FAISS packages are not installed in the current environment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

logger = logging.getLogger("support_agent.rag.vectorstore")

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation when packages are absent
# ---------------------------------------------------------------------------

try:
    from langchain_community.vectorstores import FAISS as FAISSStore  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("langchain_community not available — FAISS vector store disabled")

try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore

    _OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    _OPENAI_EMBEDDINGS_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

    _HF_EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

        _HF_EMBEDDINGS_AVAILABLE = True
    except ImportError:
        _HF_EMBEDDINGS_AVAILABLE = False

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStoreRetriever


# ---------------------------------------------------------------------------
# Embedding factory
# ---------------------------------------------------------------------------


def _build_embeddings(
    openai_api_key: Optional[str] = None,
    model_name: str = "text-embedding-3-small",
    hf_model_name: str = "all-MiniLM-L6-v2",
) -> "Embeddings":
    """Return best available embeddings implementation.

    Priority:
    1. OpenAIEmbeddings  (requires OPENAI_API_KEY)
    2. HuggingFaceEmbeddings  (local, no API key needed)
    3. Raise RuntimeError if neither is available.
    """
    if _OPENAI_EMBEDDINGS_AVAILABLE and openai_api_key:
        logger.info("Using OpenAI embeddings — model=%s", model_name)
        return OpenAIEmbeddings(api_key=openai_api_key, model=model_name)  # type: ignore[call-arg]

    if _HF_EMBEDDINGS_AVAILABLE:
        logger.info("Using HuggingFace sentence-transformers — model=%s", hf_model_name)
        return HuggingFaceEmbeddings(model_name=hf_model_name)  # type: ignore[call-arg]

    raise RuntimeError(
        "No embedding backend available. "
        "Install 'langchain-openai' (with OPENAI_API_KEY) "
        "or 'langchain-huggingface' + 'sentence-transformers'."
    )


# ---------------------------------------------------------------------------
# FAISSVectorStore
# ---------------------------------------------------------------------------


class FAISSVectorStore:
    """Thin wrapper around LangChain's FAISS integration with persistence.

    Features:
    - Lazy load/create of the underlying FAISS index.
    - Automatic persistence after indexing.
    - Similarity search with L2-distance → cosine-like normalised scores.
    - Metadata filtering post-search (FAISS doesn't support native filters).
    """

    def __init__(
        self,
        index_dir: str = "./data/faiss_index",
        index_name: str = "support_kb",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        hf_embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS is not available. "
                "Install: pip install faiss-cpu langchain-community"
            )
        self.index_dir = Path(index_dir)
        self.index_name = index_name
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.hf_embedding_model = hf_embedding_model

        self._embeddings: Optional["Embeddings"] = None
        self._store: Optional[FAISSStore] = None  # type: ignore[valid-type]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embeddings(self) -> "Embeddings":
        if self._embeddings is None:
            self._embeddings = _build_embeddings(
                openai_api_key=self.openai_api_key,
                model_name=self.embedding_model,
                hf_model_name=self.hf_embedding_model,
            )
        return self._embeddings

    @property
    def _index_path(self) -> Path:
        return self.index_dir / self.index_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_or_create(self) -> "FAISSStore":
        """Load persisted FAISS index from disk, or create a new empty one.

        Returns the underlying LangChain FAISS store object.
        """
        if self._store is not None:
            return self._store

        embeddings = self._get_embeddings()
        idx_path = self._index_path

        if idx_path.exists() and (idx_path / "index.faiss").exists():
            logger.info("Loading existing FAISS index from %s", idx_path)
            self._store = FAISSStore.load_local(  # type: ignore[attr-defined]
                str(idx_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.info("No existing FAISS index found — creating new store at %s", idx_path)
            # Create a minimal store from a placeholder document, then clear it.
            # FAISS requires at least one document to initialise.
            from langchain_core.documents import Document  # type: ignore

            placeholder = Document(
                page_content="__placeholder__",
                metadata={"source": "__init__", "doc_type": "internal"},
            )
            self._store = FAISSStore.from_documents([placeholder], embeddings)  # type: ignore[attr-defined]
            self.save()

        return self._store

    def save(self) -> None:
        """Persist the current FAISS index to disk."""
        if self._store is None:
            logger.warning("Nothing to save — store not initialised.")
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._index_path))
        logger.info("FAISS index saved → %s", self._index_path)

    def add_documents(self, documents: List["Document"], batch_size: int = 100) -> int:
        """Embed *documents* and upsert into the FAISS index.

        Args:
            documents: LangChain ``Document`` objects to index.
            batch_size: Number of documents to embed per batch.

        Returns:
            Total number of documents added.
        """
        store = self.load_or_create()
        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            store.add_documents(batch)
            total += len(batch)
            logger.debug("Indexed batch %d–%d (%d docs)", i, i + len(batch), total)
            self.save()  # persist after each batch

        logger.info("Added %d documents to FAISS index", total)
        return total

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple["Document", float]]:
        """Search for documents similar to *query*.

        Args:
            query: Natural language query string.
            k: Maximum number of results to return.
            filter: Optional metadata key-value pairs to restrict results.
                    Applied as a post-search filter (FAISS has no native filter).

        Returns:
            List of (Document, score) tuples.  Score is a normalised cosine-
            like similarity in [0, 1] (higher is better), converted from the
            L2 distance FAISS returns internally.
        """
        store = self.load_or_create()
        # Fetch more candidates so post-filtering still returns k results.
        fetch_k = k * 4 if filter else k
        raw: List[Tuple["Document", float]] = store.similarity_search_with_score(  # type: ignore[attr-defined]
            query, k=fetch_k
        )

        # Convert L2 distance → similarity score in [0, 1].
        # FAISS L2 distances are unbounded; we use exponential decay.
        results: List[Tuple["Document", float]] = []
        for doc, l2_dist in raw:
            similarity = float(1.0 / (1.0 + l2_dist))
            results.append((doc, similarity))

        # Apply metadata filter if provided.
        if filter:
            filtered: List[Tuple["Document", float]] = []
            for doc, score in results:
                if all(doc.metadata.get(key) == val for key, val in filter.items()):
                    filtered.append((doc, score))
            results = filtered

        # Skip internal placeholder documents.
        results = [(doc, score) for doc, score in results if doc.page_content != "__placeholder__"]

        return results[:k]

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> "VectorStoreRetriever":
        """Return a LangChain-compatible ``BaseRetriever`` for use in chains."""
        store = self.load_or_create()
        kwargs = search_kwargs or {"k": 4}
        return store.as_retriever(search_kwargs=kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Return basic statistics about the current index."""
        store = self.load_or_create()
        try:
            n_docs = store.index.ntotal  # type: ignore[attr-defined]
        except AttributeError:
            n_docs = -1

        return {
            "backend": "faiss",
            "index_path": str(self._index_path),
            "document_count": n_docs,
            "index_exists": (self._index_path / "index.faiss").exists(),
        }

    def reset(self) -> None:
        """Delete the persisted index from disk and reset the in-memory store."""
        import shutil

        if self._index_path.exists():
            shutil.rmtree(self._index_path)
            logger.info("Deleted FAISS index at %s", self._index_path)
        self._store = None


# ---------------------------------------------------------------------------
# PGVectorStore stub (future production backend)
# ---------------------------------------------------------------------------


class PGVectorStore:
    """Stub for pgvector backend — swap in when moving to production PostgreSQL.

    TODO: Implement using ``langchain_community.vectorstores.PGVector`` or
    ``langchain_postgres.PGVector`` (newer package).
    """

    def __init__(
        self,
        connection_string: str,
        collection_name: str = "support_kb",
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key
        logger.warning(
            "PGVectorStore is a stub — not yet implemented. "
            "Use FAISSVectorStore for local development."
        )

    def add_documents(self, documents: List["Document"], batch_size: int = 100) -> int:
        """TODO: Implement PGVector document upsert."""
        raise NotImplementedError("PGVectorStore.add_documents not yet implemented.")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple["Document", float]]:
        """TODO: Implement PGVector similarity search."""
        raise NotImplementedError("PGVectorStore.similarity_search_with_score not yet implemented.")

    def get_stats(self) -> Dict[str, Any]:
        """TODO: Return pgvector collection stats."""
        return {
            "backend": "pgvector",
            "collection": self.collection_name,
            "document_count": -1,
            "note": "stub — not implemented",
        }


# ---------------------------------------------------------------------------
# VectorStoreManager — singleton factory
# ---------------------------------------------------------------------------


class VectorStoreManager:
    """Centralised factory for vector store backends.

    Call :func:`get_store` with ``backend="faiss"`` (default) or
    ``backend="pgvector"``.  Each backend is lazily instantiated and cached.
    """

    def __init__(self) -> None:
        self._faiss: Optional[FAISSVectorStore] = None
        self._pgvector: Optional[PGVectorStore] = None

    def get_faiss_store(
        self,
        index_dir: str = "./data/faiss_index",
        index_name: str = "support_kb",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        hf_embedding_model: str = "all-MiniLM-L6-v2",
    ) -> FAISSVectorStore:
        """Return (or create) the singleton FAISSVectorStore."""
        if self._faiss is None:
            self._faiss = FAISSVectorStore(
                index_dir=index_dir,
                index_name=index_name,
                openai_api_key=openai_api_key,
                embedding_model=embedding_model,
                hf_embedding_model=hf_embedding_model,
            )
        return self._faiss

    def get_pgvector_store(
        self,
        connection_string: str = "",
        collection_name: str = "support_kb",
        openai_api_key: Optional[str] = None,
    ) -> PGVectorStore:
        """Return (or create) the singleton PGVectorStore stub."""
        if self._pgvector is None:
            self._pgvector = PGVectorStore(
                connection_string=connection_string,
                collection_name=collection_name,
                openai_api_key=openai_api_key,
            )
        return self._pgvector

    def get_store(self, backend: str = "faiss", **kwargs: Any):
        """Return the vector store for *backend* (``"faiss"`` | ``"pgvector"``)."""
        if backend == "faiss":
            return self.get_faiss_store(**{k: v for k, v in kwargs.items() if k in {
                "index_dir", "index_name", "openai_api_key",
                "embedding_model", "hf_embedding_model"
            }})
        elif backend == "pgvector":
            return self.get_pgvector_store(**{k: v for k, v in kwargs.items() if k in {
                "connection_string", "collection_name", "openai_api_key"
            }})
        else:
            raise ValueError(f"Unknown vector store backend: {backend!r}. Choose 'faiss' or 'pgvector'.")


# ---------------------------------------------------------------------------
# Module-level singleton + convenience factory
# ---------------------------------------------------------------------------

_manager: Optional[VectorStoreManager] = None


def get_vectorstore_client() -> VectorStoreManager:
    """Return the module-level singleton :class:`VectorStoreManager`."""
    global _manager
    if _manager is None:
        _manager = VectorStoreManager()
    return _manager


def get_vectorstore_from_settings(settings: Any) -> FAISSVectorStore:
    """Convenience helper — build a FAISSVectorStore from app settings.

    Args:
        settings: ``app.config.Settings`` instance (or any object with the
                  expected attributes).

    Returns:
        Configured :class:`FAISSVectorStore` singleton.
    """
    manager = get_vectorstore_client()
    return manager.get_faiss_store(
        index_dir=getattr(settings, "FAISS_INDEX_DIR", "./data/faiss_index"),
        index_name=getattr(settings, "FAISS_INDEX_NAME", "support_kb"),
        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
        embedding_model=getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        hf_embedding_model=getattr(settings, "HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
