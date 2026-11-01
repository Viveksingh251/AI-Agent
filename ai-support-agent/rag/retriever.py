"""Retrieval layer for the AI Support Agent RAG pipeline.

Provides two retriever classes:

- :class:`SupportKBRetriever` — pure vector similarity retriever with score
  thresholding and metadata filtering.
- :class:`HybridRetriever` — combines vector search with BM25 keyword search
  using Reciprocal Rank Fusion (RRF). Requires ``rank_bm25``; gracefully
  degrades to pure vector search if the package is absent.

All heavy imports are guarded — this module is safely importable even when
LangChain / FAISS packages are not installed.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

logger = logging.getLogger("support_agent.rag.retriever")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from langchain_core.documents import Document  # type: ignore

    _LC_AVAILABLE = True
except ImportError:
    _LC_AVAILABLE = False

try:
    from langchain_core.retrievers import BaseRetriever  # type: ignore
    from langchain_core.callbacks import CallbackManagerForRetrieverRun  # type: ignore

    _BASE_RETRIEVER_AVAILABLE = True
except ImportError:
    _BASE_RETRIEVER_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

if TYPE_CHECKING:
    from rag.vectorstore import FAISSVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a document at position *rank* (0-indexed)."""
    return 1.0 / (k + rank + 1)


def _merge_rrf(
    vector_results: List[Tuple["Document", float]],
    bm25_results: List[Tuple["Document", float]],
    alpha: float = 0.5,
) -> List[Tuple["Document", float]]:
    """Merge vector and BM25 results using Reciprocal Rank Fusion.

    Args:
        vector_results: (Document, score) pairs from vector search (ranked).
        bm25_results: (Document, score) pairs from BM25 (ranked).
        alpha: Weight for vector RRF score vs BM25 RRF score.

    Returns:
        Merged list of (Document, fused_score) tuples, sorted descending.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, "Document"] = {}

    for rank, (doc, _) in enumerate(vector_results):
        key = doc.metadata.get("source", doc.page_content[:80])
        scores[key] = scores.get(key, 0.0) + alpha * _rrf_score(rank)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.metadata.get("source", doc.page_content[:80])
        scores[key] = scores.get(key, 0.0) + (1.0 - alpha) * _rrf_score(rank)
        doc_map[key] = doc

    merged = [(doc_map[k], scores[k]) for k in scores]
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


# ---------------------------------------------------------------------------
# SupportKBRetriever
# ---------------------------------------------------------------------------


class SupportKBRetriever:
    """Vector-similarity retriever for the support knowledge base.

    Wraps :class:`rag.vectorstore.FAISSVectorStore` and adds:
    - Score-threshold filtering (drop results below ``similarity_threshold``).
    - Metadata filtering by ``doc_type`` and/or ``category``.
    - LangChain ``BaseRetriever`` adapter via :meth:`as_langchain_retriever`.

    Parameters
    ----------
    vectorstore:
        A :class:`rag.vectorstore.FAISSVectorStore` instance. If ``None``,
        one is created from environment / provided kwargs.
    top_k:
        Maximum number of results to return per query.
    similarity_threshold:
        Minimum similarity score (0–1) for a result to be included.
        FAISS L2 distances are converted to [0, 1] via ``1/(1+d)``.
    index_dir:
        Passed to ``FAISSVectorStore`` when *vectorstore* is ``None``.
    openai_api_key:
        OpenAI API key for embeddings.
    hf_embedding_model:
        HuggingFace model name (fallback embeddings).
    """

    def __init__(
        self,
        vectorstore: Optional["FAISSVectorStore"] = None,
        top_k: int = 4,
        similarity_threshold: float = 0.3,
        index_dir: str = "./data/faiss_index",
        index_name: str = "support_kb",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        hf_embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        if vectorstore is not None:
            self._store = vectorstore
        else:
            from rag.vectorstore import FAISSVectorStore

            self._store = FAISSVectorStore(
                index_dir=index_dir,
                index_name=index_name,
                openai_api_key=openai_api_key,
                embedding_model=embedding_model,
                hf_embedding_model=hf_embedding_model,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        results: List[Tuple["Document", float]],
        doc_type: Optional[str] = None,
        category: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple["Document", float]]:
        """Filter results by metadata predicates."""
        filtered = []
        for doc, score in results:
            meta = doc.metadata

            if doc_type and meta.get("doc_type") != doc_type:
                continue
            if category and meta.get("category") != category:
                continue
            if extra_filters:
                if not all(meta.get(k) == v for k, v in extra_filters.items()):
                    continue

            filtered.append((doc, score))
        return filtered

    def _apply_threshold(
        self,
        results: List[Tuple["Document", float]],
    ) -> List[Tuple["Document", float]]:
        """Remove results below ``self.similarity_threshold``."""
        return [(doc, score) for doc, score in results if score >= self.similarity_threshold]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        doc_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List["Document"]:
        """Retrieve knowledge-base documents relevant to *query*.

        Args:
            query: Natural language question or statement.
            doc_type: Filter by document type (e.g. ``"faq"``, ``"policy"``).
            category: Filter by category (e.g. ``"shipping"``, ``"returns"``).
            limit: Override ``self.top_k`` for this call.

        Returns:
            List of :class:`langchain_core.documents.Document` objects,
            sorted by descending similarity.
        """
        k = limit or self.top_k
        # Fetch extra candidates to account for threshold filtering.
        fetch_k = k * 3

        raw = self._store.similarity_search_with_score(query, k=fetch_k)
        passed = self._apply_threshold(raw)
        filtered = self._apply_filters(passed, doc_type=doc_type, category=category)

        docs = [doc for doc, _ in filtered[:k]]
        logger.debug(
            "retrieve: query=%r → %d raw → %d above threshold → %d after filter → %d returned",
            query[:60],
            len(raw),
            len(passed),
            len(filtered),
            len(docs),
        )
        return docs

    def retrieve_with_scores(
        self,
        query: str,
        doc_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple["Document", float]]:
        """Same as :meth:`retrieve` but returns (Document, score) tuples."""
        k = limit or self.top_k
        fetch_k = k * 3

        raw = self._store.similarity_search_with_score(query, k=fetch_k)
        passed = self._apply_threshold(raw)
        filtered = self._apply_filters(passed, doc_type=doc_type, category=category)
        return filtered[:k]

    def retrieve_with_metadata(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List["Document"]:
        """Retrieve documents matching *query* with arbitrary metadata *filters*.

        Args:
            query: Search query.
            filters: Dict of metadata key → expected value pairs.
            limit: Override default top-k.

        Returns:
            Filtered list of ``Document`` objects.
        """
        k = limit or self.top_k
        fetch_k = k * 3

        raw = self._store.similarity_search_with_score(query, k=fetch_k)
        passed = self._apply_threshold(raw)
        filtered = self._apply_filters(passed, extra_filters=filters)
        return [doc for doc, _ in filtered[:k]]

    def as_langchain_retriever(self) -> Any:
        """Return a LangChain ``BaseRetriever`` compatible object.

        The returned object can be used directly in LangChain chains /
        ``ConversationalRetrievalChain``.
        """
        if not _BASE_RETRIEVER_AVAILABLE:
            logger.warning(
                "langchain_core.retrievers.BaseRetriever not available — "
                "returning FAISS native retriever instead."
            )
            return self._store.as_retriever(search_kwargs={"k": self.top_k})

        retriever_instance = self

        class _LCRetriever(BaseRetriever):  # type: ignore[misc]
            """LangChain-compatible wrapper around SupportKBRetriever."""

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: "CallbackManagerForRetrieverRun",
            ) -> List["Document"]:
                return retriever_instance.retrieve(query)

            async def _aget_relevant_documents(
                self,
                query: str,
                *,
                run_manager: "CallbackManagerForRetrieverRun",
            ) -> List["Document"]:
                return retriever_instance.retrieve(query)

        return _LCRetriever()

    def get_stats(self) -> Dict[str, Any]:
        """Return retriever configuration and underlying store stats."""
        return {
            "retriever_type": "SupportKBRetriever",
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "store": self._store.get_stats(),
        }


# ---------------------------------------------------------------------------
# HybridRetriever  (BM25 + vector search via RRF)
# ---------------------------------------------------------------------------


class HybridRetriever(SupportKBRetriever):
    """Hybrid retriever combining dense vector search with sparse BM25 keyword search.

    If ``rank_bm25`` is not installed, automatically falls back to pure vector
    search and logs a warning.

    Parameters
    ----------
    corpus_docs:
        Pre-loaded list of ``Document`` objects that form the BM25 corpus.
        If ``None``, the BM25 component is disabled (fallback to vector-only).
    alpha:
        Weight for vector RRF scores (``1 - alpha`` goes to BM25).
    All other parameters are forwarded to :class:`SupportKBRetriever`.
    """

    def __init__(
        self,
        corpus_docs: Optional[List["Document"]] = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self._corpus_docs: List["Document"] = corpus_docs or []
        self._bm25: Optional[Any] = None

        if self._corpus_docs and _BM25_AVAILABLE:
            self._build_bm25_index(self._corpus_docs)
        elif self._corpus_docs and not _BM25_AVAILABLE:
            logger.warning(
                "rank_bm25 package not installed — HybridRetriever falling back to vector-only. "
                "Install with: pip install rank-bm25"
            )

    # ------------------------------------------------------------------
    # BM25 helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokeniser. Override for language-specific needs."""
        return text.lower().split()

    def _build_bm25_index(self, documents: List["Document"]) -> None:
        """Build or rebuild the BM25 index from *documents*."""
        if not _BM25_AVAILABLE:
            return
        tokenised_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(tokenised_corpus)  # type: ignore[name-defined]
        self._corpus_docs = documents
        logger.info("BM25 index built with %d documents", len(documents))

    def load_corpus(self, documents: List["Document"]) -> None:
        """Load (or replace) the BM25 corpus at runtime.

        Call this after ingestion to enable hybrid search without restart.
        """
        self._corpus_docs = documents
        if _BM25_AVAILABLE:
            self._build_bm25_index(documents)
        else:
            logger.warning("rank_bm25 not available — BM25 component disabled.")

    def _bm25_search(self, query: str, k: int) -> List[Tuple["Document", float]]:
        """Run BM25 keyword search and return top-k (Document, score) pairs."""
        if not _BM25_AVAILABLE or self._bm25 is None or not self._corpus_docs:
            return []

        tokens = self._tokenize(query)
        raw_scores = self._bm25.get_scores(tokens)  # type: ignore[attr-defined]

        # Pair with docs and sort.
        scored = list(zip(self._corpus_docs, raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Override retrieve
    # ------------------------------------------------------------------

    def retrieve(  # type: ignore[override]
        self,
        query: str,
        doc_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List["Document"]:
        """Retrieve using hybrid BM25 + vector search with RRF fusion.

        Falls back to pure vector search if BM25 is unavailable.
        """
        k = limit or self.top_k
        fetch_k = k * 3

        # Vector results.
        vector_raw = self._store.similarity_search_with_score(query, k=fetch_k)
        vector_passed = self._apply_threshold(vector_raw)
        vector_filtered = self._apply_filters(
            vector_passed, doc_type=doc_type, category=category
        )

        if not _BM25_AVAILABLE or self._bm25 is None:
            # Pure vector fallback.
            return [doc for doc, _ in vector_filtered[:k]]

        # BM25 results.
        bm25_raw = self._bm25_search(query, k=fetch_k)
        bm25_filtered = self._apply_filters(
            bm25_raw, doc_type=doc_type, category=category
        )

        # Fuse with RRF.
        fused = _merge_rrf(vector_filtered, bm25_filtered, alpha=self.alpha)
        docs = [doc for doc, _ in fused[:k]]

        logger.debug(
            "hybrid retrieve: query=%r → vector=%d, bm25=%d → fused=%d → returned=%d",
            query[:60],
            len(vector_filtered),
            len(bm25_filtered),
            len(fused),
            len(docs),
        )
        return docs

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update(
            {
                "retriever_type": "HybridRetriever",
                "alpha": self.alpha,
                "bm25_available": _BM25_AVAILABLE,
                "corpus_size": len(self._corpus_docs),
            }
        )
        return stats


# ---------------------------------------------------------------------------
# Factory from settings
# ---------------------------------------------------------------------------


def get_retriever_from_settings(
    settings: Any,
    use_hybrid: bool = False,
    corpus_docs: Optional[List[Any]] = None,
) -> SupportKBRetriever:
    """Build a retriever from app settings.

    Args:
        settings: ``app.config.Settings`` instance.
        use_hybrid: If ``True``, return a :class:`HybridRetriever`.
        corpus_docs: Documents for the BM25 corpus (only used when *use_hybrid*
                     is ``True``).
    """
    kwargs: Dict[str, Any] = {
        "top_k": getattr(settings, "TOP_K_RETRIEVAL", 4),
        "similarity_threshold": getattr(settings, "SIMILARITY_THRESHOLD", 0.3),
        "index_dir": getattr(settings, "FAISS_INDEX_DIR", "./data/faiss_index"),
        "index_name": getattr(settings, "FAISS_INDEX_NAME", "support_kb"),
        "openai_api_key": getattr(settings, "OPENAI_API_KEY", None),
        "embedding_model": getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "hf_embedding_model": getattr(settings, "HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    }

    if use_hybrid:
        return HybridRetriever(corpus_docs=corpus_docs, **kwargs)
    return SupportKBRetriever(**kwargs)
