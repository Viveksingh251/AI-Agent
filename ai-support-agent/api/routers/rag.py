"""RAG & Knowledge Base API router.

Endpoints:
    POST /api/v1/rag/query    — query the knowledge base (any authenticated user)
    POST /api/v1/rag/ingest   — trigger ingestion (admin only)
    GET  /api/v1/rag/stats    — index statistics (support_agent or admin)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.security import AuthenticatedUser, UserRole, get_current_active_user, require_roles

logger = logging.getLogger("support_agent.api.rag")

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RAGQueryRequest(BaseModel):
    """Payload for a knowledge-base query."""

    query: str = Field(..., min_length=3, max_length=2000, description="Natural language query")
    top_k: int = Field(default=4, ge=1, le=20, description="Maximum results to return")
    doc_type: Optional[str] = Field(
        default=None,
        description="Filter by document type: faq | policy | product | guide | general",
    )
    category: Optional[str] = Field(
        default=None,
        description="Filter by category: shipping | returns | payment | product | account",
    )
    threshold: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0–1, higher = stricter)",
    )
    use_hybrid: bool = Field(
        default=False,
        description="Use hybrid BM25 + vector search (requires rank-bm25 package)",
    )


class RAGResult(BaseModel):
    """A single retrieval result."""

    content: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Similarity score (0–1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class RAGQueryResponse(BaseModel):
    """Response for a knowledge-base query."""

    query: str
    results: List[RAGResult]
    total_found: int
    retriever_type: str
    threshold_applied: float


class IngestRequest(BaseModel):
    """Payload for triggering document ingestion."""

    docs_dir: str = Field(default="./data/knowledge_base", description="Source documents directory")
    reset: bool = Field(default=False, description="Delete existing index before ingesting")
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    batch_size: int = Field(default=100, ge=10, le=500)


class IngestResponse(BaseModel):
    """Response after triggering ingestion."""

    status: str
    documents_loaded: int
    chunks_created: int
    chunks_indexed: int
    elapsed_seconds: float
    error: Optional[str] = None


class IndexStats(BaseModel):
    """Knowledge-base index statistics."""

    backend: str
    index_path: Optional[str] = None
    document_count: int
    index_exists: bool
    retriever_type: str
    top_k_default: int
    similarity_threshold: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_retriever(settings: Settings, top_k: int, threshold: float, use_hybrid: bool = False):
    """Build a retriever from settings — with graceful import failure."""
    try:
        from rag.retriever import HybridRetriever, SupportKBRetriever

        kwargs: Dict[str, Any] = {
            "top_k": top_k,
            "similarity_threshold": threshold,
            "index_dir": settings.FAISS_INDEX_DIR,
            "index_name": getattr(settings, "FAISS_INDEX_NAME", "support_kb"),
            "openai_api_key": settings.OPENAI_API_KEY,
            "embedding_model": getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "hf_embedding_model": getattr(settings, "HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        }

        if use_hybrid:
            return HybridRetriever(**kwargs)
        return SupportKBRetriever(**kwargs)

    except Exception as exc:
        logger.error("Failed to initialise retriever: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG retriever unavailable: {exc}",
        )


def _get_pipeline(settings: Settings):
    """Build an ingestion pipeline from settings — with graceful import failure."""
    try:
        from rag.ingest import DocumentIngestionPipeline

        return DocumentIngestionPipeline(
            index_dir=settings.FAISS_INDEX_DIR,
            index_name=getattr(settings, "FAISS_INDEX_NAME", "support_kb"),
            openai_api_key=settings.OPENAI_API_KEY,
            embedding_model=getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            hf_embedding_model=getattr(settings, "HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
    except Exception as exc:
        logger.error("Failed to initialise ingestion pipeline: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ingestion pipeline unavailable: {exc}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Query the knowledge base",
    description=(
        "Retrieve relevant documents from the company knowledge base "
        "(policies, FAQs, product docs, shipping info) using semantic search."
    ),
)
async def query_knowledge_base(
    payload: RAGQueryRequest,
    current_user: AuthenticatedUser = Depends(get_current_active_user),
    settings: Settings = Depends(get_settings),
) -> RAGQueryResponse:
    """Search the knowledge base for documents relevant to *payload.query*.

    Available to all authenticated users (customers, support agents, admins).
    """
    threshold = payload.threshold if payload.threshold is not None else 0.3

    retriever = _get_retriever(
        settings,
        top_k=payload.top_k,
        threshold=threshold,
        use_hybrid=payload.use_hybrid,
    )

    try:
        scored_results = retriever.retrieve_with_scores(
            query=payload.query,
            doc_type=payload.doc_type,
            category=payload.category,
            limit=payload.top_k,
        )
    except Exception as exc:
        logger.error("Retrieval failed for query %r: %s", payload.query[:60], exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {exc}",
        )

    results = [
        RAGResult(
            content=doc.page_content,
            score=round(score, 4),
            metadata=doc.metadata,
        )
        for doc, score in scored_results
    ]

    retriever_type = type(retriever).__name__

    logger.info(
        "RAG query by user=%s role=%s → %d results",
        current_user.user_id,
        current_user.role,
        len(results),
    )

    return RAGQueryResponse(
        query=payload.query,
        results=results,
        total_found=len(results),
        retriever_type=retriever_type,
        threshold_applied=threshold,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger document ingestion (admin only)",
    description=(
        "Run the document ingestion pipeline to (re-)index knowledge-base "
        "documents into the FAISS vector store. Admin role required."
    ),
)
async def trigger_ingestion(
    payload: IngestRequest,
    current_user: AuthenticatedUser = Depends(require_roles(UserRole.ADMIN)),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Synchronously run the ingestion pipeline and return stats.

    For large document sets, consider moving this to a background task.

    .. note::
        This endpoint runs ingestion **synchronously** in the request/response
        cycle. For production use with large corpora, wrap with
        ``fastapi.BackgroundTasks`` or delegate to a Celery / ARQ worker.
    """
    pipeline = _get_pipeline(settings)

    # Override pipeline settings from request.
    pipeline.chunk_size = payload.chunk_size
    pipeline.chunk_overlap = payload.chunk_overlap

    try:
        stats = pipeline.run(
            source_dir=payload.docs_dir,
            reset=payload.reset,
            batch_size=payload.batch_size,
        )
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {exc}",
        )

    logger.info(
        "Ingestion triggered by admin=%s → %s",
        current_user.user_id,
        stats.get("status"),
    )

    return IngestResponse(
        status=stats.get("status", "unknown"),
        documents_loaded=stats.get("documents_loaded", 0),
        chunks_created=stats.get("chunks_created", 0),
        chunks_indexed=stats.get("chunks_indexed", 0),
        elapsed_seconds=stats.get("elapsed_seconds", 0.0),
        error=stats.get("error"),
    )


@router.get(
    "/stats",
    response_model=IndexStats,
    summary="Knowledge-base index statistics",
    description=(
        "Return current FAISS index statistics: document count, path, backend, "
        "and retriever configuration. Requires support_agent or admin role."
    ),
)
async def get_index_stats(
    current_user: AuthenticatedUser = Depends(
        require_roles(UserRole.SUPPORT_AGENT, UserRole.ADMIN)
    ),
    settings: Settings = Depends(get_settings),
) -> IndexStats:
    """Return live statistics for the knowledge-base vector index."""
    try:
        from rag.vectorstore import get_vectorstore_from_settings

        store = get_vectorstore_from_settings(settings)
        raw_stats = store.get_stats()
    except Exception as exc:
        logger.error("Could not fetch index stats: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Index stats unavailable: {exc}",
        )

    return IndexStats(
        backend=raw_stats.get("backend", "unknown"),
        index_path=raw_stats.get("index_path"),
        document_count=raw_stats.get("document_count", -1),
        index_exists=raw_stats.get("index_exists", False),
        retriever_type="SupportKBRetriever",
        top_k_default=getattr(settings, "TOP_K_RETRIEVAL", 4),
        similarity_threshold=getattr(settings, "SIMILARITY_THRESHOLD", 0.3),
    )

