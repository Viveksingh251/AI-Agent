"""RAG package exports."""

try:
    from rag.ingest import DocumentIngestionPipeline, get_pipeline_from_settings
    from rag.retriever import HybridRetriever, SupportKBRetriever, get_retriever_from_settings
    from rag.vectorstore import (
        FAISSVectorStore,
        PGVectorStore,
        VectorStoreManager,
        get_vectorstore_client,
        get_vectorstore_from_settings,
    )

    __all__ = [
        "DocumentIngestionPipeline",
        "get_pipeline_from_settings",
        "SupportKBRetriever",
        "HybridRetriever",
        "get_retriever_from_settings",
        "FAISSVectorStore",
        "PGVectorStore",
        "VectorStoreManager",
        "get_vectorstore_client",
        "get_vectorstore_from_settings",
    ]
except ImportError as _err:
    import logging

    logging.getLogger("support_agent.rag").warning(
        "RAG package partially unavailable — some dependencies missing: %s", _err
    )
    __all__ = []
