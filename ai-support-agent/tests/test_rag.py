"""Tests for Document Ingestion Pipeline and Retriever."""

import pytest
from langchain_core.documents import Document

from rag.ingest import DocumentIngestionPipeline
from rag.retriever import HybridRetriever


def test_ingestion_pipeline_initialization():
    """Verifies DocumentIngestionPipeline parameter defaults."""
    pipeline = DocumentIngestionPipeline(chunk_size=800, chunk_overlap=150)
    assert pipeline.chunk_size == 800
    assert pipeline.chunk_overlap == 150


def test_document_splitting():
    """Verifies document splitting skeleton behavior."""
    pipeline = DocumentIngestionPipeline()
    sample_docs = [
        Document(page_content="Policy clause 1. Policy clause 2.", metadata={"source": "faq.md"})
    ]
    chunks = pipeline.split_documents(sample_docs)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "faq.md"


@pytest.mark.asyncio
async def test_retriever_retrieve():
    """Verifies that HybridRetriever returns ranked documents for query."""
    retriever = HybridRetriever(top_k=3)
    results = await retriever.retrieve("How do I return a damaged product?")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert hasattr(results[0], "page_content")

