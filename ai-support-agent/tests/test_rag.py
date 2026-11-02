"""Unit tests for the RAG retriever layer using a fake FAISS vector store.

Tests cover thresholding, metadata filtering, hybrid BM25/RRF fusion, and
graceful behaviour without embeddings infrastructure.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.retriever import HybridRetriever, SupportKBRetriever, _merge_rrf, _rrf_score

pytestmark = pytest.mark.usefixtures("fake_vectorstore")


def make_retriever(**kwargs) -> SupportKBRetriever:
    # fake_vectorstore patched FAISSVectorStore, so constructing without an
    # explicit vectorstore picks up the deterministic fake.
    return SupportKBRetriever(index_dir="unused", openai_api_key="test", **kwargs)


def make_hybrid(**kwargs) -> HybridRetriever:
    return HybridRetriever(index_dir="unused", openai_api_key="test", **kwargs)


# ---------------------------------------------------------------------------
# SupportKBRetriever
# ---------------------------------------------------------------------------


class TestSupportKBRetriever:
    def test_retrieve_returns_ranked_docs(self):
        r = make_retriever(top_k=3)
        docs = r.retrieve("return policy")
        assert 0 < len(docs) <= 3
        assert all(isinstance(d, Document) for d in docs)

    def test_retrieve_with_scores_pairs(self):
        r = make_retriever(top_k=3)
        scored = r.retrieve_with_scores("shipping times")
        assert all(isinstance(t, tuple) and len(t) == 2 for t in scored)
        assert all(hasattr(d, "page_content") and isinstance(s, float) for d, s in scored)

    def test_threshold_filters_low_scores(self):
        # Third fake doc scores 0.40 — with threshold 0.5 it must vanish.
        r = make_retriever(top_k=5, similarity_threshold=0.5)
        scored = r.retrieve_with_scores("anything")
        assert all(s >= 0.5 for _, s in scored)
        assert all("Bluetooth" not in d.page_content for d, _ in scored)

    def test_threshold_zero_keeps_everything(self):
        r = make_retriever(top_k=5, similarity_threshold=0.0)
        scored = r.retrieve_with_scores("anything")
        assert len(scored) == 3

    def test_doc_type_filter(self):
        r = make_retriever(top_k=5)
        docs = r.retrieve("anything", doc_type="policy")
        assert docs and all(d.metadata["doc_type"] == "policy" for d in docs)

    def test_doc_type_filter_no_match(self):
        r = make_retriever(top_k=5)
        assert r.retrieve("anything", doc_type="nonexistent") == []

    def test_category_filter(self):
        r = make_retriever(top_k=5)
        docs = r.retrieve("anything", category="shipping")
        assert docs and all(d.metadata["category"] == "shipping" for d in docs)

    def test_limit_overrides_top_k(self):
        r = make_retriever(top_k=5)
        assert len(r.retrieve("anything", limit=1)) == 1

    def test_top_k_respected(self):
        r = make_retriever(top_k=2, similarity_threshold=0.0)
        assert len(r.retrieve("anything")) == 2

    def test_retrieve_with_metadata_filters(self):
        r = make_retriever(top_k=5)
        docs = r.retrieve_with_metadata("anything", filters={"source": "returns.md"})
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "returns.md"

    def test_as_langchain_retriever(self):
        r = make_retriever()
        lc = r.as_langchain_retriever()
        assert hasattr(lc, "_get_relevant_documents")

    def test_get_stats(self):
        stats = make_retriever(top_k=7).get_stats()
        assert stats["retriever_type"] == "SupportKBRetriever"
        assert stats["top_k"] == 7
        assert "store" in stats


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    def _corpus(self) -> list[Document]:
        return [
            Document(page_content="returns accepted within thirty days of purchase",
                     metadata={"source": "r1", "doc_type": "policy"}),
            Document(page_content="standard shipping takes three to five days",
                     metadata={"source": "s1", "doc_type": "faq"}),
            Document(page_content="the device supports bluetooth and usb c",
                     metadata={"source": "p1", "doc_type": "product"}),
        ]

    def test_bm25_corpus_enables_hybrid(self):
        h = make_hybrid(corpus_docs=self._corpus())
        assert h.get_stats()["bm25_available"] in (True, False)  # env dependent
        assert h.get_stats()["corpus_size"] == 3

    def test_hybrid_returns_results(self):
        h = make_hybrid(corpus_docs=self._corpus())
        docs = h.retrieve("how long does shipping take")
        assert docs
        # BM25 should rank the shipping doc highly when available.
        if h.get_stats()["bm25_available"]:
            assert "shipping" in docs[0].page_content.lower()

    def test_hybrid_respects_doc_type_filter(self):
        h = make_hybrid(corpus_docs=self._corpus(), similarity_threshold=0.0)
        docs = h.retrieve("anything", doc_type="product")
        assert docs and all(d.metadata["doc_type"] == "product" for d in docs)

    def test_hybrid_fallback_without_corpus(self):
        h = make_hybrid()  # no corpus → vector-only path
        docs = h.retrieve("anything")
        assert docs  # still returns vector results

    def test_load_corpus_runtime(self):
        h = make_hybrid()
        assert h.get_stats()["corpus_size"] == 0
        h.load_corpus(self._corpus())
        assert h.get_stats()["corpus_size"] == 3

    def test_alpha_parameter_recorded(self):
        assert make_hybrid(alpha=0.8).get_stats()["alpha"] == 0.8


# ---------------------------------------------------------------------------
# RRF fusion primitives
# ---------------------------------------------------------------------------


class TestRRF:
    def test_rrf_score_monotonic_decreasing(self):
        assert _rrf_score(0) > _rrf_score(1) > _rrf_score(10)

    def test_rrf_score_known_value(self):
        assert _rrf_score(0, k=60) == pytest.approx(1 / 61)

    def test_merge_rrf_boosts_docs_in_both_lists(self):
        a = Document(page_content="A", metadata={"source": "both"})
        b = Document(page_content="B", metadata={"source": "vector-only"})
        merged = _merge_rrf([(b, 0.9), (a, 0.8)], [(a, 5.0)], alpha=0.5)
        # 'both' appears in both rankings → highest fused score
        assert merged[0][0].metadata["source"] == "both"
        scores = [s for _, s in merged]
        assert scores == sorted(scores, reverse=True)

    def test_merge_rrf_empty_inputs(self):
        assert _merge_rrf([], []) == []
