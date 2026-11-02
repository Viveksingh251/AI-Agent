"""Knowledge Base Search Tool — real implementation backed by the RAG retriever."""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.tools import tool

from agent.tools.base import audit_log, tool_error

logger = logging.getLogger("support_agent.tools.kb")


@tool
async def search_knowledge_base(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search the official policy / FAQ knowledge base for passages relevant
    to a customer support question.

    Use this for questions about return policy, shipping, warranties,
    product specs, or general how-to questions. Do NOT use this for
    customer-specific data (orders, tickets, account details).

    Args:
        query: The customer question or keywords to search for.
        top_k: Maximum number of passages to return (1-10).

    Returns:
        Structured dict: {"success": bool, "query": str,
        "results": [{"content", "score", "source", "doc_type"}], "count": int}
    """
    action = "search_knowledge_base"
    top_k = max(1, min(int(top_k), 10))
    try:
        if not query or not query.strip():
            return {"success": False, "error": "Query must not be empty.", "results": []}

        docs: list[dict[str, Any]] = []
        try:
            from app.config import get_settings  # noqa: PLC0415
            from rag.retriever import SupportKBRetriever  # noqa: PLC0415

            settings = get_settings()
            retriever = SupportKBRetriever(
                top_k=top_k,
                similarity_threshold=getattr(settings, "SIMILARITY_THRESHOLD", 0.3),
                index_dir=settings.FAISS_INDEX_DIR,
                openai_api_key=settings.OPENAI_API_KEY,
            )
            scored = retriever.retrieve_with_scores(query=query.strip())
            docs = [
                {
                    "content": doc.page_content,
                    "score": round(float(score), 4),
                    "source": doc.metadata.get("source", doc.metadata.get("filename", "unknown")),
                    "doc_type": doc.metadata.get("doc_type", "general"),
                }
                for doc, score in scored[:top_k]
            ]
        except Exception as exc:  # noqa: BLE001
            # Graceful degradation: fall back to keyword scanning the raw
            # markdown KB files when the FAISS index or embedding stack is
            # unavailable.
            logger.warning("RAG retriever unavailable, falling back to file scan: %s", exc)
            docs = _keyword_fallback(query.strip(), top_k)

        await audit_log(
            action,
            {"query": query, "results_returned": len(docs)},
            success=True,
        )
        return {"success": True, "query": query, "results": docs, "count": len(docs)}
    except Exception as exc:  # noqa: BLE001
        return tool_error(action, exc)


def _keyword_fallback(query: str, top_k: int) -> list[dict[str, Any]]:
    """Simple keyword scoring over the markdown KB files in data/knowledge_base."""
    import re
    from pathlib import Path

    kb_dir = Path("data/knowledge_base")
    if not kb_dir.is_dir():
        return []

    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    scored: list[tuple[float, str, str]] = []
    for path in kb_dir.glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Split into paragraph chunks.
        for chunk in re.split(r"\n\s*\n", text):
            chunk = chunk.strip()
            if len(chunk) < 40:
                continue
            lowered = chunk.lower()
            score = sum(lowered.count(t) for t in terms) / max(len(terms), 1)
            if score > 0:
                scored.append((score, chunk, path.name))

    scored.sort(reverse=True, key=lambda s: s[0])
    total = sum(s for s, _, _ in scored) or 1.0
    return [
        {
            "content": chunk[:1200],
            "score": round(score / total, 4),
            "source": filename,
            "doc_type": "policy",
        }
        for score, chunk, filename in scored[:top_k]
    ]
