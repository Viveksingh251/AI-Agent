"""Document Ingestion Pipeline for the AI Support Agent RAG system.

CLI usage::

    python -m rag.ingest --docs-dir ./data/knowledge_base
    python -m rag.ingest --docs-dir ./data/knowledge_base --reset
    python -m rag.ingest --docs-dir ./data/knowledge_base --chunk-size 800 --overlap 150

Supports PDF, Markdown (.md), plain text (.txt), and HTML (.html / .htm).
All heavy third-party imports are guarded so this file remains importable even
when LangChain / FAISS are absent in the environment.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("support_agent.rag.ingest")

# ---------------------------------------------------------------------------
# Optional LangChain imports
# ---------------------------------------------------------------------------

try:
    from langchain_core.documents import Document  # type: ignore

    _LC_CORE_AVAILABLE = True
except ImportError:
    _LC_CORE_AVAILABLE = False
    logger.warning("langchain_core not available — ingestion disabled")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

    _SPLITTER_AVAILABLE = True
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        _SPLITTER_AVAILABLE = True
    except ImportError:
        _SPLITTER_AVAILABLE = False

# Loaders — fall back to plain TextLoader if unstructured isn't present.
try:
    from langchain_community.document_loaders import (  # type: ignore
        TextLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
    )

    _UNSTRUCTURED_AVAILABLE = True
except ImportError:
    _UNSTRUCTURED_AVAILABLE = False

try:
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore

    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    from langchain_community.document_loaders import TextLoader  # type: ignore  # noqa: F811

    _TEXT_LOADER_AVAILABLE = True
except ImportError:
    _TEXT_LOADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maps from file extension to doc_type label inferred from the file name.
_EXT_TO_DOCTYPE: Dict[str, str] = {}

# Infer doc_type from the file stem (e.g. "shipping_policy" → "shipping").
_STEM_KEYWORDS: Dict[str, str] = {
    "shipping": "shipping",
    "return": "returns",
    "refund": "returns",
    "faq": "faq",
    "product": "product",
    "catalog": "product",
    "policy": "policy",
    "warranty": "warranty",
    "guide": "guide",
    "manual": "manual",
}


def _infer_doc_type(stem: str) -> str:
    """Return a doc_type label based on keywords in the file stem."""
    stem_lower = stem.lower()
    for keyword, doc_type in _STEM_KEYWORDS.items():
        if keyword in stem_lower:
            return doc_type
    return "general"


def _load_single_file(path: Path) -> List["Document"]:
    """Load a single file using the best available loader for its extension.

    Returns a list of ``Document`` objects (may be a single-element list).
    Falls back to reading the raw text if loaders are unavailable.
    """
    ext = path.suffix.lower()
    docs: List["Document"] = []

    try:
        if ext == ".pdf" and _PDF_AVAILABLE:
            loader = PyPDFLoader(str(path))  # type: ignore[name-defined]
            docs = loader.load()

        elif ext == ".md" and _UNSTRUCTURED_AVAILABLE:
            loader = UnstructuredMarkdownLoader(str(path), mode="elements")  # type: ignore[name-defined]
            docs = loader.load()

        elif ext in (".html", ".htm") and _UNSTRUCTURED_AVAILABLE:
            loader = UnstructuredHTMLLoader(str(path))  # type: ignore[name-defined]
            docs = loader.load()

        elif _TEXT_LOADER_AVAILABLE:
            loader = TextLoader(str(path), encoding="utf-8")  # type: ignore[name-defined]
            docs = loader.load()

        else:
            # Pure-stdlib fallback — read raw text.
            raw = path.read_text(encoding="utf-8", errors="replace")
            if _LC_CORE_AVAILABLE:
                docs = [Document(page_content=raw, metadata={})]
            else:
                raise RuntimeError("langchain_core.documents.Document not available.")

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s with preferred loader (%s) — trying raw read.", path, exc)
        raw = path.read_text(encoding="utf-8", errors="replace")
        if _LC_CORE_AVAILABLE:
            docs = [Document(page_content=raw, metadata={})]

    # Enrich every document with standardised metadata.
    doc_type = _infer_doc_type(path.stem)
    stat = path.stat()
    for doc in docs:
        doc.metadata.update(
            {
                "source": str(path),
                "filename": path.name,
                "file_type": ext.lstrip("."),
                "doc_type": doc_type,
                "last_modified": stat.st_mtime,
            }
        )
        # Preserve any YAML front-matter metadata extracted by Unstructured.
        # (Unstructured may parse YAML front-matter into the metadata dict.)

    return docs


# ---------------------------------------------------------------------------
# DocumentIngestionPipeline
# ---------------------------------------------------------------------------


class DocumentIngestionPipeline:
    """End-to-end pipeline: load → split → embed → index.

    Parameters
    ----------
    index_dir:
        Directory where the FAISS index is persisted.
    index_name:
        FAISS index name (folder within ``index_dir``).
    chunk_size:
        Target character count per text chunk.
    chunk_overlap:
        Overlap between consecutive chunks (for context continuity).
    openai_api_key:
        OpenAI API key for embeddings. Falls back to HuggingFace if absent.
    embedding_model:
        OpenAI embedding model name.
    hf_embedding_model:
        HuggingFace model name used as fallback.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".html", ".htm", ".rst", ".csv"}

    def __init__(
        self,
        index_dir: str = "./data/faiss_index",
        index_name: str = "support_kb",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        hf_embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.index_dir = index_dir
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.hf_embedding_model = hf_embedding_model

        # Lazily initialised.
        self._splitter: Optional[Any] = None
        self._vectorstore: Optional[Any] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_splitter(self) -> Any:
        if self._splitter is not None:
            return self._splitter
        if not _SPLITTER_AVAILABLE:
            raise RuntimeError(
                "RecursiveCharacterTextSplitter not available. "
                "Install: pip install langchain langchain-text-splitters"
            )
        self._splitter = RecursiveCharacterTextSplitter(  # type: ignore[name-defined]
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return self._splitter

    def _get_vectorstore(self) -> Any:
        """Return the lazily initialised :class:`FAISSVectorStore`."""
        if self._vectorstore is None:
            from rag.vectorstore import FAISSVectorStore

            self._vectorstore = FAISSVectorStore(
                index_dir=self.index_dir,
                index_name=self.index_name,
                openai_api_key=self.openai_api_key,
                embedding_model=self.embedding_model,
                hf_embedding_model=self.hf_embedding_model,
            )
        return self._vectorstore

    # ------------------------------------------------------------------
    # Public pipeline steps
    # ------------------------------------------------------------------

    def load_documents(self, source_dir: str) -> List["Document"]:
        """Recursively load all supported documents from *source_dir*.

        Skips hidden files and unsupported extensions.

        Returns:
            List of raw (un-split) ``Document`` objects.
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        all_docs: List["Document"] = []
        skipped = 0

        for file_path in sorted(source_path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue  # skip hidden files
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                skipped += 1
                logger.debug("Skipped unsupported file: %s", file_path.name)
                continue

            logger.info("Loading %s", file_path)
            try:
                docs = _load_single_file(file_path)
                all_docs.extend(docs)
                logger.debug("  → %d document(s) loaded", len(docs))
            except Exception as exc:  # noqa: BLE001
                logger.error("Error loading %s: %s", file_path, exc)

        logger.info(
            "Loaded %d document(s) from %d file(s) — %d file(s) skipped",
            len(all_docs),
            len(all_docs),
            skipped,
        )
        return all_docs

    def split_documents(self, documents: List["Document"]) -> List["Document"]:
        """Split *documents* into chunks using ``RecursiveCharacterTextSplitter``.

        Metadata from the original document is preserved in every chunk.

        Returns:
            List of chunked ``Document`` objects.
        """
        splitter = self._get_splitter()
        chunks = splitter.split_documents(documents)
        logger.info(
            "Split %d document(s) → %d chunk(s) (size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def embed_and_index(
        self,
        chunks: List["Document"],
        batch_size: int = 100,
    ) -> int:
        """Embed *chunks* and upsert into the FAISS vector store.

        Args:
            chunks: Pre-split document chunks to index.
            batch_size: Number of chunks per embedding batch.

        Returns:
            Number of chunks successfully indexed.
        """
        store = self._get_vectorstore()
        total = store.add_documents(chunks, batch_size=batch_size)
        logger.info("Indexed %d chunks into FAISS", total)
        return total

    def run(
        self,
        source_dir: str,
        reset: bool = False,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Orchestrate the full ingestion pipeline.

        Args:
            source_dir: Directory containing source documents.
            reset: If ``True``, delete the existing index before ingesting.
            batch_size: Embedding batch size.

        Returns:
            Stats dictionary with keys:
            ``source_dir``, ``documents_loaded``, ``chunks_created``,
            ``chunks_indexed``, ``elapsed_seconds``, ``status``.
        """
        t0 = time.monotonic()
        stats: Dict[str, Any] = {
            "source_dir": str(source_dir),
            "documents_loaded": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "elapsed_seconds": 0.0,
            "status": "error",
        }

        try:
            if reset:
                store = self._get_vectorstore()
                store.reset()
                logger.info("Index reset — starting fresh.")

            # Step 1: Load
            docs = self.load_documents(source_dir)
            stats["documents_loaded"] = len(docs)

            if not docs:
                logger.warning("No documents found in %s — nothing to index.", source_dir)
                stats["status"] = "empty"
                return stats

            # Step 2: Split
            chunks = self.split_documents(docs)
            stats["chunks_created"] = len(chunks)

            # Step 3: Embed & index
            indexed = self.embed_and_index(chunks, batch_size=batch_size)
            stats["chunks_indexed"] = indexed
            stats["status"] = "success"

        except Exception as exc:  # noqa: BLE001
            logger.error("Ingestion pipeline failed: %s", exc, exc_info=True)
            stats["error"] = str(exc)

        finally:
            stats["elapsed_seconds"] = round(time.monotonic() - t0, 2)

        return stats


# ---------------------------------------------------------------------------
# Factory from settings
# ---------------------------------------------------------------------------


def get_pipeline_from_settings(settings: Any) -> DocumentIngestionPipeline:
    """Build a :class:`DocumentIngestionPipeline` from app settings.

    Args:
        settings: ``app.config.Settings`` instance (or compatible object).
    """
    return DocumentIngestionPipeline(
        index_dir=getattr(settings, "FAISS_INDEX_DIR", "./data/faiss_index"),
        index_name=getattr(settings, "FAISS_INDEX_NAME", "support_kb"),
        chunk_size=getattr(settings, "RAG_CHUNK_SIZE", 1000),
        chunk_overlap=getattr(settings, "RAG_CHUNK_OVERLAP", 200),
        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
        embedding_model=getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        hf_embedding_model=getattr(settings, "HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.ingest",
        description="Ingest knowledge-base documents into the FAISS vector store.",
    )
    parser.add_argument(
        "--docs-dir",
        default="./data/knowledge_base",
        help="Directory containing source documents (default: ./data/knowledge_base)",
    )
    parser.add_argument(
        "--index-dir",
        default="./data/faiss_index",
        help="Directory to persist the FAISS index (default: ./data/faiss_index)",
    )
    parser.add_argument(
        "--index-name",
        default="support_kb",
        help="FAISS index name / subfolder (default: support_kb)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Character chunk size for text splitting (default: 1000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks per embedding batch (default: 100)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing index before ingesting",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


def main() -> None:
    """CLI entry point for the ingestion pipeline."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Try to load settings from environment; fall back to CLI args.
    try:
        from app.config import get_settings

        settings = get_settings()
        pipeline = get_pipeline_from_settings(settings)
        # Override with CLI args where explicitly provided.
        pipeline.index_dir = args.index_dir
        pipeline.index_name = args.index_name
        pipeline.chunk_size = args.chunk_size
        pipeline.chunk_overlap = args.overlap
    except Exception:  # noqa: BLE001
        pipeline = DocumentIngestionPipeline(
            index_dir=args.index_dir,
            index_name=args.index_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )

    print(f"\n🚀 Starting ingestion from: {args.docs_dir}")
    print(f"   Index: {args.index_dir}/{args.index_name}")
    print(f"   Chunk size: {args.chunk_size}, overlap: {args.overlap}")
    if args.reset:
        print("   ⚠️  --reset: existing index will be deleted!\n")

    stats = pipeline.run(
        source_dir=args.docs_dir,
        reset=args.reset,
        batch_size=args.batch_size,
    )

    print("\n✅ Ingestion complete!")
    print(f"   Documents loaded : {stats['documents_loaded']}")
    print(f"   Chunks created   : {stats['chunks_created']}")
    print(f"   Chunks indexed   : {stats['chunks_indexed']}")
    print(f"   Elapsed          : {stats['elapsed_seconds']}s")
    print(f"   Status           : {stats['status']}")
    if "error" in stats:
        print(f"   Error            : {stats['error']}")


if __name__ == "__main__":
    main()
