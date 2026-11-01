"""CLI Tool to Ingest Support Documentation into Vector Store."""

import argparse
import asyncio
import logging
from rag.ingest import DocumentIngestionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_ingest")


async def main(docs_path: str, chunk_size: int, chunk_overlap: int) -> None:
    """Runs document ingestion and indexing pipeline.
    
    Args:
        docs_path: Directory path containing documentation files (.md, .pdf, .txt).
        chunk_size: Token/character size for each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    TODO:
        - Integrate rich progress bar.
        - Support incremental sync (re-index only modified files using MD5 hashes).
    """
    logger.info(f"Starting ingestion from: {docs_path}")
    pipeline = DocumentIngestionPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    stats = await pipeline.run(docs_path)
    logger.info(f"Ingestion completed: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest support docs into vectorstore")
    parser.add_argument("--docs-dir", type=str, default="./docs", help="Path to knowledge base documents")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Document chunk size")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap")
    args = parser.parse_args()

    asyncio.run(main(args.docs_dir, args.chunk_size, args.overlap))

