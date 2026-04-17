"""
CLI Interface for the RAG System
Provides a simple command-line way to ingest documents and query the pipeline.

Usage:
    python cli.py ingest path/to/document.pdf
    python cli.py ingest-dir path/to/docs/
    python cli.py query "What is the return policy?"
    python cli.py stats
    python cli.py delete filename.pdf
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import RAGPipeline
from utils.logger import setup_logger


def print_response(response) -> None:
    """Pretty-print a RAGResponse to stdout."""
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(response.answer)

    if response.citations:
        print("\n" + "-" * 70)
        print("CITATIONS")
        print("-" * 70)
        for cit in response.citations:
            print(f"  {cit}")

    if response.retrieved_chunks:
        print("\n" + "-" * 70)
        print(f"Retrieved {len(response.retrieved_chunks)} chunk(s)")
        for rc in response.retrieved_chunks:
            print(
                f"  [{rc.rank}] {rc.chunk.source} p.{rc.chunk.page} "
                f"(score={rc.score:.4f})"
            )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Production RAG System CLI",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a single document")
    ingest_parser.add_argument("source", help="Path to the document")

    # ingest-dir
    ingest_dir_parser = subparsers.add_parser("ingest-dir", help="Ingest all docs in a directory")
    ingest_dir_parser.add_argument("directory", help="Path to the directory")
    ingest_dir_parser.add_argument(
        "--no-recursive", action="store_true", help="Do not recurse subdirectories"
    )

    # query
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Your question in quotes")
    query_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # stats
    subparsers.add_parser("stats", help="Show knowledge base statistics")

    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete a document by source name")
    delete_parser.add_argument("source", help="Filename as stored (e.g. report.pdf)")

    args = parser.parse_args()

    setup_logger(debug=args.debug)
    pipeline = RAGPipeline(debug=args.debug)

    if args.command == "ingest":
        n = pipeline.ingest(args.source)
        print(f"✅ Ingested {n} chunks from {args.source}")

    elif args.command == "ingest-dir":
        n = pipeline.ingest_directory(
            args.directory, recursive=not args.no_recursive
        )
        print(f"✅ Ingested {n} total chunks from {args.directory}")

    elif args.command == "query":
        response = pipeline.query(args.question)
        if args.json:
            out = {
                "answer": response.answer,
                "citations": response.citations,
                "is_fallback": response.is_fallback,
                "retrieved_chunks": [
                    {
                        "source": rc.chunk.source,
                        "page": rc.chunk.page,
                        "score": rc.score,
                        "text": rc.chunk.text[:200],
                    }
                    for rc in response.retrieved_chunks
                ],
            }
            print(json.dumps(out, indent=2))
        else:
            print_response(response)

    elif args.command == "stats":
        stats = pipeline.get_stats()
        print("\nKnowledge Base Statistics")
        print("-" * 30)
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif args.command == "delete":
        deleted = pipeline.delete_document(args.source)
        print(f"✅ Deleted {deleted} chunks for '{args.source}'")


if __name__ == "__main__":
    main()
