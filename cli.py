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
        description="Production Multi-Tenant RAG System CLI",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--tenant-id",
        "-t",
        "--user-id",
        "-u",
        default="cli_user",
        help="Tenant ID for multi-tenant data isolation (default: cli_user)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a single document")
    ingest_parser.add_argument("source", help="Path to the document")

    # ingest-dir
    ingest_dir_parser = subparsers.add_parser(
        "ingest-dir", help="Ingest all docs in a directory"
    )
    ingest_dir_parser.add_argument("directory", help="Path to the directory")
    ingest_dir_parser.add_argument(
        "--no-recursive", action="store_true", help="Do not recurse subdirectories"
    )

    # query
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Your question in quotes")
    query_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # stats
    subparsers.add_parser(
        "stats",
        aliases=["list-docs"],
        help="Show knowledge base statistics and list tenant documents",
    )

    # delete single document
    delete_parser = subparsers.add_parser(
        "delete", help="Delete a document by source name"
    )
    delete_parser.add_argument("source", help="Filename as stored (e.g. report.pdf)")

    # delete all tenant data
    subparsers.add_parser(
        "delete-all",
        aliases=["delete-my-data", "delete-tenant-data"],
        help="Delete all data for this tenant (drops ChromaDB collection entirely)",
    )

    args = parser.parse_args()

    setup_logger(debug=args.debug)
    tenant_id = args.tenant_id
    pipeline = RAGPipeline(tenant_id=tenant_id, debug=args.debug)

    if args.command == "ingest":
        n = pipeline.ingest(args.source)
        print(f"✅ Ingested {n} chunks from {args.source} (Tenant: {tenant_id})")

    elif args.command == "ingest-dir":
        n = pipeline.ingest_directory(args.directory, recursive=not args.no_recursive)
        print(f"✅ Ingested {n} total chunks from {args.directory} (Tenant: {tenant_id})")

    elif args.command == "query":
        response = pipeline.query(args.question)
        if args.json:
            out = {
                "tenant_id": tenant_id,
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

    elif args.command in ("stats", "list-docs"):
        stats = pipeline.get_stats()
        chunks = pipeline.get_all_chunks()
        doc_map = {}
        for c in chunks:
            src = c.source
            doc_map[src] = doc_map.get(src, 0) + 1

        print(f"\nKnowledge Base Statistics (Tenant: {tenant_id})")
        print("-" * 50)
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print(f"\nDocuments ({len(doc_map)} total):")
        if not doc_map:
            print("  (No documents ingested for this tenant)")
        else:
            for src, count in sorted(doc_map.items()):
                print(f"  - {src} ({count} chunk(s))")

    elif args.command == "delete":
        deleted = pipeline.delete_document(args.source)
        print(f"✅ Deleted {deleted} chunks for '{args.source}' (Tenant: {tenant_id})")

    elif args.command in ("delete-all", "delete-my-data", "delete-tenant-data"):
        pipeline.delete_all_tenant_data()
        print(f"🗑️ Successfully deleted all data for tenant '{tenant_id}' (collection dropped)")


if __name__ == "__main__":
    main()

