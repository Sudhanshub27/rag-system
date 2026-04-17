"""
RAG System — Quick Start Example
Demonstrates the full pipeline:
  1. Ingest a sample document
  2. Ask questions with citations
  3. Show pipeline stats
"""

import sys
from pathlib import Path

# Ensure project root is importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import RAGPipeline


def main():
    print("🚀 Initializing RAG Pipeline…")
    pipeline = RAGPipeline(debug=False)

    # Step 1: Ingest sample document
    doc_path = Path(__file__).parent.parent / "docs" / "sample_doc.txt"
    print(f"\n📄 Ingesting: {doc_path.name}")
    n_chunks = pipeline.ingest(str(doc_path))
    print(f"   ✅ {n_chunks} chunks indexed")

    # Step 2: Query the pipeline
    questions = [
        "What is Retrieval-Augmented Generation?",
        "How does chunking work and what is the recommended overlap?",
        "What is Reciprocal Rank Fusion?",
        "What metrics are used to evaluate RAG systems?",
        "What is the capital of France?",   # Should trigger fallback
    ]

    for question in questions:
        print(f"\n{'='*70}")
        print(f"❓ {question}")
        print("-" * 70)

        response = pipeline.query(question)

        if response.is_fallback:
            print(f"⚠️  {response.answer}")
        else:
            print(response.answer[:600] + ("…" if len(response.answer) > 600 else ""))

        if response.citations:
            print("\n📌 Citations:")
            for cit in response.citations:
                print(f"   {cit}")

    # Step 3: Stats
    print(f"\n{'='*70}")
    print("📊 Knowledge Base Stats")
    print("-" * 70)
    for k, v in pipeline.get_stats().items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
