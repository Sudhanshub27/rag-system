#!/usr/bin/env python3
"""
Ask My Documents RAG — Universal One-Click Launcher Script
Usage:
    python run.py             # Launch Streamlit Web UI
    python run.py app         # Launch Streamlit Web UI
    python run.py query "..." # Query via CLI
    python run.py test        # Run unit tests
    python run.py eval        # Run evaluation benchmark
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(PROJECT_ROOT)


def main():
    args = sys.argv[1:]
    mode = args[0].lower() if args else "app"

    if mode in ("app", "ui", "web"):
        print("🚀 Starting Streamlit Web UI...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

    elif mode in ("query", "cli"):
        query_text = (
            " ".join(args[1:])
            if len(args) > 1
            else "What is Retrieval-Augmented Generation?"
        )
        print(f"🔍 Querying: '{query_text}'")
        subprocess.run([sys.executable, "cli.py", "query", query_text])

    elif mode == "ingest":
        if len(args) < 2:
            print("Usage: python run.py ingest <file_or_directory_path>")
            sys.exit(1)
        subprocess.run([sys.executable, "cli.py", "ingest", args[1]])

    elif mode in ("test", "tests"):
        print("🧪 Running PyTest test suite...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=chunking",
                "--cov=retrieval",
                "--cov=generation",
                "--cov-report=term-missing",
            ]
        )

    elif mode in ("eval", "evaluation"):
        print("📊 Running Quality Evaluation benchmark...")
        subprocess.run([sys.executable, "evaluation/evaluate.py"])

    else:
        print("Ask My Documents RAG — One-Click Launcher\n")
        print("Usage:")
        print("  python run.py            # Launch Web UI (default)")
        print('  python run.py query "?" # Run CLI query')
        print("  python run.py ingest path# Ingest document or directory")
        print("  python run.py test       # Run unit test suite")
        print("  python run.py eval       # Run quality evaluation")


if __name__ == "__main__":
    main()
