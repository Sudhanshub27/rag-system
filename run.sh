#!/usr/bin/env bash
# ============================================================
# Ask My Documents RAG — One-Click Launcher Script
# ============================================================

set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️ Virtual environment (.venv) not found. Using system Python."
fi

# 2. Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "📄 .env file not found. Creating .env from .env.example..."
        cp .env.example .env
        echo "⚠️ Please update .env with your API key if needed."
    fi
fi

# 3. Parse command or default to Streamlit Web UI
MODE="${1:-app}"

case "$MODE" in
    app|ui|web)
        echo "🚀 Starting Streamlit Web UI..."
        streamlit run app.py
        ;;
    cli|query)
        shift
        if [ $# -eq 0 ]; then
            python cli.py query "What is Retrieval-Augmented Generation?"
        else
            python cli.py query "$@"
        fi
        ;;
    ingest)
        shift
        python cli.py ingest "$@"
        ;;
    test|tests)
        echo "🧪 Running PyTest test suite..."
        pytest --cov=chunking --cov=retrieval --cov=generation --cov-report=term-missing
        ;;
    eval|evaluation)
        echo "📊 Running Quality Evaluation..."
        python evaluation/evaluate.py
        ;;
    *)
        echo "Usage: ./run.sh [app|query|ingest|test|eval]"
        echo ""
        echo "  ./run.sh            Launch Streamlit Web UI (default)"
        echo "  ./run.sh query \"?\"  Query the RAG pipeline via CLI"
        echo "  ./run.sh ingest file Ingest a document or directory"
        echo "  ./run.sh test       Run the unit test suite with coverage"
        echo "  ./run.sh eval       Run the RAGAS evaluation benchmark"
        ;;
esac
