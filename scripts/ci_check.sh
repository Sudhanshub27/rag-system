#!/usr/bin/env bash
set -e

echo "=========================================="
echo "🔍 Running Automated CI Local Pre-Push Checks"
echo "=========================================="

# Find Python / Virtualenv
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
    RUFF=".venv/bin/ruff"
    BLACK=".venv/bin/black"
    PYTEST=".venv/bin/pytest"
else
    PYTHON="python"
    RUFF="ruff"
    BLACK="black"
    PYTEST="pytest"
fi

echo "1️⃣  Running Ruff Linting..."
$RUFF check .

echo "2️⃣  Running Black Formatting Check..."
$BLACK --check .

echo "3️⃣  Running PyTest Suite..."
$PYTEST -m "not integration" --cov=chunking --cov=retrieval --cov=generation --cov-report=term-missing

echo "=========================================="
echo "✅ All CI Checks Passed! Safe to push."
echo "=========================================="
