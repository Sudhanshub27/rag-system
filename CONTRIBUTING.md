# Contributing Guidelines

Thank you for contributing to the RAG system! Please follow these guidelines before submitting a pull request.

## Local Testing & Verification

1. Install dependencies (runtime + dev):
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```

2. Run code formatting and lint checks:
   ```bash
   ruff check .
   black --check .
   ```

3. Run the fast unit test suite with coverage:
   ```bash
   pytest -m "not integration" --cov=chunking --cov=retrieval --cov=generation
   ```

4. Run the full test suite (including integration tests):
   ```bash
   pytest
   ```

Ensure all tests pass and coverage remains above 70% before opening a PR.
