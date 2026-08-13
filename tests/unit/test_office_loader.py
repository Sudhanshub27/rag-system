"""
Unit tests for Word, Excel, CSV, JSON, and HTML document loaders.
"""

import json
from pathlib import Path

from ingestion.office_loader import (
    CsvLoader,
    DocxLoader,
    ExcelLoader,
    HtmlLoader,
    JsonLoader,
)


def test_csv_loader(tmp_path: Path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "Name,Age,Role\nAlice,30,Developer\nBob,35,Designer", encoding="utf-8"
    )

    loader = CsvLoader()
    docs = loader.load(str(csv_file))

    assert len(docs) == 1
    assert "Alice" in docs[0].content
    assert "Developer" in docs[0].content
    assert docs[0].metadata["file_type"] == "csv"


def test_json_loader(tmp_path: Path):
    json_file = tmp_path / "data.json"
    json_data = {"project": "RAG System", "version": "2.0.0", "status": "Active"}
    json_file.write_text(json.dumps(json_data), encoding="utf-8")

    loader = JsonLoader()
    docs = loader.load(str(json_file))

    assert len(docs) == 1
    assert "RAG System" in docs[0].content
    assert docs[0].metadata["file_type"] == "json"


def test_html_loader(tmp_path: Path):
    html_file = tmp_path / "page.html"
    html_content = (
        "<html><body><h1>Title</h1><p>This is test content.</p></body></html>"
    )
    html_file.write_text(html_content, encoding="utf-8")

    loader = HtmlLoader()
    docs = loader.load(str(html_file))

    assert len(docs) == 1
    assert "Title" in docs[0].content
    assert "This is test content." in docs[0].content
    assert docs[0].metadata["file_type"] == "html"


def test_docx_loader_fallback(tmp_path: Path):
    docx_file = tmp_path / "sample.docx"
    docx_file.write_text(
        "Sample plain text fallback content for word doc", encoding="utf-8"
    )

    loader = DocxLoader()
    docs = loader.load(str(docx_file))

    assert len(docs) == 1
    assert "Sample plain text" in docs[0].content


def test_excel_loader_fallback(tmp_path: Path):
    excel_file = tmp_path / "sample.xlsx"
    excel_file.write_text("Column1,Column2\nVal1,Val2", encoding="utf-8")

    loader = ExcelLoader()
    # ExcelLoader handles non-binary text gracefully or attempts openpyxl
    try:
        docs = loader.load(str(excel_file))
        assert isinstance(docs, list)
    except Exception as e:
        assert "Excel loading failed" in str(e)
