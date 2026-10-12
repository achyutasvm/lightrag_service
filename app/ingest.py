from pathlib import Path

from pypdf import PdfReader


def load_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (1-indexed page number, text) for each page that has text.

    Each page becomes its own LightRAG document so answers can cite pages:
    LightRAG tracks sources per document (file_path), not per character
    offset, so page-level documents are what make page citations possible.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def page_source(filename: str, page: int) -> str:
    """The file_path recorded in LightRAG for one page, e.g. "report.pdf#page=3"."""
    return f"{filename}#page={page}"


def parse_page_source(file_path: str) -> tuple[str, int | None]:
    """Inverse of page_source; tolerates paths without a page suffix."""
    name, sep, page = file_path.rpartition("#page=")
    if sep and page.isdigit():
        return name, int(page)
    return file_path, None
