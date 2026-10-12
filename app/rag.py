import asyncio
import functools
import logging
import re
from pathlib import Path
from typing import Optional

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.utils import EmbeddingFunc

from app import config, ingest

logger = logging.getLogger(__name__)

_REFERENCES_SECTION = re.compile(
    r"\n+#{1,6}\s*References\s*\n([\s\S]*)$", re.IGNORECASE
)
# A run of inline citations such as "[1]", "[2][5]" or "[1], [3]".
_INLINE_CITATIONS = re.compile(r"\[\d+\](?:\s*,?\s*\[\d+\])*")

# Appended to LightRAG's answer prompt so inline citations are bare reference
# numbers, which ask() then rewrites as page numbers.
CITATION_INSTRUCTIONS = (
    "Cite supporting facts inline using only the bracketed reference number, "
    "for example [1] or [2][3]. Do not write the words 'Document Chunk' or "
    "'Reference' in the answer text."
)

_rag: Optional[LightRAG] = None
_rag_lock = asyncio.Lock()


async def get_rag() -> LightRAG:
    """Create and initialize the LightRAG instance once per process."""
    global _rag
    async with _rag_lock:
        if _rag is None:
            Path(config.WORKING_DIR).mkdir(parents=True, exist_ok=True)
            rag = LightRAG(
                working_dir=config.WORKING_DIR,
                llm_model_func=openai_complete,
                llm_model_name=config.LLM_MODEL,
                llm_model_max_async=config.LLM_MAX_ASYNC,
                embedding_func=EmbeddingFunc(
                    embedding_dim=config.EMBEDDING_DIM,
                    func=functools.partial(
                        openai_embed.func, model=config.EMBEDDING_MODEL
                    ),
                ),
                chunk_token_size=config.CHUNK_TOKEN_SIZE,
                chunk_overlap_token_size=config.CHUNK_OVERLAP_TOKEN_SIZE,
            )
            await rag.initialize_storages()
            await initialize_pipeline_status()
            _rag = rag
    return _rag


async def close_rag() -> None:
    global _rag
    if _rag is not None:
        await _rag.finalize_storages()
        _rag = None


async def insert_pdf(
    pdf_path: Path, track_id: Optional[str] = None
) -> tuple[str, int]:
    """Queue every page of a PDF for entity/relationship extraction.

    Returns (track_id, page_count). Pages already ingested (same file name and
    page number) are skipped by LightRAG, so re-running is safe.
    """
    rag = await get_rag()
    pages = ingest.load_pdf_pages(pdf_path)
    if not pages:
        raise ValueError("No extractable text found in PDF")
    filename = pdf_path.name
    track_id = await rag.ainsert(
        [text for _, text in pages],
        ids=[f"{filename}-p{page}" for page, _ in pages],
        file_paths=[ingest.page_source(filename, page) for page, _ in pages],
        track_id=track_id,
    )
    return track_id, len(pages)


async def ingest_status(track_id: str) -> dict[str, int]:
    """Count the documents of one ingest run by status (processed, failed, ...).

    Pages that were already in the graph are recorded by LightRAG as "failed"
    duplicates; they're counted as "duplicate" instead so callers can tell
    them apart from real failures.
    """
    rag = await get_rag()
    docs = await rag.aget_docs_by_track_id(track_id)
    counts: dict[str, int] = {}
    for doc in docs.values():
        status = getattr(doc.status, "value", str(doc.status))
        if (doc.metadata or {}).get("is_duplicate"):
            status = "duplicate"
        counts[status] = counts.get(status, 0) + 1
    return counts


async def ask(
    question: str,
    history: Optional[list[dict]] = None,
    mode: Optional[str] = None,
) -> tuple[str, list[dict]]:
    """Answer a question from the knowledge graph.

    Returns (answer, sources), where sources are {"source", "page"} dicts in
    the order LightRAG ranked the chunks it used as context.
    """
    rag = await get_rag()
    param = QueryParam(
        mode=mode or config.QUERY_MODE,
        conversation_history=(history or [])[-config.HISTORY_TURNS :],
        include_references=True,
        user_prompt=CITATION_INSTRUCTIONS,
        # No reranker model is configured for LightRAG.
        enable_rerank=False,
    )
    result = await rag.aquery_llm(question, param=param)

    answer = (result.get("llm_response") or {}).get("content") or ""
    data = result.get("data") or {}

    # reference_id -> (file name, page) for every chunk LightRAG put in context.
    ref_pages: dict[str, tuple[str, Optional[int]]] = {
        str(ref.get("reference_id")): ingest.parse_page_source(
            str(ref.get("file_path", ""))
        )
        for ref in data.get("references") or []
    }

    # LightRAG's prompt makes the LLM end with a "### References" list of the
    # (up to 5) references it actually cited. Use it to pick the sources to
    # report, then drop it from the answer text.
    cited: list[str] = []
    match = _REFERENCES_SECTION.search(answer)
    if match:
        cited = re.findall(r"\[(\d+)\]", match.group(1))
        answer = answer[: match.start()].rstrip()
    if not cited:
        cited = re.findall(r"\[(\d+)\]", answer)

    # Rewrite inline citations as pages: "[1][3]" -> "(p. 9, p. 18)".
    def to_pages(m: re.Match) -> str:
        labels: list[str] = []
        for ref_id in re.findall(r"\d+", m.group(0)):
            _, page = ref_pages.get(ref_id, ("", None))
            label = f"p. {page}" if page is not None else None
            if label and label not in labels:
                labels.append(label)
        return f"({', '.join(labels)})" if labels else m.group(0)

    answer = _INLINE_CITATIONS.sub(to_pages, answer)
    # The model often writes "products[1]"; keep a space before "(p. ...)".
    answer = re.sub(r"(?<=[^\s(])\((?=p\. \d)", " (", answer)

    # Cited references first; fall back to everything in context if the
    # answer cited nothing.
    order = [r for r in dict.fromkeys(cited) if r in ref_pages] or list(ref_pages)
    sources: list[dict] = []
    seen: set[tuple[str, Optional[int]]] = set()
    for ref_id in order:
        name, page = ref_pages[ref_id]
        if (name, page) not in seen:
            seen.add((name, page))
            sources.append({"source": name, "page": page})
    return answer, sources
