import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import config, ingest, rag

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

logger = logging.getLogger(__name__)

# Background ingest tasks, kept referenced so they aren't garbage-collected.
_ingest_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await rag.get_rag()
    yield
    await rag.close_rag()


app = FastAPI(title="LightRAG Knowledge Graph API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    # LightRAG query mode; defaults to QUERY_MODE (hybrid).
    mode: Optional[Literal["naive", "local", "global", "hybrid", "mix"]] = None


class Source(BaseModel):
    source: str
    page: Optional[int] = None


class ChatResponse(BaseModel):
    text: str
    sources: list[Source]
    mode: str


class IngestResponse(BaseModel):
    filename: str
    pages: int
    track_id: str
    # Building the knowledge graph takes minutes (an LLM call per chunk), so
    # ingestion runs in the background; poll GET /ingest/{track_id}.
    status: Literal["processing"]


class IngestStatusResponse(BaseModel):
    track_id: str
    counts: dict[str, int]
    done: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    docs_dir = Path(config.DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / filename

    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (max 25MB)")
            out.write(chunk)

    try:
        page_count = len(ingest.load_pdf_pages(dest))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc
    if page_count == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    # ainsert() only returns once every page has been through entity
    # extraction, so run it in the background under a track id we choose.
    track_id = f"upload-{uuid.uuid4().hex[:12]}"

    async def run() -> None:
        try:
            await rag.insert_pdf(dest, track_id=track_id)
        except Exception:
            logger.exception("Ingest failed for %s", filename)

    task = asyncio.create_task(run())
    _ingest_tasks.add(task)
    task.add_done_callback(_ingest_tasks.discard)

    return IngestResponse(
        filename=filename, pages=page_count, track_id=track_id, status="processing"
    )


@app.get("/ingest/{track_id}", response_model=IngestStatusResponse)
async def ingest_progress(track_id: str) -> IngestStatusResponse:
    counts = await rag.ingest_status(track_id)
    if not counts:
        raise HTTPException(status_code=404, detail="Unknown track id")
    pending = sum(
        n for s, n in counts.items() if s not in ("processed", "failed", "duplicate")
    )
    return IngestStatusResponse(track_id=track_id, counts=counts, done=pending == 0)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    question = user_messages[-1].content
    # Earlier turns (everything before the last user message) give LightRAG
    # context for follow-up questions.
    last_index = max(i for i, m in enumerate(request.messages) if m.role == "user")
    history = [m.model_dump() for m in request.messages[:last_index]]

    mode = request.mode or config.QUERY_MODE
    answer, sources = await rag.ask(question, history=history, mode=mode)
    return ChatResponse(text=answer, sources=[Source(**s) for s in sources], mode=mode)
