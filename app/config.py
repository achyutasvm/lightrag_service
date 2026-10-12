import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# LightRAG's OpenAI helpers read OPENAI_API_KEY from the environment directly;
# this just fails fast with a clear message when it's missing.
OPENAI_API_KEY = _require("OPENAI_API_KEY")

# LLM used to extract entities/relationships during ingestion and to write
# answers at query time.
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))

# LightRAG stores its knowledge graph (NetworkX, saved as GraphML), vector
# indexes (nano-vectordb), document status, and LLM cache as files here.
WORKING_DIR = os.environ.get("WORKING_DIR", "rag_storage")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs")

# naive = plain vector search over chunks; local = entities and their
# neighbourhood; global = relationships/themes; hybrid = local + global;
# mix = hybrid + naive.
QUERY_MODES = ("naive", "local", "global", "hybrid", "mix")
QUERY_MODE = os.environ.get("QUERY_MODE", "hybrid")
if QUERY_MODE not in QUERY_MODES:
    raise RuntimeError(f"QUERY_MODE must be one of {QUERY_MODES}")

# How many earlier messages to pass as conversation history.
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "6"))

CHUNK_TOKEN_SIZE = int(os.environ.get("CHUNK_TOKEN_SIZE", "1200"))
CHUNK_OVERLAP_TOKEN_SIZE = int(os.environ.get("CHUNK_OVERLAP_TOKEN_SIZE", "100"))
LLM_MAX_ASYNC = int(os.environ.get("LLM_MAX_ASYNC", "8"))

# Comma-separated origins allowed to call the FastAPI service (e.g. the
# Next.js chat app's dev/prod URLs).
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
