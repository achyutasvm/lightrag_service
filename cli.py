"""Command-line access to the LightRAG knowledge graph.

    python cli.py ingest [file.pdf ...]   # default: every PDF in docs/
    python cli.py ask "question" [--mode hybrid]
    python cli.py status
"""

import argparse
import asyncio
from pathlib import Path

from app import config, rag


async def cmd_ingest(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in args.files] or sorted(Path(config.DOCS_DIR).glob("*.pdf"))
    if not paths:
        print(f"No PDFs given and none found in {config.DOCS_DIR}/")
        return
    for path in paths:
        print(f"Ingesting {path} (entity/relationship extraction with {config.LLM_MODEL})...")
        track_id, pages = await rag.insert_pdf(path)
        print(f"  {pages} page(s): {await rag.ingest_status(track_id)}")


async def cmd_ask(args: argparse.Namespace) -> None:
    answer, sources = await rag.ask(args.question, mode=args.mode)
    print(answer)
    print("\nSources:")
    for s in sources:
        print(f" - {s['source']} p. {s['page']}")


async def cmd_status(_: argparse.Namespace) -> None:
    r = await rag.get_rag()
    print(r.get_processing_status() if not asyncio.iscoroutinefunction(r.get_processing_status) else await r.get_processing_status())


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="extract entities/relationships from PDFs into the graph")
    p_ingest.add_argument("files", nargs="*")
    p_ask = sub.add_parser("ask", help="answer a question from the graph")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=config.QUERY_MODES, default=config.QUERY_MODE)
    sub.add_parser("status", help="document counts by processing status")
    args = parser.parse_args()
    try:
        await {"ingest": cmd_ingest, "ask": cmd_ask, "status": cmd_status}[args.command](args)
    finally:
        await rag.close_rag()


if __name__ == "__main__":
    asyncio.run(main())
