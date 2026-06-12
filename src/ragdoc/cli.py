"""Command-line interface.

    ragdoc ingest docs/ --index .ragdoc_index
    ragdoc ask "What is the refund policy?" --index .ragdoc_index
    ragdoc chat --index .ragdoc_index
"""

from __future__ import annotations

import argparse
import sys

from ragdoc.config import Settings
from ragdoc.pipeline import RagPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragdoc", description="Document Q&A over your files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index documents (txt, md, pdf).")
    p_ingest.add_argument("paths", nargs="+", help="Files or directories to index.")
    p_ingest.add_argument("--index", default=None, help="Index directory to write.")

    p_ask = sub.add_parser("ask", help="Ask a single question.")
    p_ask.add_argument("question")
    p_ask.add_argument("--index", default=None, help="Index directory to read.")
    p_ask.add_argument("--show-context", action="store_true", help="Print retrieved chunks.")

    p_chat = sub.add_parser("chat", help="Interactive Q&A loop.")
    p_chat.add_argument("--index", default=None, help="Index directory to read.")

    return parser


def _print_answer(answer, show_context: bool) -> None:
    print(f"\n{answer.text}\n")
    print(f"[backend: {answer.backend}]")
    for i, result in enumerate(answer.sources, start=1):
        found_by = "+".join(result.found_by)
        print(f"  [{i}] {result.chunk.source} (chunk {result.chunk.chunk_id}, via {found_by})")
        if show_context:
            print(f"      {result.chunk.text[:300]}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = Settings()
    pipeline = RagPipeline(settings=settings)

    if args.command == "ingest":
        count = pipeline.ingest(args.paths)
        directory = pipeline.save_index(args.index)
        print(f"Indexed {count} chunks -> {directory}")
        return 0

    pipeline.load_index(args.index)

    if args.command == "ask":
        _print_answer(pipeline.ask(args.question), args.show_context)
        return 0

    if args.command == "chat":
        print(f"Loaded {len(pipeline.retriever)} chunks. Ctrl-D or 'exit' to quit.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in {"exit", "quit"}:
                return 0
            if question:
                _print_answer(pipeline.ask(question), show_context=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
