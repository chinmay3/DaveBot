from __future__ import annotations

import argparse
import json
from pathlib import Path

from daverag.config import settings
from daverag.eval import load_eval_cases, run_eval
from daverag.service import RagService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DaveRAG utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Build and persist the retrieval index")

    ask_parser = subparsers.add_parser("ask", help="Query the RAG service locally")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--topic")

    eval_parser = subparsers.add_parser("eval", help="Run offline evaluation cases")
    eval_parser.add_argument("--file", default="eval/sample_eval.json")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = RagService(settings)
    if args.command == "index":
        service.build_index(persist=True)
        print(f"Indexed {service.stats.documents if service.stats else 0} documents into {settings.index_dir}")
        return
    if args.command == "ask":
        response = service.ask(args.question, topic=args.topic)
        print(json.dumps(response.model_dump(mode="json"), indent=2))
        return
    if args.command == "eval":
        service.build_index(persist=False)
        cases = load_eval_cases(Path(args.file))
        results, metrics = run_eval(service, cases)
        print(json.dumps({"metrics": metrics, "results": [r.model_dump() for r in results]}, indent=2))


if __name__ == "__main__":
    main()
