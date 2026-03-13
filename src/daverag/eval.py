from __future__ import annotations

import json
from pathlib import Path

from daverag.schemas import EvalCase, EvalResult
from daverag.service import RagService


def load_eval_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text())
    return [EvalCase.model_validate(item) for item in data]


def run_eval(service: RagService, cases: list[EvalCase]) -> tuple[list[EvalResult], dict[str, float]]:
    results: list[EvalResult] = []
    for case in cases:
        response = service.ask(case.question)
        top_source_id = response.sources[0].source_id if response.sources else None
        answer_lower = response.answer.lower()
        expected_terms_hit = all(term.lower() in answer_lower for term in case.expected_answer_contains)
        results.append(
            EvalResult(
                question=case.question,
                answer=response.answer,
                top_source_id=top_source_id,
                hit_expected_source=(case.expected_source_id == top_source_id),
                answer_contains_expected_terms=expected_terms_hit,
            )
        )
    total = max(len(results), 1)
    metrics = {
        "retrieval_hit_rate": sum(item.hit_expected_source for item in results) / total,
        "answer_term_hit_rate": sum(item.answer_contains_expected_terms for item in results) / total,
    }
    return results, metrics
