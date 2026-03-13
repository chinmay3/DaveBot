from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI

from daverag.config import Settings
from daverag.schemas import Citation, SearchResult


SYSTEM_PROMPT = (
    "You are a question-answering assistant for Dave's Hot Chicken. "
    "Answer only from the provided retrieved context. "
    "If the context is insufficient, say you do not have enough information. "
    "Do not invent menu items, dates, ingredients, or policies. "
    "Return a concise grounded answer."
)


class GenerationBackend(ABC):
    @abstractmethod
    def answer(self, question: str, results: list[SearchResult]) -> str:
        raise NotImplementedError


class ExtractiveGenerationBackend(GenerationBackend):
    def answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "I do not have enough verified information in the dataset to answer that."
        top = results[0].document
        return top.answer


class OpenAIGenerationBackend(GenerationBackend):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI generation backend.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model

    def answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "I do not have enough verified information in the dataset to answer that."
        context_blocks = []
        for result in results:
            doc = result.document
            context_blocks.append(
                f"[{doc.source_id}] Title: {doc.title}\nQuestion: {doc.question}\nAnswer: {doc.answer}"
            )
        prompt = (
            f"User question:\n{question}\n\n"
            f"Retrieved context:\n" + "\n\n".join(context_blocks) + "\n\n"
            "Instructions:\n"
            "Answer in 2-4 sentences. Use only the retrieved context. Do not include source_ids or citations in the prose answer."
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.output_text.strip()


def build_generation_backend(settings: Settings) -> GenerationBackend:
    if settings.generation_backend == "openai":
        return OpenAIGenerationBackend(settings)
    return ExtractiveGenerationBackend()


def citations_from_results(results: list[SearchResult]) -> list[Citation]:
    return [
        Citation(
            source_id=result.document.source_id,
            title=result.document.title,
            topic=result.document.topic,
        )
        for result in results
    ]
