from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class RawMetadata(BaseModel):
    topic: str
    type: str


class RawRecord(BaseModel):
    source_id: str
    title: str
    text: str
    metadata: RawMetadata


class NormalizedDocument(BaseModel):
    id: str
    source_id: str
    title: str
    question: str
    answer: str
    topic: str
    document_type: str
    text_for_embedding: str
    text_for_generation: str
    normalized_question: str
    normalized_answer: str
    keywords: list[str]
    version: int = 1
    is_active: bool = True
    date_added: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SearchResult(BaseModel):
    document: NormalizedDocument
    vector_score: float
    keyword_score: float
    rerank_score: float
    combined_score: float


class Citation(BaseModel):
    source_id: str
    title: str
    topic: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    topic: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=25)


class AskResponse(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    insufficient_context: bool
    sources: list[Citation]
    retrieved: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


class EvalCase(BaseModel):
    question: str
    expected_source_id: str | None = None
    expected_answer_contains: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    question: str
    answer: str
    top_source_id: str | None
    hit_expected_source: bool
    answer_contains_expected_terms: bool


class DatasetStats(BaseModel):
    documents: int
    topics: dict[str, int]
    document_types: dict[str, int]

    @computed_field
    @property
    def topic_count(self) -> int:
        return len(self.topics)
