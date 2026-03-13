from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import shutil

from daverag.classification import classify_query, topic_filter_for_class
from daverag.config import Settings, settings as default_settings
from daverag.data import load_documents
from daverag.embeddings import build_embedding_backend
from daverag.generation import build_generation_backend, citations_from_results
from daverag.index import DocumentIndex
from daverag.retrieval import HybridRetriever
from daverag.schemas import AskResponse, DatasetStats, HealthResponse


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.embedding_backend = build_embedding_backend(self.settings)
        self.generation_backend = build_generation_backend(self.settings)
        self.index: DocumentIndex | None = None
        self.retriever: HybridRetriever | None = None
        self.stats: DatasetStats | None = None

    def build_index(self, persist: bool = True) -> None:
        documents, stats = load_documents(self.settings.data_path)
        matrix = self.embedding_backend.embed([doc.text_for_embedding for doc in documents])
        self.index = DocumentIndex(documents=documents, matrix=matrix)
        self.retriever = HybridRetriever(self.index)
        self.stats = stats
        if persist:
            self.index.save(self.settings.index_dir)

    def load_or_build_index(self) -> None:
        if self.index and self.retriever:
            return
        index_dir = self.settings.index_dir
        if (index_dir / "documents.json").exists() and (index_dir / "embeddings.npy").exists():
            self.index = DocumentIndex.load(index_dir)
            self.retriever = HybridRetriever(self.index)
            _, self.stats = load_documents(self.settings.data_path)
            return
        self.build_index(persist=True)

    def rebuild_index(self) -> None:
        if self.settings.index_dir.exists():
            shutil.rmtree(self.settings.index_dir)
        self.build_index(persist=True)

    @lru_cache(maxsize=256)
    def _cached_query(self, question: str, topic: str | None, top_k: int | None) -> AskResponse:
        self.load_or_build_index()
        assert self.index and self.retriever
        resolved_topic = topic
        if not resolved_topic:
            resolved_topic = topic_filter_for_class(classify_query(question))
        query_vector = self.embedding_backend.embed([question])[0]
        if self.index.matrix.ndim != 2 or self.index.matrix.shape[1] != query_vector.shape[0]:
            self.rebuild_index()
            assert self.index and self.retriever
        results = self.retriever.search(
            query=question,
            query_vector=query_vector,
            top_k=top_k or self.settings.retrieval_top_k,
            rerank_top_k=self.settings.rerank_top_k,
            min_score=self.settings.min_retrieval_score,
            topic=resolved_topic,
        )
        if not results and resolved_topic:
            results = self.retriever.search(
                query=question,
                query_vector=query_vector,
                top_k=top_k or self.settings.retrieval_top_k,
                rerank_top_k=self.settings.rerank_top_k,
                min_score=self.settings.min_retrieval_score,
                topic=None,
            )
        answer = self.generation_backend.answer(question, results)
        insufficient_context = not results
        if insufficient_context:
            confidence = "low"
        elif results[0].combined_score > 0.8:
            confidence = "high"
        else:
            confidence = "medium"
        return AskResponse(
            answer=answer,
            confidence=confidence,
            insufficient_context=insufficient_context,
            sources=citations_from_results(results),
            retrieved=results,
        )

    def ask(self, question: str, topic: str | None = None, top_k: int | None = None) -> AskResponse:
        return self._cached_query(question, topic, top_k)

    def health(self) -> HealthResponse:
        self.load_or_build_index()
        assert self.index
        return HealthResponse(status="ok", documents_indexed=len(self.index.documents))
