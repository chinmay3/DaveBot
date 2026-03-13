from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from daverag.index import DocumentIndex
from daverag.schemas import NormalizedDocument, SearchResult


WORD_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def expand_query(query: str) -> str:
    q = query.lower()
    expansions: list[str] = []

    if "combo" in q or "meal" in q:
        expansions.extend(["combo", "meal", "fries", "#1", "#2", "#3", "#4"])
    if "slider" in q and "tender" in q:
        expansions.extend(["one slider one tender", "dave's #3", "combo meal"])
    if "shake" in q or "drink" in q or "beverage" in q:
        expansions.extend(["top-loaded shakes", "burstin beverages", "milkshake"])
    if "menu" in q or "available" in q:
        expansions.extend(["sliders", "tenders", "combo meals", "shakes"])

    if not expansions:
        return query
    return f"{query} {' '.join(expansions)}"


class HybridRetriever:
    def __init__(self, index: DocumentIndex) -> None:
        self.index = index
        self.doc_term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        for doc in index.documents:
            tokens = tokenize(doc.text_for_embedding)
            frequencies = Counter(tokens)
            self.doc_term_freqs.append(frequencies)
            self.doc_lengths.append(sum(frequencies.values()))
            for token in frequencies:
                self.doc_freqs[token] += 1
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

    def _bm25_score(self, query: str, document_index: int, k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        tokens = tokenize(query)
        doc_tf = self.doc_term_freqs[document_index]
        doc_len = self.doc_lengths[document_index]
        total_docs = len(self.index.documents)
        for token in tokens:
            term_freq = doc_tf.get(token, 0)
            if not term_freq:
                continue
            doc_freq = self.doc_freqs.get(token, 0)
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            denom = term_freq + k1 * (1 - b + b * doc_len / self.avg_doc_length)
            score += idf * ((term_freq * (k1 + 1)) / denom)
        return score

    def _domain_boost(self, query: str, document: NormalizedDocument) -> float:
        q = query.lower()
        text = f"{document.title} {document.question} {document.answer}".lower()
        boost = 0.0

        if "combo" in q or "meal" in q:
            if "combo meal" in text or "#" in text or "fries" in text:
                boost += 0.45
        if "slider" in q and "tender" in q:
            if "one tender and one slider" in text or "one slider and one tender" in text:
                boost += 0.9
        if "shake" in q and "shake" in text:
            boost += 0.55
        if "available" in q or "menu" in q:
            if any(term in text for term in ["sliders", "tenders", "shakes", "cauliflower"]):
                boost += 0.2

        return boost

    def _rerank_score(self, query: str, document: NormalizedDocument) -> float:
        query_tokens = set(tokenize(query))
        doc_tokens = set(tokenize(document.text_for_generation))
        if not query_tokens or not doc_tokens:
            return 0.0
        overlap = len(query_tokens & doc_tokens) / len(query_tokens)
        title_boost = 0.15 if any(token in document.title.lower() for token in query_tokens) else 0.0
        keyword_boost = 0.05 * len(query_tokens & set(document.keywords))
        return overlap + title_boost + keyword_boost

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int,
        rerank_top_k: int,
        min_score: float,
        topic: str | None = None,
    ) -> list[SearchResult]:
        expanded_query = expand_query(query)
        matrix = self.index.matrix
        vector_scores = matrix @ query_vector
        candidates: list[SearchResult] = []
        for idx, document in enumerate(self.index.documents):
            if topic and document.topic != topic:
                continue
            vector_score = float(vector_scores[idx])
            keyword_score = self._bm25_score(expanded_query, idx)
            combined_score = 0.65 * vector_score + 0.35 * keyword_score + self._domain_boost(query, document)
            candidates.append(
                SearchResult(
                    document=document,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    rerank_score=0.0,
                    combined_score=combined_score,
                )
            )
        candidates.sort(key=lambda item: item.combined_score, reverse=True)
        rerank_candidates = candidates[:top_k]
        for candidate in rerank_candidates:
            candidate.rerank_score = self._rerank_score(expanded_query, candidate.document)
            candidate.combined_score = 0.5 * candidate.combined_score + 0.5 * candidate.rerank_score
        rerank_candidates.sort(key=lambda item: item.combined_score, reverse=True)
        return [item for item in rerank_candidates[:rerank_top_k] if item.combined_score >= min_score]
