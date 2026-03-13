from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI

from daverag.config import Settings


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class LocalHashEmbeddingBackend(EmbeddingBackend):
    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                token_hash = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(token_hash[:4], "big") % self.dimensions
                sign = -1.0 if token_hash[4] % 2 else 1.0
                vectors[row, index] += sign
            norm = math.sqrt(float(np.dot(vectors[row], vectors[row])))
            if norm:
                vectors[row] /= norm
        return vectors


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI embedding backend.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)


def build_embedding_backend(settings: Settings) -> EmbeddingBackend:
    if settings.embedding_backend == "openai":
        return OpenAIEmbeddingBackend(settings)
    return LocalHashEmbeddingBackend()
