from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from daverag.schemas import NormalizedDocument


class DocumentIndex:
    def __init__(self, documents: list[NormalizedDocument], matrix: np.ndarray) -> None:
        self.documents = documents
        self.matrix = matrix.astype(np.float32)
        self.by_id = {doc.id: doc for doc in documents}

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "documents.json").write_text(
            json.dumps([doc.model_dump(mode="json") for doc in self.documents], indent=2)
        )
        np.save(output_dir / "embeddings.npy", self.matrix)
        (output_dir / "index_meta.json").write_text(
            json.dumps(
                {
                    "embedding_dimensions": int(self.matrix.shape[1]) if self.matrix.ndim == 2 else 0,
                    "document_count": len(self.documents),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, output_dir: Path) -> "DocumentIndex":
        documents = [
            NormalizedDocument.model_validate(item)
            for item in json.loads((output_dir / "documents.json").read_text())
        ]
        matrix = np.load(output_dir / "embeddings.npy")
        return cls(documents=documents, matrix=matrix)
