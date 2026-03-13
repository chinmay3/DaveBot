from __future__ import annotations

import json
from pathlib import Path

from daverag.normalization import dataset_stats, to_document
from daverag.schemas import DatasetStats, NormalizedDocument, RawRecord


def load_raw_records(path: Path) -> list[RawRecord]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array.")
    return [RawRecord.model_validate(item) for item in data]


def load_documents(path: Path) -> tuple[list[NormalizedDocument], DatasetStats]:
    records = load_raw_records(path)
    documents = [to_document(record) for record in records]
    return documents, dataset_stats(documents)
