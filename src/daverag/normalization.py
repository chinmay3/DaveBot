from __future__ import annotations

import re
from collections import Counter

from daverag.schemas import DatasetStats, NormalizedDocument, RawRecord


QA_PATTERN = re.compile(r"^Question:\s*(.*?)\nAnswer:\s*(.*)$", re.S)
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "can",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "some",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
}


def normalize_text(value: str) -> str:
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\bdaves hot chicken\b", "dave's hot chicken", value, flags=re.I)
    return value


def parse_qa_block(text: str) -> tuple[str, str]:
    match = QA_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"Text block is not in Question/Answer format: {text[:80]!r}")
    question, answer = match.groups()
    return normalize_text(question), normalize_text(answer)


def extract_keywords(*values: str, limit: int = 12) -> list[str]:
    counts: Counter[str] = Counter()
    for value in values:
        for token in TOKEN_PATTERN.findall(value.lower()):
            if len(token) < 3 or token in STOPWORDS:
                continue
            counts[token] += 1
    return [token for token, _ in counts.most_common(limit)]


def to_document(record: RawRecord) -> NormalizedDocument:
    question, answer = parse_qa_block(record.text)
    title = normalize_text(record.title)
    topic = normalize_text(record.metadata.topic)
    document_type = normalize_text(record.metadata.type)
    text_for_embedding = (
        f"Title: {title}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Topic: {topic}\n"
        f"Type: {document_type}\n"
        f"Source ID: {record.source_id}"
    )
    text_for_generation = f"{question} {answer}"
    return NormalizedDocument(
        id=record.source_id,
        source_id=record.source_id,
        title=title,
        question=question,
        answer=answer,
        topic=topic,
        document_type=document_type,
        text_for_embedding=text_for_embedding,
        text_for_generation=text_for_generation,
        normalized_question=question.lower(),
        normalized_answer=answer.lower(),
        keywords=extract_keywords(title, question, answer, topic),
    )


def dataset_stats(documents: list[NormalizedDocument]) -> DatasetStats:
    topic_counts: Counter[str] = Counter(doc.topic for doc in documents)
    type_counts: Counter[str] = Counter(doc.document_type for doc in documents)
    return DatasetStats(
        documents=len(documents),
        topics=dict(topic_counts),
        document_types=dict(type_counts),
    )
