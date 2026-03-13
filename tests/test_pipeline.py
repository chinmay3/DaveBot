from daverag.config import Settings
from daverag.service import RagService


def build_service() -> RagService:
    return RagService(
        Settings(
            data_path="dave_data.json",
            index_dir=".artifacts/test-index",
            embedding_backend="local",
            generation_backend="extractive",
        )
    )


def test_index_builds() -> None:
    service = build_service()
    service.build_index(persist=False)
    assert service.stats is not None
    assert service.stats.documents == 197


def test_ask_returns_expected_source() -> None:
    service = build_service()
    service.build_index(persist=False)
    response = service.ask("Who founded Dave's Hot Chicken?")
    assert response.sources
    assert response.sources[0].source_id == "restaurant-history-002"
    assert "Dave Kopushyan" in response.answer
