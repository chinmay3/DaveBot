from pathlib import Path
import os

from pydantic import BaseModel


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class Settings(BaseModel):
    app_name: str = "DaveRAG"
    data_path: Path = Path("dave_data.json")
    index_dir: Path = Path(".artifacts/index")
    embedding_backend: str = "local"
    generation_backend: str = "extractive"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    retrieval_top_k: int = 12
    rerank_top_k: int = 5
    min_retrieval_score: float = 0.15


def _env_value(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


_load_dotenv()

settings = Settings(
    data_path=Path(_env_value("DATA_PATH", "dave_data.json")),
    index_dir=Path(_env_value("INDEX_DIR", ".artifacts/index")),
    embedding_backend=_env_value("EMBEDDING_BACKEND", "local") or "local",
    generation_backend=_env_value("GENERATION_BACKEND", "extractive") or "extractive",
    openai_api_key=_env_value("OPENAI_API_KEY"),
    openai_chat_model=_env_value("OPENAI_CHAT_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",
    openai_embedding_model=_env_value("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    or "text-embedding-3-small",
    retrieval_top_k=int(_env_value("RETRIEVAL_TOP_K", "12") or "12"),
    rerank_top_k=int(_env_value("RERANK_TOP_K", "5") or "5"),
    min_retrieval_score=float(_env_value("MIN_RETRIEVAL_SCORE", "0.15") or "0.15"),
)
