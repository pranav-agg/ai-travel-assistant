"""Embedding provider, in one place.

Centralised here because `ingest.py` and `retriever.py` must agree exactly:
a vector store is only queryable by the same model that wrote it. Which
motivates the other thing this module does -- see `EmbeddingFingerprint`.
"""

from __future__ import annotations
import json
from dataclasses import dataclass

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    VECTORSTORE_DIR,
)


FINGERPRINT_FILE = VECTORSTORE_DIR / ".embedding_model.json"


class EmbeddingConfigError(RuntimeError):
    """Raised when embeddings cannot be constructed or would be mismatched."""

def get_embeddings():
    """The embedding model used for both indexing and querying."""
    if not OPENAI_API_KEY:
        raise EmbeddingConfigError(
            "OPENAI_API_KEY is not set. Add it to .env, then rebuild the "
            "vector store with `python -m src.ingest --reset`."
        )


    from langchain_openai import OpenAIEmbeddings


    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        chunk_size=EMBEDDING_BATCH_SIZE,
    )


@dataclass(frozen=True)
class EmbeddingFingerprint:
    model: str

    @classmethod
    def load(cls) -> "EmbeddingFingerprint | None":
        try:
            data = json.loads(FINGERPRINT_FILE.read_text(encoding="utf-8"))
            return cls(model=data["model"])
        except (OSError, ValueError, KeyError):
            return None


    @classmethod
    def write(cls, model: str = EMBEDDING_MODEL) -> None:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        FINGERPRINT_FILE.write_text(
            json.dumps({"model": model}, indent=2), encoding="utf-8"
        )




def check_fingerprint() -> None:
    "Fail with an actionable message if the store was built by another model."
    recorded = EmbeddingFingerprint.load()
    if recorded is None:
        return
    if recorded.model != EMBEDDING_MODEL:
        raise EmbeddingConfigError(
            f"The vector store was built with '{recorded.model}' but the app "
            f"is configured for '{EMBEDDING_MODEL}'. Embeddings from "
            f"different models are not comparable. Rebuild it:\n\n"
            f"    python -m src.ingest --reset"
        )
