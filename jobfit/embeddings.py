"""Text embeddings + a tiny in-memory vector store (no external DB needed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:  # shape (n, dim)
        ...


class FastEmbedder:
    """Local CPU embeddings using fastembed (ONNX). Downloads the model once (~130 MB)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.array(list(self._model.embed(list(texts))), dtype=np.float32)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity, shape (len(a), len(b))."""
    return normalize(a) @ normalize(b).T


@dataclass
class SearchResult:
    text: str
    score: float
    index: int


@dataclass
class VectorStore:
    """Minimal vector store: keeps normalised vectors in a numpy matrix."""

    embedder: Embedder
    texts: list[str] = field(default_factory=list)
    _matrix: np.ndarray | None = None

    def add(self, texts: Sequence[str]) -> None:
        texts = [t for t in texts if t.strip()]
        if not texts:
            return
        vecs = normalize(self.embedder.embed(texts))
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])
        self.texts.extend(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        if self._matrix is None or not query.strip():
            return []
        q = normalize(self.embedder.embed([query]))[0]
        scores = self._matrix @ q
        top = np.argsort(-scores)[: min(k, len(self.texts))]
        return [SearchResult(self.texts[i], float(scores[i]), int(i)) for i in top]
