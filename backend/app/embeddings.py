"""Text embeddings.

`Embedder` is the interface the pipeline (and, from Phase 4, retrieval) depends on; tests inject
a deterministic fake. `FastEmbedEmbedder` runs BAAI/bge-small-en-v1.5 locally through ONNX
Runtime: no API cost, no PyTorch, ~65 MB model, a few hundred passages per second on CPU.
"""

import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Dimension per supported model: the pgvector column is created with a fixed size.
MODEL_DIMENSIONS = {"BAAI/bge-small-en-v1.5": 384}
EMBEDDING_DIMENSION = 384

EmbedProgress = Callable[[int, int], None]  # (done, total)


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed_passages(
        self, texts: Sequence[str], on_progress: EmbedProgress | None = None
    ) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class FastEmbedEmbedder:
    """Thread-safe, lazily loaded: the model is downloaded/loaded on first use, not at startup,
    so the API starts quickly and health checks stay cheap."""

    def __init__(self, model_name: str, *, cache_dir: str, batch_size: int = 32) -> None:
        if model_name not in MODEL_DIMENSIONS:
            raise ValueError(
                f"Unsupported embedding model {model_name!r}; supported: {sorted(MODEL_DIMENSIONS)}"
            )
        self.model_name = model_name
        self.dimension = MODEL_DIMENSIONS[model_name]
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _get_model(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding  # heavy import, deferred

                started = time.perf_counter()
                self._model = TextEmbedding(self.model_name, cache_dir=self._cache_dir)
                logger.info(
                    "embedding model loaded",
                    extra={
                        "model": self.model_name,
                        "seconds": round(time.perf_counter() - started, 1),
                    },
                )
            return self._model

    def embed_passages(
        self, texts: Sequence[str], on_progress: EmbedProgress | None = None
    ) -> np.ndarray:
        model = self._get_model()
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(model.passage_embed(batch, batch_size=self._batch_size))
            if on_progress:
                on_progress(min(start + len(batch), len(texts)), len(texts))
        if not vectors:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _normalise(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.asarray(next(iter(self._get_model().query_embed([text]))), dtype=np.float32)
        return _normalise(vector[None, :])[0]
