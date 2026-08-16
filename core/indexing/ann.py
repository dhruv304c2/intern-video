"""An ann: a namespace paired with the embedder and store its vectors belong to."""

from typing import NamedTuple

from core.embedder import Embedder
from core.vectorstore import Metadata, VectorStore


class Ann(NamedTuple):
    namespace: str
    embedder: Embedder
    store: VectorStore

    def record(
        self, clip_path: str, metadata: Metadata
    ) -> None:
        """Embed `clip_path` and record it in this ann's store under its namespace."""
        self.store.add(
            self.namespace,
            self.embedder.embed(clip_path),
            metadata,
        )

    def search(
        self, clip_path: str, topk: int = 5
    ) -> list[tuple[Metadata, float]]:
        """Embed `clip_path` and return its topk nearest neighbors in this ann's store/namespace."""
        return self.store.search(
            self.namespace,
            self.embedder.embed(clip_path),
            topk=topk,
        )
