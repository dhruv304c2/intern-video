"""An ann: a namespace paired with the embedder and store its vectors belong to."""

from typing import NamedTuple

from core.embedder import Embedder
from core.vectorstore import VectorStore


class Ann(NamedTuple):
    namespace: str
    embedder: Embedder
    store: VectorStore

    def record(
        self, clip_path: str, metadata: dict
    ) -> None:
        """Embed `clip_path` and record it in this ann's store under its namespace."""
        self.store.add(
            self.namespace,
            self.embedder.embed(clip_path),
            metadata,
        )
