"""A collection: a namespace paired with the embedders and store its vectors belong to."""

from typing import NamedTuple

from core.embedder import Embedder
from core.vectorstore import Metadata, VectorStore


class Collection(NamedTuple):
    namespace: str
    index_embedder: Embedder
    query_embedder: Embedder
    store: VectorStore

    @classmethod
    def symmetric(
        cls,
        namespace: str,
        embedder: Embedder,
        store: VectorStore,
    ) -> "Collection":
        """A collection whose index and query embedder are the same - the common case."""
        return cls(namespace, embedder, embedder, store)

    def record(
        self, clip_path: str, metadata: Metadata
    ) -> None:
        """Embed `clip_path` with the index embedder and record it in this collection's store under its namespace."""
        self.store.add(
            self.namespace,
            self.index_embedder.embed(clip_path),
            metadata,
        )

    def search(
        self, clip_path: str, topk: int = 5
    ) -> list[tuple[Metadata, float]]:
        """Embed `clip_path` with the query embedder and return its topk nearest neighbors in this collection's store/namespace."""
        return self.store.search(
            self.namespace,
            self.query_embedder.embed(clip_path),
            topk=topk,
        )
