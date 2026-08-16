"""Multi-ann similarity search: query a clip against every ann, keyed by namespace."""

from typing import NamedTuple

from core.indexing import Ann, Indexer, default_anns
from core.vectorstore import Metadata, VectorStore


class Retriever(NamedTuple):
    anns: list[Ann]

    def retrieve(
        self, clip_path: str, topk: int = 5
    ) -> dict[str, list[tuple[Metadata, float]]]:
        """Search every ann this retriever was built with for `clip_path`'s nearest neighbors, keyed by namespace."""
        return {
            ann.namespace: ann.search(clip_path, topk=topk)
            for ann in self.anns
        }


def build_pipeline(
    store: VectorStore,
) -> tuple[Indexer, Retriever]:
    """An Indexer/Retriever pair sharing the same anns, so what's indexed is exactly what's searched."""
    anns = default_anns(store)
    return Indexer(anns), Retriever(anns)
