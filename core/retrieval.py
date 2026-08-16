"""Multi-ann similarity search: query a clip against every ann, keyed by namespace."""

from typing import NamedTuple

from core.indexing import Ann, Ingestor, default_anns
from core.vectorstore import Metadata, VectorStore


def find_similar(
    clip_path: str,
    anns: list[Ann],
    topk: int = 5,
) -> dict[str, list[tuple[Metadata, float]]]:
    """Search every ann in `anns` for `clip_path`'s nearest neighbors, keyed by namespace."""
    return {
        ann.namespace: ann.search(clip_path, topk=topk)
        for ann in anns
    }


class Retriever(NamedTuple):
    anns: list[Ann]

    def retrieve(
        self, clip_path: str, topk: int = 5
    ) -> dict[str, list[tuple[Metadata, float]]]:
        """Search every ann this retriever was built with for `clip_path`'s nearest neighbors."""
        return find_similar(clip_path, self.anns, topk=topk)


def build_pipeline(
    store: VectorStore,
) -> tuple[Ingestor, Retriever]:
    """An Ingestor/Retriever pair sharing the same anns, so what's indexed is exactly what's searched."""
    anns = default_anns(store)
    return Ingestor(anns), Retriever(anns)
