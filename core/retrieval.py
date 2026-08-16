"""Multi-ann similarity search: query a clip against every ann, keyed by namespace."""

from core.indexing import Ann
from core.vectorstore import Metadata


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
