"""RRFRetriever: index scenes into, and fuse nearest-neighbor rankings across, multiple collections via Reciprocal Rank Fusion."""

from typing import NamedTuple, cast

from core.indexing import (
    Collection,
    SceneClip,
    build_scene_meta,
)

RRF_K = 60


class RRFRetriever(NamedTuple):
    collections: list[Collection]

    def index_scene(self, clip: SceneClip) -> None:
        """Embed `clip` under every collection this retriever was built with, recording it in each collection's store under its namespace."""
        scene_meta = build_scene_meta(clip)
        for collection in self.collections:
            collection.record(clip.path, scene_meta)

    def retrieve(
        self, clip_path: str, topk: int = 5
    ) -> list[tuple[str, float]]:
        """Search every collection for `clip_path`'s neighbors and fuse their rankings via RRF: score(clip) = sum(1 / (RRF_K + rank + 1)) over every collection it appears in. Returns the topk clips by fused score, descending."""
        scores: dict[str, float] = {}
        for collection in self.collections:
            matches = collection.search(
                clip_path, topk=topk
            )
            for rank, (meta, _) in enumerate(matches):
                clip = cast(str, meta["clip"])
                scores[clip] = scores.get(
                    clip, 0.0
                ) + 1.0 / (RRF_K + rank + 1)
        return sorted(
            scores.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:topk]
