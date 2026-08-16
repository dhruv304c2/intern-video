"""Index an encoded scene clip into one or more anns.

Split out of ingest.py so indexing isn't tied to scene-splitting - any
caller with a `SceneClip` (from ingest.py's split_scenes, or elsewhere) can
build an `Indexer` against its own anns.
"""

from typing import NamedTuple

from core.indexing.ann import Ann
from core.indexing.meta import SceneClip, build_scene_meta


class Indexer(NamedTuple):
    anns: list[Ann]

    def index_scene(self, clip: SceneClip) -> None:
        """Embed `clip` under every ann this indexer was built with, recording it in that ann's store under its namespace."""
        print(
            f"[scene {clip.scene}] extracting thumbnails",
            flush=True,
        )
        scene_meta = build_scene_meta(clip)
        for ann in self.anns:
            print(
                f"[scene {clip.scene}] embedding for {ann.namespace!r}",
                flush=True,
            )
            ann.record(clip.path, scene_meta)
