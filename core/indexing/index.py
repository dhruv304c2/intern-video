"""Index an encoded scene clip: rate-distortion curve + InternVideo2 embedding.

Split out of ingest.py so indexing isn't tied to scene-splitting - any
caller with a `SceneClip` (from ingest.py's split_scenes, or elsewhere) can
reuse `index_scene` against a VectorStore.
"""

from core.embedder import Embedder, InternVideo2Embedder
from core.indexing.meta import SceneClip, build_scene_meta
from core.rd_curve import compute_rd_curve
from core.vectorstore import VectorStore


def index_scene(
    clip: SceneClip,
    store: VectorStore,
    embedder: Embedder | None = None,
) -> None:
    """Compute `clip`'s rate-distortion curve and InternVideo2 embedding, and record both in `store`.

    Defaults `embedder` to InternVideo2Embedder - requires the checkpoint
    (see README "One-time setup") unless a different `embedder` is passed in.
    """
    embedder = embedder or InternVideo2Embedder()
    print(
        f"[scene {clip.scene}] extracting thumbnails",
        flush=True,
    )
    scene_meta = build_scene_meta(clip)
    print(
        f"[scene {clip.scene}] computing rate-distortion curve",
        flush=True,
    )
    curve = compute_rd_curve(clip.path)
    store.add(
        "rd-curve",
        [vmaf for _, vmaf in curve],
        {**scene_meta, "kbps_rungs": [k for k, _ in curve]},
    )
    print(
        f"[scene {clip.scene}] embedding with InternVideo2",
        flush=True,
    )
    store.add(
        "internvideo2",
        embedder.embed(clip.path),
        scene_meta,
    )
