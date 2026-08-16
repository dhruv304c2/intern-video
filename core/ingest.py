"""Split a video into scenes (PySceneDetect) and encode each scene clip."""

import os

from scenedetect import ContentDetector, detect

from core.embedder.internvideo2 import _ORIG_CWD
from core.encode import encode_video
from core.indexing import SceneClip


def split_scenes(
    video_path: str,
    out_dir: str,
    kbps: int = 2500,
    max_scenes: int = 200,
) -> list[SceneClip]:
    """Detect scene cuts in `video_path` and write one encoded clip per scene into `out_dir`. Caps at `max_scenes` (a false-positive-heavy detection run on fast-cut/animated content can otherwise yield hundreds of spurious near-instant scenes) - only the first `max_scenes` are encoded, the rest are dropped."""
    # core.embedder.internvideo2 (imported above) changes the process's cwd
    # as a side effect of loading the vendored model config - resolve
    # user-supplied paths against the original cwd it captured, not the
    # current one.
    video_path = os.path.abspath(
        os.path.join(_ORIG_CWD, video_path)
    )
    out_dir = os.path.abspath(
        os.path.join(_ORIG_CWD, out_dir)
    )
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    scene_list = detect(video_path, ContentDetector()) or [
        (None, None)
    ]
    if len(scene_list) > max_scenes:
        print(
            f"{base}: {len(scene_list)} scene(s) detected, "
            f"capping at {max_scenes}",
            flush=True,
        )
        scene_list = scene_list[:max_scenes]
    n = len(scene_list)
    print(f"{base}: {n} scene(s) detected", flush=True)

    clips = []
    for i, (start, end) in enumerate(scene_list, start=1):
        out_path = os.path.join(
            out_dir, f"{base}-Scene-{i:03d}.mp4"
        )
        start_s = (
            start.seconds if start is not None else None
        )
        end_s = end.seconds if end is not None else None
        print(f"[{i}/{n}] encoding {out_path}", flush=True)
        encode_video(
            video_path,
            out_path,
            kbps=kbps,
            start=start_s,
            end=end_s,
        )
        clips.append(
            SceneClip(
                out_path,
                base,
                i,
                start_s or 0.0,
                end_s or 0.0,
            )
        )
    return clips
