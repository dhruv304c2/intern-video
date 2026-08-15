"""Split a video into scenes (PySceneDetect) and encode each scene clip."""

import argparse
import os

from scenedetect import ContentDetector, detect

from core.encode import encode_video
from core.vectorstore import ChromaVectorStore, VectorStore

# captured before the core.indexing import below, which transitively imports
# core.embedder - changing the process's cwd as a side effect of loading
# the vendored model config
_ORIG_CWD = os.getcwd()

from core.embedder import Embedder
from core.indexing import SceneClip, index_scene


def split_scenes(
    video_path: str,
    out_dir: str,
    kbps: int = 2500,
) -> list[SceneClip]:
    """Detect scene cuts in `video_path` and write one encoded clip per scene into `out_dir`."""
    # core.indexing (imported above) has already changed the process's
    # cwd by this point - resolve against the original cwd, not the current one.
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


def ingest_video(
    video_path: str,
    out_dir: str,
    kbps: int = 2500,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> list[str]:
    """Detect scene cuts in `video_path` and write one encoded clip per scene into `out_dir`.

    If `store` is given, also indexes each scene into it (see
    core/indexing.py::index_scene) - always runs when a store is given,
    requiring the InternVideo2 checkpoint (see README "One-time setup")
    unless a different `embedder` is passed in.
    """
    clips = split_scenes(video_path, out_dir, kbps=kbps)
    if store is not None:
        for clip in clips:
            index_scene(clip, store, embedder)
    return [clip.path for clip in clips]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", help="path to a source video file"
    )
    parser.add_argument(
        "out_dir",
        help="directory to write encoded scene clips into",
    )
    parser.add_argument(
        "--kbps",
        type=int,
        default=2500,
        help="target video bitrate in kbps",
    )
    parser.add_argument(
        "--index",
        help="vector store root to also record each scene's rate-distortion curve and InternVideo2 embedding into",
    )
    args = parser.parse_args()

    store = (
        ChromaVectorStore(args.index)
        if args.index
        else None
    )
    outputs = ingest_video(
        args.video,
        args.out_dir,
        kbps=args.kbps,
        store=store,
    )
    print(
        f"wrote {len(outputs)} scene clips to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
