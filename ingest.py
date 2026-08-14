"""Split a video into scenes (PySceneDetect) and encode each scene clip."""

import argparse
import os
import subprocess

from scenedetect import ContentDetector, detect

from core.encode import encode_video
from core.rd_curve import compute_rd_curve
from core.vectorstore import VectorStore

THUMBNAIL_FRACTIONS = [0.25, 0.5, 0.75]


def extract_thumbnails(clip_path: str) -> list[str]:
    """Grab a few still frames from `clip_path`, written alongside it as `<clip>-thumb-N.jpg`."""
    duration = float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                clip_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    stem = os.path.splitext(clip_path)[0]
    paths = []
    for i, frac in enumerate(THUMBNAIL_FRACTIONS, start=1):
        out_path = f"{stem}-thumb-{i}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(duration * frac),
                "-i",
                clip_path,
                "-frames:v",
                "1",
                out_path,
            ],
            check=True,
            capture_output=True,
        )
        paths.append(out_path)
    return paths


def ingest_video(
    video_path: str,
    out_dir: str,
    kbps: int = 2500,
    store: VectorStore | None = None,
) -> list[str]:
    """Detect scene cuts in `video_path` and write one encoded clip per scene into `out_dir`.

    If `store` is given, also computes each scene's rate-distortion curve (see
    rd_curve.py), adds it to the store's "rd-curve" namespace, and embeds
    the scene with InternVideo2 (see embed.py) into the store's "internvideo2"
    namespace - the only embedder used, so this always runs when a store is
    given, requiring the checkpoint (see README "One-time setup").
    """
    # embed.py (imported lazily below) changes the process's cwd as a
    # side effect of loading the vendored model config - resolve paths to
    # absolute up front so later scenes in this loop aren't affected.
    video_path = os.path.abspath(video_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    scene_list = detect(video_path, ContentDetector()) or [
        (None, None)
    ]
    n = len(scene_list)
    print(f"{base}: {n} scene(s) detected", flush=True)

    outputs = []
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
        outputs.append(out_path)

        if store is not None:
            print(
                f"[{i}/{n}] extracting thumbnails",
                flush=True,
            )
            scene_meta = {
                "clip": out_path,
                "source_video": base,
                "scene": i,
                "start": start_s or 0.0,
                "end": end_s or 0.0,
                "thumbnails": extract_thumbnails(out_path),
            }
            print(
                f"[{i}/{n}] computing rate-distortion curve",
                flush=True,
            )
            curve = compute_rd_curve(out_path)
            store.add(
                "rd-curve",
                [vmaf for _, vmaf in curve],
                {
                    **scene_meta,
                    "kbps_rungs": [k for k, _ in curve],
                },
            )
            print(
                f"[{i}/{n}] embedding with InternVideo2",
                flush=True,
            )
            from core.embed import embed_video

            store.add(
                "internvideo2",
                embed_video(out_path),
                scene_meta,
            )
    return outputs


def main():
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

    store = VectorStore(args.index) if args.index else None
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
