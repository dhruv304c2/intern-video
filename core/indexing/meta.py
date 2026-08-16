"""Scene metadata: the SceneClip record and the thumbnail/metadata dict built from it."""

import os
import subprocess
from typing import NamedTuple

THUMBNAIL_FRACTIONS = [0.25, 0.5, 0.75]


class SceneClip(NamedTuple):
    """One encoded scene clip, with enough metadata for an Indexer to record it."""

    path: str
    source_video: str
    scene: int
    start: float
    end: float


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


def build_scene_meta(clip: SceneClip) -> dict:
    """Extract `clip`'s thumbnails and assemble the metadata dict stored alongside its vectors."""
    return {
        "clip": clip.path,
        "source_video": clip.source_video,
        "scene": clip.scene,
        "start": clip.start,
        "end": clip.end,
        "thumbnails": extract_thumbnails(clip.path),
    }
