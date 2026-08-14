"""Rate-distortion curve for a scene: VMAF vs bitrate at a fixed set of rungs.

Purely descriptive data recorded per scene (see ingest.py) for display - not
used for content-similarity matching. InternVideo2 (embed.py) is the only
embedding used for that.
"""

import re
import subprocess
import tempfile
from pathlib import Path

from core.encode import encode_video

DEFAULT_KBPS_RUNGS = [500, 1000, 2000, 4000, 8000]


def _vmaf(reference: str, encoded: str) -> float:
    """VMAF score of `encoded` against `reference`, via ffmpeg's libvmaf filter."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            encoded,
            "-i",
            reference,
            "-lavfi",
            "libvmaf",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(
        r"VMAF score: ([\d.]+)", result.stderr
    )
    if not match:
        raise RuntimeError(
            f"couldn't parse vmaf from ffmpeg output: {result.stderr}"
        )
    return float(match.group(1))


def compute_rd_curve(
    scene_path: str,
    kbps_rungs: list[int] = DEFAULT_KBPS_RUNGS,
) -> list[tuple[int, float]]:
    """Encode `scene_path` at each bitrate in `kbps_rungs`, measuring VMAF against the source.

    Returns (kbps, vmaf) pairs in rung order - the scene's rate-distortion curve.
    """
    with tempfile.TemporaryDirectory() as tmp:
        curve = []
        for kbps in kbps_rungs:
            encoded = str(Path(tmp) / f"{kbps}.mp4")
            encode_video(scene_path, encoded, kbps=kbps)
            curve.append((kbps, _vmaf(scene_path, encoded)))
        return curve
