"""Rate-distortion curve for a scene: SSIM vs bitrate at a fixed set of rungs.

Also doubles as the scene's embedding: two scenes with similar compressibility
(similar RD curves) get similar vectors, so it feeds the vector index's
"find scenes like this one" namespace (see vectorstore.VectorStore) without
needing a learned embedding model.
"""

import re
import subprocess
import tempfile
from pathlib import Path

from encoder.encode import encode_video

DEFAULT_KBPS_RUNGS = [500, 1000, 2000, 4000, 8000]


def _ssim(reference: str, encoded: str) -> float:
    """Average SSIM ("All") of `encoded` against `reference`, via ffmpeg's ssim filter."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            encoded,
            "-i",
            reference,
            "-lavfi",
            "ssim",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"All:([\d.]+)", result.stderr)
    if not match:
        raise RuntimeError(
            f"couldn't parse ssim from ffmpeg output: {result.stderr}"
        )
    return float(match.group(1))


def compute_rd_curve(
    scene_path: str,
    kbps_rungs: list[int] = DEFAULT_KBPS_RUNGS,
) -> list[tuple[int, float]]:
    """Encode `scene_path` at each bitrate in `kbps_rungs`, measuring SSIM against the source.

    Returns (kbps, ssim) pairs in rung order - the scene's rate-distortion curve.
    """
    with tempfile.TemporaryDirectory() as tmp:
        curve = []
        for kbps in kbps_rungs:
            encoded = str(Path(tmp) / f"{kbps}.mp4")
            encode_video(scene_path, encoded, kbps=kbps)
            curve.append((kbps, _ssim(scene_path, encoded)))
        return curve
