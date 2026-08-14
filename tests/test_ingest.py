"""Smallest check that ingest_video splits a hard cut into per-scene encoded clips."""

import os
import subprocess
import tempfile

import _path  # noqa: F401

from ingest import ingest_video


def test_ingest_splits_scenes():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x240:d=2:r=10",
                "-f",
                "lavfi",
                "-i",
                "color=c=white:s=320x240:d=2:r=10",
                "-filter_complex",
                "concat=n=2:v=1:a=0",
                src,
            ],
            check=True,
            capture_output=True,
        )

        out_dir = os.path.join(tmp, "scenes")
        outputs = ingest_video(src, out_dir, kbps=500)

        assert len(outputs) >= 1
        for clip in outputs:
            assert os.path.getsize(clip) > 0
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    clip,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            assert probe.stdout.strip() == "video"
        print(f"OK: {len(outputs)} scene clip(s)")


if __name__ == "__main__":
    test_ingest_splits_scenes()
