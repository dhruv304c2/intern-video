"""Smallest check that encode_video actually produces a playable video stream."""

import os
import subprocess
import sys
import tempfile

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
)

from encoder.encode import encode_video


def test_encode_produces_video_stream():
    with (
        tempfile.NamedTemporaryFile(suffix=".mp4") as src,
        tempfile.NamedTemporaryFile(suffix=".mp4") as out,
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=10",
                src.name,
            ],
            check=True,
            capture_output=True,
        )
        encode_video(src.name, out.name, kbps=500)
        assert os.path.getsize(out.name) > 0
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
                out.name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert probe.stdout.strip() == "video"
        print("OK: encoded", out.name)


if __name__ == "__main__":
    test_encode_produces_video_stream()
