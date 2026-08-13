"""Smallest check that compute_rd_curve produces a monotonic-ish quality-vs-bitrate curve."""

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

from encoder.rd_curve import compute_rd_curve


def test_higher_bitrate_gives_higher_or_equal_ssim():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=10",
                src,
            ],
            check=True,
            capture_output=True,
        )

        curve = compute_rd_curve(
            src, kbps_rungs=[200, 4000]
        )

        assert [k for k, _ in curve] == [200, 4000]
        assert all(0.0 <= ssim <= 1.0 for _, ssim in curve)
        assert curve[1][1] >= curve[0][1]
        print("OK:", curve)


if __name__ == "__main__":
    test_higher_bitrate_gives_higher_or_equal_ssim()
