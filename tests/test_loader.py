"""Smallest check that YtDlpLoader downloads a real video file. Requires network access."""

import os
import subprocess
import tempfile

import _path  # noqa: F401

from core.loader import YtDlpLoader

# "Me at the zoo" - the first video ever uploaded to YouTube (2005), 19s long.
# Historically significant enough to be a stable fixture unlikely to disappear.
TEST_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def test_load_downloads_a_playable_video() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = YtDlpLoader().load(TEST_URL, tmp)

        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
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
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert probe.stdout.strip() == "video"

        mtime = os.path.getmtime(path)
        cached_path = YtDlpLoader().load(TEST_URL, tmp)
        assert cached_path == path
        assert os.path.getmtime(path) == mtime

        print(
            f"OK: downloaded {path}, cache hit on repeat load"
        )


if __name__ == "__main__":
    test_load_downloads_a_playable_video()
