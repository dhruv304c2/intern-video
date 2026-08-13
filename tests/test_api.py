"""Smallest check that GET /videos reflects what ingest_video wrote into the store."""

import os
import subprocess
import sys
import tempfile

from fastapi.testclient import TestClient

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
)

from api import create_app
from encoder.ingest import ingest_video
from vectorstore import VectorStore


def test_list_videos_reflects_ingested_scenes():
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

        scenes_dir = os.path.join(tmp, "scenes")
        store = VectorStore(os.path.join(tmp, "index"))
        ingest_video(src, scenes_dir, kbps=500, store=store)

        client = TestClient(
            create_app(store, media_root=scenes_dir)
        )
        videos = client.get("/videos").json()

        assert len(videos) == 1
        video = videos[0]
        assert video["source_video"] == "src"
        assert len(video["scenes"]) >= 1
        scene_numbers = [
            s["scene"] for s in video["scenes"]
        ]
        assert scene_numbers == sorted(scene_numbers)
        for scene in video["scenes"]:
            assert scene["rd_curve"]["kbps"]
            assert scene["rd_curve"]["ssim"]
            assert scene["has_embedding"] is False
            assert scene["clip"].startswith("/media/")
            assert len(scene["thumbnails"]) == 3
            for thumb in scene["thumbnails"]:
                assert client.get(thumb).status_code == 200
        print(f"OK: {len(video['scenes'])} scene(s) listed")


if __name__ == "__main__":
    test_list_videos_reflects_ingested_scenes()
