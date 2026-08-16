"""Smallest check that find_similar returns an indexed scene as its own top match."""

import os
import subprocess
import tempfile

import _path  # noqa: F401

from core.embedder import (
    InternVideo2Embedder,
    RdCurveEmbedder,
)
from core.indexing import Ann
from core.retrieval import find_similar
from core.vectorstore import ChromaVectorStore
from ingest import ingest_video


def test_find_similar_returns_self_as_top_match() -> None:
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
        store = ChromaVectorStore(
            os.path.join(tmp, "index")
        )
        anns = [
            Ann(
                "internvideo2",
                InternVideo2Embedder(),
                store,
            ),
            Ann("rd-curve", RdCurveEmbedder(), store),
        ]
        outputs = ingest_video(
            src, scenes_dir, kbps=500, anns=anns
        )

        results = find_similar(outputs[0], anns, topk=5)

        assert set(results) == {"internvideo2", "rd-curve"}
        for matches in results.values():
            assert matches
            top_meta, top_similarity = matches[0]
            assert top_meta["clip"] == outputs[0]
            assert top_similarity > 0.99
        print(f"OK: self is top match in {list(results)}")


if __name__ == "__main__":
    test_find_similar_returns_self_as_top_match()
