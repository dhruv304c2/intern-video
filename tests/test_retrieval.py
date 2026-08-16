"""Smallest check that Collection.search returns an indexed scene as its own top match."""

import os
import subprocess
import tempfile

import _path  # noqa: F401

from core.embedder import (
    InternVideo2Embedder,
    RdCurveEmbedder,
)
from core.indexing import Collection
from core.ingest import split_scenes
from core.retrieval import RRFRetriever
from core.vectorstore import ChromaVectorStore


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
        internvideo2 = InternVideo2Embedder()
        rd_curve = RdCurveEmbedder()
        collections = [
            Collection.symmetric(
                namespace="internvideo2",
                embedder=internvideo2,
                store=store,
            ),
            Collection.symmetric(
                namespace="rd-curve",
                embedder=rd_curve,
                store=store,
            ),
        ]
        retriever = RRFRetriever(collections)
        clips = split_scenes(src, scenes_dir, kbps=500)
        for clip in clips:
            retriever.index_scene(clip)
        outputs = [clip.path for clip in clips]

        results = {
            collection.namespace: collection.search(
                outputs[0], topk=5
            )
            for collection in collections
        }

        assert set(results) == {"internvideo2", "rd-curve"}
        for matches in results.values():
            assert matches
            top_meta, top_similarity = matches[0]
            assert top_meta["clip"] == outputs[0]
            assert top_similarity > 0.99
        print(f"OK: self is top match in {list(results)}")

        fused = retriever.retrieve(outputs[0], topk=5)
        assert fused
        assert fused[0][0] == outputs[0]
        print(
            f"OK: RRF fused top match is self: {fused[0]}"
        )


if __name__ == "__main__":
    test_find_similar_returns_self_as_top_match()
