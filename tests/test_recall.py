"""Smallest checks for RDRecallTest: self-exclusion, dataset indexing, and a bounded score."""

import csv
import os
import subprocess
import tempfile

import _path  # noqa: F401

from core.embedder import InternVideo2Embedder
from core.indexing import Collection
from core.retrieval import RRFRetriever
from core.vectorstore import ChromaVectorStore
from recall.rd_recall import (
    RDRecallTest,
    _other_clips,
    _rd_curve_similarity,
)


class _LocalLoader:
    """A VideoLoader stub that treats `url` as an already-local file path - no network call."""

    def load(self, url: str, out_dir: str) -> str:
        return url


def test_rd_curve_similarity_pure() -> None:
    curve = [(500, 80.0), (1000, 90.0), (2000, 95.0)]
    assert _rd_curve_similarity(curve, curve) > 0.999

    other = [(500, 0.0), (1000, 0.0), (2000, 0.0)]
    assert _rd_curve_similarity(curve, other) < 0.2
    print(
        "OK: rd curve similarity is correct on identical/far-apart curves"
    )


def _make_video(path: str) -> None:
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
            path,
        ],
        check=True,
        capture_output=True,
    )


def test_run_indexes_dataset_and_score_is_bounded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        index_src = os.path.join(tmp, "index.mp4")
        query_src = os.path.join(tmp, "query.mp4")
        _make_video(index_src)
        _make_video(query_src)

        store = ChromaVectorStore(
            os.path.join(tmp, "index")
        )
        collection = Collection.symmetric(
            namespace="internvideo2",
            embedder=InternVideo2Embedder(),
            store=store,
        )
        retriever = RRFRetriever([collection])
        test = RDRecallTest(
            retriever,
            _LocalLoader(),
            video_dir=tmp,
            scenes_dir=os.path.join(tmp, "scenes"),
        )

        index_scenes = list(test._load_scenes([index_src]))
        assert len(index_scenes) >= 2
        for clip in index_scenes:
            retriever.index_scene(clip)

        query = index_scenes[0].path
        assert query not in _other_clips(
            retriever, query, topk=5
        )

        score = test.score(query, topk=5)
        assert 0.0 <= score <= 1.0
        print(f"OK: score={score:.4f}")

        index_csv = os.path.join(tmp, "index.csv")
        query_csv = os.path.join(tmp, "query.csv")
        with open(index_csv, "w", newline="") as f:
            csv.writer(f).writerow([index_src])
        with open(query_csv, "w", newline="") as f:
            csv.writer(f).writerow([query_src])

        report_path = os.path.join(tmp, "report.html")
        run_score = test.run(
            index_csv, query_csv, report_path=report_path
        )
        assert 0.0 <= run_score <= 1.0
        assert os.path.exists(report_path)
        print(f"OK: run score={run_score:.4f}")


if __name__ == "__main__":
    test_rd_curve_similarity_pure()
    test_run_indexes_dataset_and_score_is_bounded()
