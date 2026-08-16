"""RDRecallTest: does content-similarity retrieval also surface RD-curve-similar scenes?"""

import csv
import os
from collections.abc import Iterator

from core.embedder.internvideo2 import _ORIG_CWD
from core.indexing import SceneClip
from core.ingest import split_scenes
from core.loader import VideoLoader
from core.rd_curve import compute_rd_curve
from core.retrieval import RRFRetriever
from recall.report import (
    NeighborResult,
    QueryResult,
    build_report,
)


class RDRecallTest:
    def __init__(
        self,
        retriever: RRFRetriever,
        loader: VideoLoader,
        video_dir: str = ".cache/recall/videos",
        scenes_dir: str = ".cache/recall/scenes",
    ) -> None:
        """`retriever` supplies content-similarity neighbors (build it from content collections only - e.g. `RRFRetriever([internvideo2_collection])` - not the rd-curve collection, or RD-curve similarity would leak into neighbor selection); RD-curve similarity between a query and each neighbor is computed directly (no index/search on the RD side). `loader` downloads each dataset URL (parsed from the CSVs passed to `run()`) before it's split into scenes."""
        self.retriever = retriever
        self.loader = loader
        # core.embedder.internvideo2 (imported transitively above) changes
        # the process's cwd as a side effect of loading the vendored model
        # config - resolve these user-supplied paths against the original
        # cwd it captured, not the current one.
        self.video_dir = os.path.join(_ORIG_CWD, video_dir)
        self.scenes_dir = os.path.join(
            _ORIG_CWD, scenes_dir
        )

    def score(self, clip_path: str, topk: int = 5) -> float:
        """Mean RD-curve similarity between `clip_path` and its topk content-similarity neighbors (self excluded)."""
        return self._score_detail(clip_path, topk).score

    def run(
        self,
        index_csv: str,
        query_csv: str,
        topk: int = 5,
        report_path: str = ".cache/recall/report.html",
    ) -> float:
        """Parse `index_csv` and `query_csv` (each one video URL per row, first column), download+index every `index_csv` video into `self.retriever`, then download+split (but don't index) every `query_csv` video, score each of its scene clips against the index, write an HTML report (thumbnails + RD-curve comparison) to `report_path`, and return the mean score. Keeping index and query videos disjoint means a query's neighbors are never scenes from its own source video."""
        self._index_videos(_parse_dataset(index_csv))
        clip_paths = [
            clip.path
            for clip in self._load_scenes(
                _parse_dataset(query_csv)
            )
        ]
        n = len(clip_paths)
        results = []
        for i, clip in enumerate(clip_paths, 1):
            result = self._score_detail(clip, topk)
            print(
                f"[score {i}/{n}] {clip}: {result.score:.4f}",
                flush=True,
            )
            results.append(result)
        report_path = os.path.join(_ORIG_CWD, report_path)
        build_report(results, report_path)
        print(
            f"Report written to {report_path}", flush=True
        )
        return (
            sum(r.score for r in results) / len(results)
            if results
            else 0.0
        )

    def _score_detail(
        self, clip_path: str, topk: int = 5
    ) -> QueryResult:
        """Like `score()`, but also keeps the query/neighbor RD curves for report rendering."""
        neighbors = _other_clips(
            self.retriever, clip_path, topk
        )
        query_curve = compute_rd_curve(clip_path)
        neighbor_results = []
        for neighbor in neighbors:
            curve = compute_rd_curve(neighbor)
            similarity = _rd_curve_similarity(
                query_curve, curve
            )
            neighbor_results.append(
                NeighborResult(neighbor, similarity, curve)
            )
        score = (
            sum(n.similarity for n in neighbor_results)
            / len(neighbor_results)
            if neighbor_results
            else 0.0
        )
        return QueryResult(
            clip_path, query_curve, neighbor_results, score
        )

    def _index_videos(self, urls: list[str]) -> None:
        """Download each URL via `self.loader`, split it into scenes, and index every scene into `self.retriever`."""
        for clip in self._load_scenes(
            urls, label="indexing"
        ):
            self.retriever.index_scene(clip)

    def _load_scenes(
        self, urls: list[str], label: str = "loading"
    ) -> Iterator[SceneClip]:
        """Download each URL via `self.loader` and split it into scenes, without indexing them."""
        n = len(urls)
        for i, url in enumerate(urls, 1):
            print(
                f"[video {i}/{n}] downloading {url}",
                flush=True,
            )
            video_path = self.loader.load(
                url, self.video_dir
            )
            scenes = split_scenes(
                video_path, self.scenes_dir
            )
            print(
                f"[video {i}/{n}] {label} {len(scenes)} scene(s)",
                flush=True,
            )
            yield from scenes


def _parse_dataset(csv_path: str) -> list[str]:
    """Read `csv_path` (resolved against the original cwd - see `RDRecallTest.__init__`) and return every row's first column as a video URL."""
    csv_path = os.path.join(_ORIG_CWD, csv_path)
    with open(csv_path, newline="") as f:
        return [row[0] for row in csv.reader(f) if row]


def _other_clips(
    retriever: RRFRetriever, clip_path: str, topk: int
) -> list[str]:
    """`clip_path`'s topk fused nearest neighbors via `retriever`, excluding itself."""
    matches = retriever.retrieve(clip_path, topk=topk + 1)
    return [
        clip for clip, _ in matches if clip != clip_path
    ][:topk]


def _rd_curve_similarity(
    a: list[tuple[int, float]], b: list[tuple[int, float]]
) -> float:
    """1 - mean absolute VMAF difference (scaled to [0, 1]) between two same-rung RD curves."""
    diffs = [
        abs(vmaf_a - vmaf_b)
        for (_, vmaf_a), (_, vmaf_b) in zip(a, b)
    ]
    return 1.0 - (sum(diffs) / len(diffs)) / 100.0
