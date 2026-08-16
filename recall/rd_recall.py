"""RDRecallTest: does content-similarity retrieval also surface RD-curve-similar scenes?"""

import json
import os
import random
from collections.abc import Iterator
from typing import cast

from tqdm import tqdm

from core.embedder.internvideo2 import _ORIG_CWD
from core.indexing import SceneClip, extract_thumbnails
from core.ingest import split_scenes
from core.loader import VideoLoader
from core.rd_curve import (
    DEFAULT_KBPS_RUNGS,
    compute_rd_curve,
)
from core.retrieval import RRFRetriever
from core.vectorstore.protocol import Metadata
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
        cache_path: str = ".cache/recall/indexed_urls.json",
        max_scenes: int = 50,
    ) -> None:
        """`retriever` supplies content-similarity neighbors (build it from content collections only - e.g. `RRFRetriever([internvideo2_collection])` - not the rd-curve collection, or RD-curve similarity would leak into neighbor selection); RD-curve similarity between a query and each neighbor is computed directly (no index/search on the RD side). `loader` resolves each dataset entry passed to `run()` (a URL, local path, or whatever else it understands) into a local video file before it's split into scenes. `cache_path` records which index-set entries have already been indexed, so a later `run()` skips re-downloading/re-splitting/re-indexing them - see `_index_videos`. `max_scenes` caps how many scenes each video is split into (see `core.ingest.split_scenes`)."""
        self.retriever = retriever
        self.loader = loader
        self.max_scenes = max_scenes
        # core.embedder.internvideo2 (imported transitively above) changes
        # the process's cwd as a side effect of loading the vendored model
        # config - resolve these user-supplied paths against the original
        # cwd it captured, not the current one.
        self.video_dir = os.path.join(_ORIG_CWD, video_dir)
        self.scenes_dir = os.path.join(
            _ORIG_CWD, scenes_dir
        )
        self.cache_path = os.path.join(
            _ORIG_CWD, cache_path
        )

    def score(self, clip_path: str, topk: int = 5) -> float:
        """Mean RD-curve similarity between `clip_path` and its topk content-similarity neighbors (self excluded)."""
        return self._score_detail(clip_path, topk).score

    def run(
        self,
        index_videos: list[str],
        query_videos: list[str],
        topk: int = 5,
        report_path: str = ".cache/recall/report.html",
        bypass_cache: bool = False,
    ) -> float:
        """Download+index every `index_videos` entry (a URL or local path - whatever `self.loader` accepts) into `self.retriever`, then download+split (but don't index) every `query_videos` entry, score each of its scene clips against the index, write an HTML report (thumbnails + RD-curve comparison) to `report_path`, and return the mean score. Keeping index and query videos disjoint means a query's neighbors are never scenes from its own source video. `index_videos` entries already recorded in `self.cache_path` (from a prior `run()`) are skipped - pass `bypass_cache=True` to force re-downloading/re-splitting/re-indexing all of them."""
        self._index_videos(
            index_videos, bypass_cache=bypass_cache
        )
        clip_paths = [
            clip.path
            for clip in self._load_scenes(query_videos)
        ]
        results = []
        pbar = tqdm(
            clip_paths, desc="Scoring queries", unit="clip"
        )
        for clip in pbar:
            extract_thumbnails(clip)
            result = self._score_detail(clip, topk)
            os.remove(clip)
            pbar.set_postfix_str(
                f"{os.path.basename(clip)}: {result.score:.4f}"
            )
            results.append(result)
        report_path = os.path.join(_ORIG_CWD, report_path)
        build_report(results, report_path)
        tqdm.write(f"Report written to {report_path}")
        mean_score = (
            sum(r.score for r in results) / len(results)
            if results
            else 0.0
        )
        mean_baseline = (
            sum(r.baseline_score for r in results)
            / len(results)
            if results
            else 0.0
        )
        tqdm.write(
            f"Mean score: {mean_score:.4f} vs random-neighbor baseline: {mean_baseline:.4f}"
        )
        return mean_score

    def _score_detail(
        self, clip_path: str, topk: int = 5
    ) -> QueryResult:
        """Like `score()`, but also keeps the query/neighbor RD curves for report rendering, plus a `baseline_score` - the same mean RD-curve similarity but against `topk` *random* indexed clips instead of content-similarity neighbors, so the report shows whether content similarity beats picking neighbors at random.

        Neighbors' RD curves come from their stored metadata (precomputed at
        index time - see `build_scene_meta`), not by re-reading their clip
        file, which may already have been deleted after indexing.
        """
        neighbors = _other_clips(
            self.retriever, clip_path, topk
        )
        query_curve = compute_rd_curve(clip_path)
        neighbor_results = []
        for neighbor, meta in neighbors:
            curve = _rd_curve_from_meta(meta)
            similarity = _rd_curve_similarity(
                query_curve, curve
            )
            neighbor_results.append(
                NeighborResult(neighbor, similarity, curve)
            )
        score = _mean_similarity(neighbor_results)
        baseline = _random_clips(
            self.retriever, clip_path, topk
        )
        baseline_score = _mean_similarity(
            [
                NeighborResult(
                    clip,
                    _rd_curve_similarity(
                        query_curve,
                        _rd_curve_from_meta(meta),
                    ),
                    _rd_curve_from_meta(meta),
                )
                for clip, meta in baseline
            ]
        )
        return QueryResult(
            clip_path,
            query_curve,
            neighbor_results,
            score,
            baseline_score,
        )

    def _index_videos(
        self, videos: list[str], bypass_cache: bool = False
    ) -> None:
        """Download+split+index every entry not already recorded in `self.cache_path` (skipped otherwise, unless `bypass_cache`); each entry is marked as indexed - persisted immediately - once all its scenes are recorded, so an interrupted run only redoes the videos it didn't finish."""
        indexed = (
            set()
            if bypass_cache
            else self._load_indexed_videos()
        )
        todo = [v for v in videos if v not in indexed]
        skipped = len(videos) - len(todo)
        if skipped:
            tqdm.write(
                f"[indexing] skipping {skipped} already-indexed video(s)"
            )
        for video in tqdm(
            todo, desc="Indexing videos", unit="video"
        ):
            scenes = list(self._load_scenes([video]))
            for clip in tqdm(
                scenes,
                desc="Indexing scenes",
                unit="scene",
                leave=False,
            ):
                self.retriever.index_scene(clip)
                # ponytail: thumbnails + RD curve are already captured in
                # the clip's metadata (build_scene_meta) - drop the raw
                # encoded clip right away so re-indexing many videos
                # doesn't fill up disk.
                os.remove(clip.path)
            indexed.add(video)
            self._save_indexed_videos(indexed)

    def _load_indexed_videos(self) -> set[str]:
        """Entries previously recorded as indexed in `self.cache_path` - empty if the cache file doesn't exist yet."""
        if not os.path.exists(self.cache_path):
            return set()
        with open(self.cache_path) as f:
            return set(json.load(f))

    def _save_indexed_videos(
        self, videos: set[str]
    ) -> None:
        """Persist `videos` as the indexed set to `self.cache_path`."""
        os.makedirs(
            os.path.dirname(self.cache_path), exist_ok=True
        )
        with open(self.cache_path, "w") as f:
            json.dump(sorted(videos), f)

    def _load_scenes(
        self, videos: list[str]
    ) -> Iterator[SceneClip]:
        """Resolve each entry to a local file via `self.loader` and split it into scenes, without indexing them."""
        pbar = tqdm(
            videos, desc="Downloading videos", unit="video"
        )
        for video in pbar:
            pbar.set_postfix_str(video)
            video_path = self.loader.load(
                video, self.video_dir
            )
            scenes = split_scenes(
                video_path,
                self.scenes_dir,
                max_scenes=self.max_scenes,
            )
            yield from scenes


def _other_clips(
    retriever: RRFRetriever, clip_path: str, topk: int
) -> list[tuple[str, Metadata]]:
    """`clip_path`'s topk fused nearest neighbors (with metadata) via `retriever`, excluding itself."""
    matches = retriever.retrieve(clip_path, topk=topk + 1)
    return [
        (clip, meta)
        for clip, meta, _ in matches
        if clip != clip_path
    ][:topk]


def _random_clips(
    retriever: RRFRetriever, clip_path: str, n: int
) -> list[tuple[str, Metadata]]:
    """`n` random (clip, metadata) pairs from the retriever's index, excluding `clip_path` - a baseline to compare content-similarity neighbors against."""
    collection = retriever.collections[0]
    pool = [
        (cast(str, meta["clip"]), meta)
        for meta, _ in collection.store.list_all(
            collection.namespace
        )
        if meta["clip"] != clip_path
    ]
    return random.sample(pool, min(n, len(pool)))


def _mean_similarity(
    results: list[NeighborResult],
) -> float:
    """Mean `similarity` over `results`, or 0.0 if empty."""
    return (
        sum(r.similarity for r in results) / len(results)
        if results
        else 0.0
    )


def _rd_curve_from_meta(
    meta: Metadata,
) -> list[tuple[int, float]]:
    """Reconstruct a scene's RD curve from its stored metadata (see `build_scene_meta`)."""
    vmafs = cast(list[float], meta["rd_curve"])
    return list(zip(DEFAULT_KBPS_RUNGS, vmafs))


def _rd_curve_similarity(
    a: list[tuple[int, float]], b: list[tuple[int, float]]
) -> float:
    """1 - mean absolute VMAF difference (scaled to [0, 1]) between two same-rung RD curves."""
    diffs = [
        abs(vmaf_a - vmaf_b)
        for (_, vmaf_a), (_, vmaf_b) in zip(a, b)
    ]
    return 1.0 - (sum(diffs) / len(diffs)) / 100.0
