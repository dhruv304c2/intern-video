"""`recall` subcommand: build a content-similarity retriever and run RDRecallTest, either against YouTube URLs (yt) or local video files (local)."""

import argparse
import csv
import glob
import os

from core.embedder import InternVideo2Embedder
from core.embedder.internvideo2 import _ORIG_CWD
from core.indexing import Collection
from core.loader import (
    LocalVideoLoader,
    VideoLoader,
    YtDlpLoader,
)
from core.retrieval import RRFRetriever
from core.vectorstore import ChromaVectorStore
from recall.rd_recall import RDRecallTest

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def add_recall_parser(
    subcommands: argparse._SubParsersAction,  # pyright: ignore[reportPrivateUsage]
) -> None:
    """Attach the `recall {yt,local}` subcommand tree to `subcommands`."""
    recall = subcommands.add_parser(
        "recall", help="run RDRecallTest"
    ).add_subparsers(dest="mode", required=True)

    yt = recall.add_parser(
        "yt",
        help="index/query videos downloaded from YouTube URLs listed in CSVs",
    )
    yt.add_argument(
        "index_csv",
        help="CSV of video URLs to index (one per row, first column)",
    )
    yt.add_argument(
        "query_csv",
        help="CSV of video URLs to query against the index (one per row, first column)",
    )
    _add_common_args(yt)
    yt.set_defaults(run=_run_yt)

    local = recall.add_parser(
        "local",
        help="index/query video files already on local disk",
    )
    local.add_argument(
        "index_dir",
        help="directory of local video files to index",
    )
    local.add_argument(
        "query_dir",
        help="directory of local video files to query against the index",
    )
    _add_common_args(local)
    local.set_defaults(run=_run_local)


def _add_common_args(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--bypass-cache",
        action="store_true",
        help="re-download+re-split+re-index every index video, even ones already indexed in a prior run",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=50,
        help="cap on scenes split+encoded per video (default: 50)",
    )


def _run_yt(args: argparse.Namespace) -> float:
    return _run_recall(
        YtDlpLoader(),
        _csv_column(args.index_csv),
        _csv_column(args.query_csv),
        args,
    )


def _run_local(args: argparse.Namespace) -> float:
    return _run_recall(
        LocalVideoLoader(),
        _videos_in_dir(args.index_dir),
        _videos_in_dir(args.query_dir),
        args,
    )


def _run_recall(
    loader: VideoLoader,
    index_videos: list[str],
    query_videos: list[str],
    args: argparse.Namespace,
) -> float:
    test = RDRecallTest(
        _build_retriever(),
        loader,
        max_scenes=args.max_scenes,
    )
    return test.run(
        index_videos,
        query_videos,
        bypass_cache=args.bypass_cache,
    )


def _build_retriever() -> RRFRetriever:
    store = ChromaVectorStore(".cache/index")
    return RRFRetriever(
        [
            Collection.symmetric(
                namespace="internvideo2",
                embedder=InternVideo2Embedder(),
                store=store,
            )
        ]
    )


def _resolve(path: str) -> str:
    """Resolve a user-supplied CLI path against the original cwd (see core.embedder.internvideo2)."""
    return os.path.join(_ORIG_CWD, path)


def _csv_column(csv_path: str) -> list[str]:
    """Every row's first column (a video URL) in `csv_path`."""
    with open(_resolve(csv_path), newline="") as f:
        return [row[0] for row in csv.reader(f) if row]


def _videos_in_dir(dir_path: str) -> list[str]:
    """Every video file directly inside `dir_path` (see `VIDEO_EXTENSIONS`), sorted."""
    dir_path = _resolve(dir_path)
    paths = [
        p
        for ext in VIDEO_EXTENSIONS
        for p in glob.glob(
            os.path.join(dir_path, f"*{ext}")
        )
    ]
    return sorted(paths)
