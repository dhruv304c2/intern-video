"""Build the indexing/retrieval pipeline and run ingestion through it."""

import argparse

from core.ingest import ingest_video
from core.retrieval import build_pipeline
from core.vectorstore import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", help="path to a source video file"
    )
    args = parser.parse_args()

    indexer, _ = build_pipeline(
        ChromaVectorStore(".cache/index")
    )
    outputs = ingest_video(
        args.video, ".cache/scenes", indexer=indexer
    )
    print(
        f"wrote {len(outputs)} scene clips to .cache/scenes"
    )


if __name__ == "__main__":
    main()
