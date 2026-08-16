"""Build a content-similarity retriever and run RDRecallTest over an index/query CSV pair of video URLs."""

import argparse

from core.embedder import InternVideo2Embedder
from core.indexing import Collection
from core.loader import YtDlpLoader
from core.retrieval import RRFRetriever
from core.vectorstore import ChromaVectorStore
from recall.rd_recall import RDRecallTest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "index_csv",
        help="CSV of video URLs to index (one per row, first column)",
    )
    parser.add_argument(
        "query_csv",
        help="CSV of video URLs to query against the index (one per row, first column)",
    )
    parser.add_argument(
        "--bypass-cache",
        action="store_true",
        help="re-download+re-split+re-index every index_csv video, even ones already indexed in a prior run",
    )
    args = parser.parse_args()

    store = ChromaVectorStore(".cache/index")
    retriever = RRFRetriever(
        [
            Collection.symmetric(
                namespace="internvideo2",
                embedder=InternVideo2Embedder(),
                store=store,
            )
        ]
    )

    score = RDRecallTest(retriever, YtDlpLoader()).run(
        args.index_csv,
        args.query_csv,
        bypass_cache=args.bypass_cache,
    )
    print(f"RDRecallTest mean score: {score:.4f}")


if __name__ == "__main__":
    main()
