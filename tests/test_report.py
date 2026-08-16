"""Smallest check for the HTML report builder: writes a file with the mean score and clip names."""

import os
import tempfile

import _path  # noqa: F401

from recall.report import (
    NeighborResult,
    QueryResult,
    build_report,
)


def test_build_report_writes_html_with_score_and_clips() -> (
    None
):
    with tempfile.TemporaryDirectory() as tmp:
        query_clip = os.path.join(tmp, "query.mp4")
        neighbor_clip = os.path.join(tmp, "neighbor.mp4")
        for clip in (query_clip, neighbor_clip):
            open(
                f"{os.path.splitext(clip)[0]}-thumb-2.jpg",
                "w",
            ).close()

        result = QueryResult(
            clip=query_clip,
            curve=[(500, 80.0), (1000, 90.0)],
            neighbors=[
                NeighborResult(
                    neighbor_clip,
                    0.9,
                    [(500, 78.0), (1000, 88.0)],
                )
            ],
            score=0.9,
        )

        out_path = os.path.join(tmp, "report.html")
        build_report([result], out_path)

        with open(out_path) as f:
            html = f.read()
        assert "0.9000" in html
        assert "query.mp4" in html
        assert "neighbor.mp4" in html
        print(
            "OK: report contains mean score and clip names"
        )


if __name__ == "__main__":
    test_build_report_writes_html_with_score_and_clips()
