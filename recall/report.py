"""HTML report for RDRecallTest: thumbnails + RD-curve comparison per query/neighbor pair."""

import base64
import html
import os
from typing import NamedTuple

RDCurve = list[tuple[int, float]]


class NeighborResult(NamedTuple):
    clip: str
    similarity: float
    curve: RDCurve


class QueryResult(NamedTuple):
    clip: str
    curve: RDCurve
    neighbors: list[NeighborResult]
    score: float
    baseline_score: float


def build_report(
    results: list[QueryResult], out_path: str
) -> None:
    """Render `results` (one per scored query clip) as a self-contained HTML file at `out_path`. `baseline_score` (mean RD-curve similarity to `topk` random indexed clips, instead of content-similarity neighbors) is shown alongside `score` so the report shows whether content similarity beats picking neighbors at random."""
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
    body = "".join(_query_section(r) for r in results)
    doc = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>RDRecallTest report</title><style>{_CSS}</style>"
        "</head><body>"
        "<h1>RDRecallTest report</h1>"
        f"<p>Mean score: <strong>{mean_score:.4f}</strong> vs "
        f"random-neighbor baseline: <strong>{mean_baseline:.4f}</strong> "
        f"over {len(results)} scene(s)</p>"
        f"{body}</body></html>"
    )
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(doc)


def _query_section(r: QueryResult) -> str:
    neighbors_html = "".join(
        _clip_card(n.clip, n.curve, n.similarity)
        for n in r.neighbors
    )
    return (
        '<section class="query"><h2>'
        f"{html.escape(os.path.basename(r.clip))} — score "
        f"{r.score:.4f} (baseline {r.baseline_score:.4f})</h2>"
        '<div class="row">'
        '<div class="col"><h3>Query</h3>'
        f"{_clip_card(r.clip, r.curve, None, is_query=True)}</div>"
        '<div class="col"><h3>Content-similarity neighbors</h3>'
        f'<div class="neighbors">{neighbors_html}</div></div>'
        "</div></section>"
    )


def _clip_card(
    clip: str,
    curve: RDCurve,
    similarity: float | None,
    is_query: bool = False,
) -> str:
    label = (
        "query"
        if similarity is None
        else f"similarity {similarity:.4f}"
    )
    card_class = "card query-card" if is_query else "card"
    return (
        f'<div class="{card_class}">'
        f'<img src="{_thumbnail_uri(clip)}">'
        f'<div class="label">{html.escape(os.path.basename(clip))}</div>'
        f'<div class="label">{label}</div>'
        f"{_curve_table(curve)}</div>"
    )


def _curve_table(curve: RDCurve) -> str:
    rows = "".join(
        f"<tr><td>{kbps}</td><td>{vmaf:.1f}</td></tr>"
        for kbps, vmaf in curve
    )
    return (
        '<table class="curve"><tr><th>kbps</th><th>VMAF</th></tr>'
        f"{rows}</table>"
    )


def _thumbnail_uri(clip_path: str) -> str:
    """Middle extracted thumbnail for `clip_path` (see `core.indexing.meta.extract_thumbnails`), inlined as a base64 data URI so the report is portable/self-contained."""
    stem = os.path.splitext(clip_path)[0]
    with open(f"{stem}-thumb-2.jpg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


_CSS = (
    "body{font-family:sans-serif;margin:2rem}"
    ".row{display:flex;gap:1.5rem;align-items:flex-start}"
    ".col{padding:.75rem;border-radius:8px}"
    ".col:first-child{background:#eef4ff;border:1px solid #bcd4f6}"
    ".col:last-child{border-left:3px solid #ddd;padding-left:1.25rem}"
    ".col h3{margin:0 0 .5rem;font-size:.85rem;color:#333;text-transform:uppercase;letter-spacing:.03em}"
    ".neighbors{display:flex;gap:1rem;flex-wrap:wrap}"
    ".card{border:1px solid #ccc;border-radius:8px;padding:.5rem;width:160px;background:#fff}"
    ".card.query-card{border:2px solid #3b6fd6;box-shadow:0 0 0 3px #dce8fc}"
    ".card img{width:100%;border-radius:4px}"
    ".label{font-size:.8rem;color:#555}"
    "table.curve{width:100%;font-size:.75rem;margin-top:.25rem}"
    "section.query{margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:2px solid #eee}"
    "section.query h2{margin-bottom:1rem}"
)
