"""Builds a self-contained HTML report of bitrate categorization results.

One card per video/scene: sample frames (so a reader can eyeball scene
complexity against the predicted category), the predicted category/tier/
bitrate, and classification latency. Also includes a legend of the available
categories, and a chart of predicted vs. actual (source) bitrate over time.
"""
import base64
import html
from dataclasses import dataclass

from videotag.bitrate_categories import CATEGORIES, MOTION_LEVELS, SCORE_TO_TIER
from videotag.categorize import Categorization

_STYLE = """
body { font-family: -apple-system, sans-serif; margin: 2rem; background: #111; color: #eee; }
.card { background: #1c1c1c; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }
.frames { display: flex; gap: 0.5rem; margin: 0.5rem 0; }
.frames img { height: 160px; border-radius: 4px; }
.meta { display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.9rem; }
.meta span { background: #2a2a2a; padding: 0.25rem 0.6rem; border-radius: 4px; }
h1 { font-size: 1.4rem; } h2 { margin: 0 0 0.5rem; font-size: 1.1rem; word-break: break-all; }
table { border-collapse: collapse; margin: 1rem 0 2rem; font-size: 0.85rem; }
th, td { border: 1px solid #333; padding: 0.4rem 0.6rem; text-align: left; }
th { background: #222; }
"""


@dataclass
class ReportEntry:
    video_name: str
    categorization: Categorization
    frames: list[bytes]  # raw JPEG bytes, for embedding as data: URIs
    latency_seconds: float
    start_seconds: float
    duration_seconds: float
    actual_bitrate_kbps: float
    source_video: str


def _frame_tags(frames: list[bytes]) -> str:
    return "".join(
        f'<img src="data:image/jpeg;base64,{base64.b64encode(f).decode("ascii")}">' for f in frames
    )


def _card(entry: ReportEntry) -> str:
    c = entry.categorization
    lo, hi = c.kbps_range
    return f"""
<div class="card">
  <h2>{html.escape(entry.video_name)}</h2>
  <div class="frames">{_frame_tags(entry.frames)}</div>
  <div class="meta">
    <span>source: {html.escape(entry.source_video)}</span>
    <span>t={entry.start_seconds:.1f}s (+{entry.duration_seconds:.1f}s)</span>
    <span>category #{c.category_id}: {html.escape(c.category)}</span>
    <span>motion: {html.escape(c.motion)}</span>
    <span>tier: {html.escape(c.tier)}</span>
    <span>predicted bitrate: {c.preferred_bitrate_kbps} kbps ({lo}-{hi})</span>
    <span>actual (source) bitrate: {entry.actual_bitrate_kbps:.0f} kbps</span>
    <span>latency: {entry.latency_seconds:.2f}s</span>
  </div>
</div>"""


def _legend() -> str:
    rows = "".join(
        f"<tr><td>{i}</td><td>{html.escape(caption)}</td>"
        f"<td>{MOTION_LEVELS[m]}</td><td>{SCORE_TO_TIER[m]}</td></tr>"
        for i, (caption, m) in enumerate(CATEGORIES.items())
    )
    return f"""
<h1>Available categories</h1>
<table>
<tr><th>id</th><th>caption</th><th>motion</th><th>tier</th></tr>
{rows}
</table>"""


def _category_distribution(entries: list[ReportEntry]) -> str:
    """Count of scenes per category, as a simple bar-per-row table."""
    if not entries:
        return ""
    counts = {caption: 0 for caption in CATEGORIES}
    for e in entries:
        counts[e.categorization.category] = counts.get(e.categorization.category, 0) + 1
    max_count = max(counts.values()) or 1
    rows = "".join(
        f"<tr><td>{html.escape(caption)}</td><td>{count}</td>"
        f'<td style="width:400px"><div style="background:#5aa9ff;height:14px;border-radius:3px;'
        f'width:{count / max_count * 100:.0f}%;"></div></td></tr>'
        for caption, count in counts.items()
    )
    return f"""
<h1>Category distribution</h1>
<table>
<tr><th>caption</th><th>count</th><th></th></tr>
{rows}
</table>"""


def _svg_bitrate_chart(entries: list[ReportEntry], width: int = 900, height: int = 260) -> str:
    """Predicted vs. actual bitrate over time, as overlapping filled areas in one inline SVG chart."""
    if not entries:
        return ""
    ordered = sorted(entries, key=lambda e: e.start_seconds)
    max_t = max(e.start_seconds + e.duration_seconds for e in ordered) or 1
    max_kbps = max(
        max(e.categorization.preferred_bitrate_kbps, e.actual_bitrate_kbps) for e in ordered
    ) * 1.1 or 1
    left, right, top, bottom = 45, 15, 25, 25
    baseline_y = height - bottom

    def point(t: float, kbps: float) -> str:
        x = left + (t / max_t) * (width - left - right)
        y = height - bottom - (kbps / max_kbps) * (height - top - bottom)
        return f"{x:.1f},{y:.1f}"

    def line_points(kbps_of) -> str:
        return " ".join(point(e.start_seconds + e.duration_seconds / 2, kbps_of(e)) for e in ordered)

    def area_points(kbps_of) -> str:
        line = line_points(kbps_of)
        first_x = line.split(" ")[0].split(",")[0]
        last_x = line.split(" ")[-1].split(",")[0]
        return f"{first_x},{baseline_y:.1f} {line} {last_x},{baseline_y:.1f}"

    pred_pts = line_points(lambda e: e.categorization.preferred_bitrate_kbps)
    actual_pts = line_points(lambda e: e.actual_bitrate_kbps)
    pred_area = area_points(lambda e: e.categorization.preferred_bitrate_kbps)
    actual_area = area_points(lambda e: e.actual_bitrate_kbps)
    return f"""
<h1>Predicted vs. actual bitrate over time</h1>
<svg width="{width}" height="{height}" style="background:#1c1c1c;border-radius:8px;">
  <text x="{left}" y="16" fill="#5aa9ff" font-size="12">predicted</text>
  <text x="120" y="16" fill="#ff8a5a" font-size="12">actual (source)</text>
  <text x="{left}" y="{height - 6}" fill="#888" font-size="11">0s</text>
  <text x="{width - right - 30}" y="{height - 6}" fill="#888" font-size="11">{max_t:.0f}s</text>
  <text x="2" y="{top + 5}" fill="#888" font-size="11">{max_kbps:.0f}</text>
  <text x="2" y="{height - bottom}" fill="#888" font-size="11">0</text>
  <polygon points="{actual_area}" fill="#ff8a5a" fill-opacity="0.25" stroke="none"/>
  <polygon points="{pred_area}" fill="#5aa9ff" fill-opacity="0.25" stroke="none"/>
  <polyline points="{actual_pts}" fill="none" stroke="#ff8a5a" stroke-width="2"/>
  <polyline points="{pred_pts}" fill="none" stroke="#5aa9ff" stroke-width="2"/>
</svg>"""


def build_html_report(entries: list[ReportEntry], out_path: str) -> None:
    """Write an HTML report for `entries` to `out_path`."""
    total_latency = sum(e.latency_seconds for e in entries)
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Bitrate classification report</title>
<style>{_STYLE}</style></head>
<body>
<h1>Bitrate classification report</h1>
<p>{len(entries)} clips &middot; total classification latency {total_latency:.2f}s</p>
{_svg_bitrate_chart(entries)}
{_legend()}
{_category_distribution(entries)}
{"".join(_card(e) for e in entries)}
</body></html>"""
    with open(out_path, "w") as f:
        f.write(html_doc)
