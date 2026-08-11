"""Actual per-scene bitrate of a source video, via ffprobe packet sizes.

This is the "ground truth" the report compares predictions against: the
average bitrate the source was actually encoded at over a given time range,
computed from real packet sizes rather than an encoder's reported average.
"""
import json
import subprocess


def packet_sizes(video_path: str) -> list[tuple[float, int]]:
    """Return (pts_seconds, size_bytes) for every video packet in `video_path`."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,size", "-of", "json", video_path,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    packets = json.loads(out)["packets"]
    return [(float(p["pts_time"]), int(p["size"])) for p in packets if "pts_time" in p]


def actual_bitrate_kbps(packets: list[tuple[float, int]], start: float, end: float) -> float:
    """Average bitrate of packets with pts in [start, end) seconds."""
    total_bits = sum(size for pts, size in packets if start <= pts < end) * 8
    duration = max(end - start, 1e-6)
    return total_bits / duration / 1000
