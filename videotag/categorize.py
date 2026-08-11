"""Structured wrapper around bitrate_categories.estimate_bitrate."""
from dataclasses import dataclass

from videotag.bitrate_categories import estimate_bitrate


@dataclass
class Categorization:
    category_id: int
    category: str
    motion: str
    tier: str
    preferred_bitrate_kbps: int
    kbps_range: tuple[int, int]
    ranked: list[tuple[str, float]]


def categorize(video_path: str, resolution: str = "1080p", topk: int = 3) -> Categorization:
    """Classify `video_path` into a bitrate category for `resolution`."""
    result = estimate_bitrate(video_path, resolution=resolution, topk=topk)
    lo, hi = result["kbps_range"]
    return Categorization(
        category_id=result["category_id"],
        category=result["category"],
        motion=result["motion"],
        tier=result["tier"],
        preferred_bitrate_kbps=round((lo + hi) / 2),
        kbps_range=result["kbps_range"],
        ranked=result["ranked"],
    )
