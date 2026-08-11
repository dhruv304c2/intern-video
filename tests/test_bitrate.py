"""Smallest check that bitrate estimation is wired correctly end to end."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videotag.bitrate_categories import RESOLUTION_BASE_KBPS, TIER_MULTIPLIER, estimate_bitrate
from videotag.pipeline import VENDOR_DIR

EXAMPLE_VIDEO = os.path.join(VENDOR_DIR, "demo", "example1.mp4")


def test_bitrate_estimate_shape():
    result = estimate_bitrate(EXAMPLE_VIDEO, resolution="1080p", topk=3)
    assert result["tier"] in TIER_MULTIPLIER
    lo, hi = result["kbps_range"]
    assert 0 < lo < hi
    assert result["ranked"][0][0] == result["category"]
    print("OK:", result)


def test_unknown_resolution_rejected():
    try:
        estimate_bitrate(EXAMPLE_VIDEO, resolution="8k")
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_bitrate_estimate_shape()
    test_unknown_resolution_rejected()
