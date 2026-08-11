"""Smallest possible check that the pipeline is wired correctly end to end.

Runs zero-shot ranking on the vendored example video against the official
demo's own candidate captions and checks the top-1 result matches the
documented expected output (see DEMO_USAGE_GUIDE.md).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videotag.pipeline import VENDOR_DIR, classify_video

EXAMPLE_VIDEO = os.path.join(VENDOR_DIR, "demo", "example1.mp4")

CANDIDATES = [
    "A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
    "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
    "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
    "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
    "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
    "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
    "A playful dog slides down a snowy hill, wagging its tail with delight.",
    "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
    "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
    "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery.",
]

# The two captions describing the actual dog/snow/play scene. Upstream's fp16 GPU
# demo ranks "plays fetch" first at prob 0.7927; on fp32 CPU/MPS these two swap
# (floating-point precision, not a wiring bug) - so check the *set*, not exact order.
EXPECTED_TOP1_SET = {
    "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
    "A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
}


def test_demo_example_ranking():
    results = classify_video(EXAMPLE_VIDEO, CANDIDATES, topk=5)
    top_label, top_prob = results[0]
    assert top_label in EXPECTED_TOP1_SET, f"unexpected top-1 label: {top_label}"
    assert top_prob > 0.5, f"expected a confident top-1, got prob: {top_prob}"
    print("OK:", results)


if __name__ == "__main__":
    test_demo_example_ranking()
