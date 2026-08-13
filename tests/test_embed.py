"""Smallest check that embed_video returns a real, deterministic, unit-norm vector."""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
)

from embed import embed_video

FIXTURE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
    "vendor",
    "InternVideo",
    "InternVideo2",
    "multi_modality",
    "demo",
    "example1.mp4",
)


def test_embed_video_is_deterministic_unit_vector():
    v1 = embed_video(FIXTURE)
    v2 = embed_video(FIXTURE)

    assert v1.shape == v2.shape
    assert v1.size > 0
    assert np.all(np.isfinite(v1))
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-3
    print(
        f"OK: embedding shape {v1.shape}, norm {np.linalg.norm(v1):.4f}"
    )


if __name__ == "__main__":
    test_embed_video_is_deterministic_unit_vector()
