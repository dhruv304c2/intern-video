"""Smallest check that Embedder.embed returns a real, deterministic, unit-norm vector."""

import os

import _path
import numpy as np

from core.embedder import InternVideo2Embedder

FIXTURE = os.path.join(
    _path.ROOT,
    "vendor",
    "InternVideo",
    "InternVideo2",
    "multi_modality",
    "demo",
    "example1.mp4",
)


def test_embed_video_is_deterministic_unit_vector() -> None:
    embedder = InternVideo2Embedder()
    v1 = embedder.embed(FIXTURE)
    v2 = embedder.embed(FIXTURE)

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
