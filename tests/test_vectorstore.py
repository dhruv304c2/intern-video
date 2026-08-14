"""Smallest check that VectorStore namespaces embedders apart and finds the closest match."""

import tempfile

import _path  # noqa: F401
import numpy as np

from core.vectorstore import VectorStore


def test_search_returns_closest_match_within_namespace():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(tmp)
        store.add(
            "clip-embedder",
            np.array([1.0, 0.0]),
            {"scene": "a"},
        )
        store.add(
            "clip-embedder",
            np.array([0.0, 1.0]),
            {"scene": "b"},
        )
        store.add(
            "other-embedder",
            np.array([0.0, 1.0]),
            {"scene": "c"},
        )

        results = store.search(
            "clip-embedder", np.array([0.9, 0.1]), topk=2
        )

        assert results[0][0]["scene"] == "a"
        assert results[0][1] > results[1][1]
        assert all(
            scene != "c"
            for scene, _ in [
                (m["scene"], s) for m, s in results
            ]
        )
        print("OK:", results)


def test_search_on_empty_namespace_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert (
            VectorStore(tmp).search(
                "nothing-here", np.array([1.0, 0.0])
            )
            == []
        )


if __name__ == "__main__":
    test_search_returns_closest_match_within_namespace()
    test_search_on_empty_namespace_returns_empty()
