"""Namespaced nearest-neighbor vector index - backend-agnostic contract.

Different encoders/embedders produce incompatible vector spaces (different
dims, different meanings) - so "multiple levels of indexing" means: namespace
by embedder/encoding type first (one collection per namespace), then
nearest-neighbor search within that namespace. A new scene is matched by
embedding it with a given embedder and searching only that embedder's
namespace/collection.
"""

from typing import Protocol

import numpy.typing as npt

# backend-agnostic value union - deliberately narrower than any one
# backend's own metadata type (e.g. chromadb.Metadata), so swapping the
# VectorStore implementation can't leak a backend-specific type here.
MetadataValue = (
    str
    | int
    | float
    | bool
    | list[str | int | float | bool]
)
Metadata = dict[str, MetadataValue]


class VectorStore(Protocol):
    def add(
        self,
        namespace: str,
        vector: npt.ArrayLike,
        metadata: Metadata,
    ) -> None:
        """Append one embedding + its metadata to `namespace`'s index."""
        ...

    def search(
        self,
        namespace: str,
        query: npt.ArrayLike,
        topk: int = 5,
    ) -> list[tuple[Metadata, float]]:
        """Return up to `topk` (metadata, cosine_similarity) pairs for `namespace`, best match first."""
        ...

    def list_all(
        self, namespace: str
    ) -> list[tuple[Metadata, list[float]]]:
        """Return every (metadata, vector) pair stored in `namespace`, in no particular order."""
        ...
