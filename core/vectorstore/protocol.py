"""Namespaced nearest-neighbor vector index - backend-agnostic contract.

Different encoders/embedders produce incompatible vector spaces (different
dims, different meanings) - so "multiple levels of indexing" means: namespace
by embedder/encoding type first (one collection per namespace), then
nearest-neighbor search within that namespace. A new scene is matched by
embedding it with a given embedder and searching only that embedder's
namespace/collection.
"""

from typing import Protocol

import chromadb
import numpy.typing as npt


class VectorStore(Protocol):
    def add(
        self,
        namespace: str,
        vector: npt.ArrayLike,
        metadata: chromadb.Metadata,
    ) -> None:
        """Append one embedding + its metadata to `namespace`'s index."""
        ...

    def search(
        self,
        namespace: str,
        query: npt.ArrayLike,
        topk: int = 5,
    ) -> list[tuple[chromadb.Metadata, float]]:
        """Return up to `topk` (metadata, cosine_similarity) pairs for `namespace`, best match first."""
        ...

    def list_all(
        self, namespace: str
    ) -> list[tuple[chromadb.Metadata, list[float]]]:
        """Return every (metadata, vector) pair stored in `namespace`, in no particular order."""
        ...
