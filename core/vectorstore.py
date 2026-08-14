"""On-disk vector index, namespaced per embedder, backed by Chroma.

Different encoders/embedders produce incompatible vector spaces (different
dims, different meanings) - so "multiple levels of indexing" means: namespace
by embedder/encoding type first (one Chroma collection per namespace), then
nearest-neighbor search within that namespace (Chroma's own ANN index). A new
scene is matched by embedding it with a given embedder and searching only
that embedder's namespace/collection.
"""

import uuid
from urllib.parse import urlparse

import chromadb
import numpy as np


class VectorStore:
    def __init__(self, root: str):
        """`root` is either a filesystem path (single-process use, e.g.
        tests) or an http(s):// URL pointing at a running `chroma run`
        server. A bare on-disk PersistentClient corrupts itself if a
        collection is read while empty by one process and then written to
        by another - `chroma run` (HttpClient) serializes access through
        one process and doesn't have that problem, so real deployments
        where ingest and serve run as separate processes must use it.
        """
        if root.startswith(("http://", "https://")):
            parsed = urlparse(root)
            self._client = chromadb.HttpClient(
                host=parsed.hostname, port=parsed.port
            )
        else:
            self._client = chromadb.PersistentClient(
                path=root
            )

    def _collection(self, namespace: str):
        return self._client.get_or_create_collection(
            namespace, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        namespace: str,
        vector: np.ndarray,
        metadata: dict,
    ) -> None:
        """Append one embedding + its metadata to `namespace`'s index."""
        vector = np.asarray(
            vector, dtype=np.float32
        ).reshape(-1)
        self._collection(namespace).add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector.tolist()],
            metadatas=[metadata],
        )

    def search(
        self,
        namespace: str,
        query: np.ndarray,
        topk: int = 5,
    ) -> list[tuple[dict, float]]:
        """Return up to `topk` (metadata, cosine_similarity) pairs for `namespace`, best match first."""
        query = np.asarray(query, dtype=np.float32).reshape(
            -1
        )
        collection = self._collection(namespace)
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=[query.tolist()],
            n_results=min(topk, collection.count()),
        )
        return [
            (meta, 1 - dist)
            for meta, dist in zip(
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def list_all(
        self, namespace: str
    ) -> list[tuple[dict, list[float]]]:
        """Return every (metadata, vector) pair stored in `namespace`, in no particular order."""
        collection = self._collection(namespace)
        if collection.count() == 0:
            return []
        result = collection.get(
            include=["metadatas", "embeddings"]
        )
        return [
            (meta, vector.tolist())
            for meta, vector in zip(
                result["metadatas"], result["embeddings"]
            )
        ]
