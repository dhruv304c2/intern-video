"""Chroma-backed VectorStore (see protocol.py) - one Chroma collection per namespace."""

import uuid
from typing import cast
from urllib.parse import urlparse

import chromadb
import numpy as np
import numpy.typing as npt
from chromadb.api import ClientAPI

from core.vectorstore.protocol import Metadata


class ChromaVectorStore:
    def __init__(self, root: str) -> None:
        """`root` is either a filesystem path (single-process use, e.g.
        tests) or an http(s):// URL pointing at a running `chroma run`
        server. A bare on-disk PersistentClient corrupts itself if a
        collection is read while empty by one process and then written to
        by another - `chroma run` (HttpClient) serializes access through
        one process and doesn't have that problem, so real deployments
        where ingest and serve run as separate processes must use it.
        """
        self._client: ClientAPI
        if root.startswith(("http://", "https://")):
            parsed = urlparse(root)
            self._client = chromadb.HttpClient(
                host=parsed.hostname or "localhost",
                port=parsed.port or 8000,
            )
        else:
            self._client = chromadb.PersistentClient(
                path=root
            )

    def _collection(
        self, namespace: str
    ) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            namespace, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        namespace: str,
        vector: npt.ArrayLike,
        metadata: Metadata,
    ) -> None:
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
        query: npt.ArrayLike,
        topk: int = 5,
    ) -> list[tuple[Metadata, float]]:
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
        # metadatas/distances are None only when excluded from
        # `include` - we never exclude them, so always populated.
        metadatas = result["metadatas"]
        distances = result["distances"]
        assert metadatas is not None
        assert distances is not None
        return [
            (cast(Metadata, meta), 1 - dist)
            for meta, dist in zip(
                metadatas[0], distances[0]
            )
        ]

    def list_all(
        self, namespace: str
    ) -> list[tuple[Metadata, list[float]]]:
        collection = self._collection(namespace)
        if collection.count() == 0:
            return []
        result = collection.get(
            include=["metadatas", "embeddings"]
        )
        # metadatas/embeddings are None only when excluded from
        # `include` - we never exclude them, so always populated.
        metadatas = result["metadatas"]
        embeddings = result["embeddings"]
        assert metadatas is not None
        assert embeddings is not None
        return [
            (
                cast(Metadata, meta),
                [float(v) for v in vector],
            )
            for meta, vector in zip(metadatas, embeddings)
        ]
