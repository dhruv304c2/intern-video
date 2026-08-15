"""An ann: a namespace paired with the embedder and store its vectors belong to."""

from typing import NamedTuple

from core.embedder import Embedder
from core.vectorstore import VectorStore


class Ann(NamedTuple):
    namespace: str
    embedder: Embedder
    store: VectorStore
