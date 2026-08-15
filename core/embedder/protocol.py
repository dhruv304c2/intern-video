"""Content embedder - backend-agnostic contract for scoring a video clip into a vector for content-similarity indexing."""

from typing import Protocol

import numpy as np
import numpy.typing as npt


class Embedder(Protocol):
    def embed(
        self, video_path: str
    ) -> npt.NDArray[np.float32]:
        """Return a content embedding vector for the video at `video_path`."""
        ...
