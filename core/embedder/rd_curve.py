"""Mock embedder wrapping the rate-distortion curve - see core/embedder/protocol.py.

Not a real content embedding - just fits compute_rd_curve (core/rd_curve.py)
into the Embedder protocol so a scene's VMAF-vs-bitrate curve can be
recorded as its own collection (see core/indexing/) alongside real embedders like
InternVideo2Embedder.
"""

import numpy as np
import numpy.typing as npt

from core.rd_curve import (
    DEFAULT_KBPS_RUNGS,
    compute_rd_curve,
)


class RdCurveEmbedder:
    def __init__(
        self, kbps_rungs: list[int] = DEFAULT_KBPS_RUNGS
    ) -> None:
        self._kbps_rungs = kbps_rungs

    def embed(
        self, video_path: str
    ) -> npt.NDArray[np.float32]:
        """Return the VMAF score at each of this embedder's kbps rungs, as a vector."""
        curve = compute_rd_curve(
            video_path, self._kbps_rungs
        )
        return np.array(
            [vmaf for _, vmaf in curve], dtype=np.float32
        )
