"""InternVideo2-Stage2 embedder (CPU/MPS) - see core/embedder/protocol.py.

Wraps the vendored official model code
(vendor/InternVideo/InternVideo2/multi_modality) instead of reimplementing
it. Returns the raw L2-normalized joint video embedding
(model.get_vid_feat), so it can be dropped straight into a VectorStore
under its own namespace and matched against other videos by content
similarity. Supports both vendored InternVideo2-Stage2 sizes via
`variant` - see `InternVideo2Embedder._VARIANTS`.
"""

import argparse
import contextlib
import io
import logging
import os
import sys
import warnings
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import torch

REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
VENDOR_DIR = os.path.join(
    REPO_ROOT,
    "vendor",
    "InternVideo",
    "InternVideo2",
    "multi_modality",
)
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")
_ORIG_CWD = os.getcwd()

# the vendored code logs/prints noise for things we deliberately don't need
# (flash_attn, deepspeed - CUDA-only, unused on CPU/MPS) and emits harmless
# FutureWarnings from its pinned timm/torch API usage - silence both.
warnings.filterwarnings("ignore", category=FutureWarning)
logging.disable(logging.WARNING)

# the vendored config loader resolves paths (e.g. configs/config_bert_large.json)
# relative to CWD, so it must run from inside the vendored tree. Resolve any
# user-supplied relative paths against _ORIG_CWD before this.
os.chdir(VENDOR_DIR)
sys.path.insert(0, VENDOR_DIR)

with contextlib.redirect_stdout(io.StringIO()):
    # only importable after the sys.path.insert above - not a real
    # static package, so the type checker can't resolve it.
    from demo.utils import (  # pyright: ignore[reportMissingImports]
        _frame_from_video,
        frames2tensor,
        setup_internvideo2,
    )
    from demo_config import (  # pyright: ignore[reportMissingImports]
        Config,
        eval_dict_leaf,
    )


class InternVideo2Embedder:
    # (vision_encoder name, checkpoint filename under weights/) per size -
    # both dispatch through the same vendored model_cls, see
    # models/internvideo2_stage2_visual.py's encoder_name branch.
    _VARIANTS: ClassVar[dict[str, tuple[str, str]]] = {
        "1b": (
            "pretrain_internvideo2_1b_patch14_224",
            "InternVideo2-stage2_1b-224p-f4.pt",
        ),
        "6b": (
            "pretrain_internvideo2_6b_patch14_224",
            "internvideo2-s2_6b-224p-f4_with_audio_encoder.pt",
        ),
    }

    def __init__(self, variant: str = "1b") -> None:
        if variant not in self._VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r} - choose from {sorted(self._VARIANTS)}"
            )
        self._variant = variant
        # vendored, unstubbed types - Any is honest here, not a shortcut.
        self._model: Any = None
        self._config: Any = None

    def _device(self) -> str:
        # ponytail: try MPS first on Apple Silicon, fall back to cpu if unsupported
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        encoder_name, checkpoint_name = self._VARIANTS[
            self._variant
        ]
        weights_path = os.path.join(
            WEIGHTS_DIR, checkpoint_name
        )
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Missing checkpoint at {weights_path}. See README.md for download steps."
            )
        config = Config.from_file("demo/pipeline_config.py")
        config = eval_dict_leaf(config)
        config.device = self._device()
        config.pretrained_path = weights_path
        config.model.vision_encoder.name = encoder_name
        with contextlib.redirect_stdout(io.StringIO()):
            self._model, _ = setup_internvideo2(config)
        self._config = config

    def embed(
        self, video_path: str
    ) -> npt.NDArray[np.float32]:
        """Return InternVideo2's L2-normalized joint video embedding for `video_path`."""
        self._load()
        import cv2

        video = cv2.VideoCapture(video_path)
        frames = list(_frame_from_video(video))
        video.release()

        fn = self._config.get("num_frames", 8)
        size_t = self._config.get("size_t", 224)
        device = torch.device(self._config.device)
        # ponytail: frames2tensor asserts len(frames) >= fn - an ultra-short
        # scene clip can have fewer decoded frames than that. Pad by
        # repeating the last frame rather than rejecting the clip.
        if frames and len(frames) < fn:
            frames = frames + [frames[-1]] * (
                fn - len(frames)
            )
        frames_tensor = frames2tensor(
            frames,
            fnum=fn,
            target_size=(size_t, size_t),
            device=device,
        )
        vid_feat = self._model.get_vid_feat(frames_tensor)
        return vid_feat.detach().cpu().numpy().reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", help="path to a video file"
    )
    args = parser.parse_args()
    video_path = os.path.join(_ORIG_CWD, args.video)

    vector = InternVideo2Embedder().embed(video_path)
    print(
        f"embedding shape: {vector.shape}, norm: {np.linalg.norm(vector):.4f}"
    )


if __name__ == "__main__":
    main()
