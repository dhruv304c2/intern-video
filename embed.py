"""Video content embedding via InternVideo2-Stage2-1B (CPU/MPS).

Wraps the vendored official model code
(vendor/InternVideo/InternVideo2/multi_modality) instead of reimplementing
it. Returns the raw L2-normalized joint video embedding
(model.get_vid_feat) - no text/caption comparison, unlike the old
classify_video - so it can be dropped straight into vectorstore.VectorStore
under its own namespace and matched against other videos by content
similarity.
"""

import argparse
import contextlib
import io
import logging
import os
import sys
import warnings

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(
    REPO_ROOT,
    "vendor",
    "InternVideo",
    "InternVideo2",
    "multi_modality",
)
WEIGHTS_PATH = os.path.join(
    REPO_ROOT,
    "weights",
    "InternVideo2-stage2_1b-224p-f4.pt",
)
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
    from demo.utils import (
        _frame_from_video,
        frames2tensor,
        setup_internvideo2,
    )
    from demo_config import (
        Config,
        eval_dict_leaf,
    )

_model = None
_config = None


def _device() -> str:
    # ponytail: try MPS first on Apple Silicon, fall back to cpu if unsupported
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load():
    global _model, _config
    if _model is not None:
        return
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Missing checkpoint at {WEIGHTS_PATH}. See README.md for download steps."
        )
    config = Config.from_file("demo/pipeline_config.py")
    config = eval_dict_leaf(config)
    config.device = _device()
    config.pretrained_path = WEIGHTS_PATH
    with contextlib.redirect_stdout(io.StringIO()):
        _model, _ = setup_internvideo2(config)
    _config = config


def embed_video(video_path: str) -> np.ndarray:
    """Return InternVideo2's L2-normalized joint video embedding for `video_path`."""
    _load()
    import cv2

    video = cv2.VideoCapture(video_path)
    frames = list(_frame_from_video(video))
    video.release()

    fn = _config.get("num_frames", 8)
    size_t = _config.get("size_t", 224)
    device = torch.device(_config.device)
    frames_tensor = frames2tensor(
        frames,
        fnum=fn,
        target_size=(size_t, size_t),
        device=device,
    )
    vid_feat = _model.get_vid_feat(frames_tensor)
    return vid_feat.detach().cpu().numpy().reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", help="path to a video file"
    )
    args = parser.parse_args()
    video_path = os.path.join(_ORIG_CWD, args.video)

    vector = embed_video(video_path)
    print(
        f"embedding shape: {vector.shape}, norm: {np.linalg.norm(vector):.4f}"
    )


if __name__ == "__main__":
    main()
