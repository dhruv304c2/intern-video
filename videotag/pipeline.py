"""Zero-shot video tagging with InternVideo2-Stage2-1B.

Wraps the vendored official demo code (vendor/InternVideo/InternVideo2/multi_modality)
instead of reimplementing the model. See README.md for one-time setup.
"""
import argparse
import contextlib
import io
import logging
import os
import sys
import warnings

import cv2
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_DIR = os.path.join(REPO_ROOT, "vendor", "InternVideo", "InternVideo2", "multi_modality")
WEIGHTS_PATH = os.path.join(REPO_ROOT, "weights", "InternVideo2-stage2_1b-224p-f4.pt")
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
    from demo_config import Config, eval_dict_leaf  # noqa: E402
    from demo.utils import _frame_from_video, retrieve_text, setup_internvideo2  # noqa: E402

_model = None
_tokenizer = None
_config = None


def _device() -> str:
    # ponytail: try MPS first on Apple Silicon, fall back to cpu if unsupported
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load():
    global _model, _tokenizer, _config
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
        _model, _tokenizer = setup_internvideo2(config)
    _config = config


def classify_video(video_path: str, labels: list[str], topk: int = 5) -> list[tuple[str, float]]:
    """Rank candidate labels/captions by zero-shot similarity to a video."""
    _load()
    video = cv2.VideoCapture(video_path)
    frames = list(_frame_from_video(video))
    video.release()
    topk = min(topk, len(labels))
    texts, probs = retrieve_text(
        frames, labels, model=_model, topk=topk, config=_config, device=torch.device(_config.device)
    )
    return list(zip(texts, probs.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--labels", help="path to a text file, one candidate label/caption per line")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--bitrate", action="store_true", help="estimate bitrate tier instead of ranking --labels")
    parser.add_argument("--resolution", default="1080p", help="target resolution for --bitrate, e.g. 720p, 1080p")
    args = parser.parse_args()
    args.video = os.path.join(_ORIG_CWD, args.video)

    if args.bitrate:
        from videotag.bitrate_categories import estimate_bitrate

        result = estimate_bitrate(args.video, resolution=args.resolution, topk=args.topk)
        lo, hi = result["kbps_range"]
        print(f"tier: {result['tier']} (motion={result['motion']})")
        print(f"recommended bitrate @ {args.resolution}: {lo}-{hi} kbps")
        print(f"matched category: {result['category']}")
        return

    if not args.labels:
        parser.error("--labels is required unless --bitrate is set")
    args.labels = os.path.join(_ORIG_CWD, args.labels)

    with open(args.labels) as f:
        labels = [line.strip() for line in f if line.strip()]

    for label, prob in classify_video(args.video, labels, args.topk):
        print(f"text: {label} ~ prob: {prob:.4f}")


if __name__ == "__main__":
    main()
