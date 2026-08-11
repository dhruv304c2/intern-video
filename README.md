# InternVideo2 zero-shot video tagging (CPU / Apple Silicon)

Zero-shot video classification/tagging using InternVideo2-Stage2-1B: give it a
video and a list of candidate labels/captions, get back a similarity-ranked
top-k. Runs on CPU or Apple Silicon (MPS) — no CUDA required.

This vendors the official model code (`vendor/InternVideo/InternVideo2/multi_modality`,
from https://github.com/OpenGVLab/InternVideo) rather than reimplementing the
model, and patches a handful of CUDA-only unconditional imports + two upstream
relative-import bugs so it runs on CPU/MPS. `videotag/pipeline.py` is a thin
wrapper around the official `demo/utils.py::retrieve_text`.

## Project layout

```
videotag/            reusable modules (import as e.g. `from videotag.pipeline import classify_video`)
  pipeline.py           classify_video() - the core zero-shot ranking call
  bitrate_categories.py motion caption ensemble -> bitrate tier/range
  categorize.py         Categorization dataclass wrapping bitrate_categories
  source_bitrate.py     actual bitrate of a source file via ffprobe packets ("ground truth")
  split_scenes.py       PySceneDetect scene splitting, with manifest caching
  report.py             ReportEntry + build_html_report()
tests/                test_pipeline.py, test_bitrate.py
build_report.py       orchestrates: split vids/ -> classify scenes/ -> report.html
run.sh                entry point for build_report.py
vids/                 put source videos here
scenes/               generated scene clips + cache files (safe to delete to force a re-run)
```

## One-time setup

1. Create a venv (Python 3.13; the ML stack doesn't yet have wheels for 3.14) and install deps:
   ```
   python3.13 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Pre-warm the BERT tokenizer cache (the vendored code loads it with
   `local_files_only=True`, no network fallback):
   ```
   python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-large-uncased')"
   ```
3. Get the InternVideo2-Stage2-1B checkpoint (gated, auto-approved):
   - Log in: `huggingface-cli login`
   - Visit https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4 and
     accept the access form once (instant approval).
   - Download it:
     ```
     python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OpenGVLab/InternVideo2-Stage2_1B-224p-f4', filename='InternVideo2-stage2_1b-224p-f4.pt', local_dir='weights')"
     ```

## Usage

```
python -m videotag.pipeline <video.mp4> --labels labels.txt --topk 5
```
`labels.txt` is one candidate label/caption per line. Prints `label ~ prob` lines.

As a library:
```python
from videotag.pipeline import classify_video
classify_video("video.mp4", ["a cat", "a dog", "a car"], topk=3)
```

### Bitrate estimation

```
python -m videotag.pipeline <video.mp4> --bitrate --resolution 1080p
```
Ranks the video against a set of motion captions (2-3 phrasings per tier, see
`videotag/bitrate_categories.py`), then maps the winning tier
(`low`/`medium`/`high`, from static/moderate/high motion) to a recommended
kbps range for the target resolution. The kbps ranges are heuristic reference
points (roughly Apple HLS authoring spec / common streaming ladders) scaled
by tier, not measured against real encodes — treat it as a starting point
for a bitrate ladder, not a substitute for VMAF-based per-title tuning. See
`CATEGORIZATION_FINDINGS.pdf` for how this caption scheme was arrived at.

As a library:
```python
from videotag.bitrate_categories import estimate_bitrate
estimate_bitrate("video.mp4", resolution="720p")
# {'category': ..., 'motion': 'moderate', 'tier': 'medium', 'kbps_range': (2000, 3000), 'ranked': [...]}
```

### Full report: split -> classify -> HTML

Drop source video(s) into `vids/`, then:
```
./run.sh
```
This splits each video into scenes (PySceneDetect), classifies every scene's
bitrate category, and writes `report.html` with:
- a legend of the available motion-tier captions,
- a chart of predicted bitrate vs. the *actual* bitrate of the source video
  over time (measured from real packet sizes via `ffprobe` — see
  `videotag/source_bitrate.py` — so you can see how well the prediction
  tracks reality),
- one card per scene with sample frames, category, predicted/actual bitrate,
  and classification latency.

Both scene splitting and per-scene classification are cached to disk under
`scenes/` (a `*.manifest.json` for splits, `*.cache.json` + JPEGs per clip)
so re-running `./run.sh` after tweaking the report itself doesn't re-run
PySceneDetect or the model. Run `./run.sh --clean` to wipe `scenes/` and
force a fully fresh split + classify.

## Verify the setup

```
python tests/test_pipeline.py
python tests/test_bitrate.py
```
Runs the ranking against the vendored example video and checks it matches the
documented reference output, and checks bitrate estimation returns a
well-formed tier/range.

## Performance

1B ViT + BERT-large in fp32 on CPU/MPS — expect tens of seconds per video, not
real-time. Fine for tagging a handful of videos; batch a large corpus overnight.

## What was patched in the vendored code, and why

- `models/backbones/internvideo2/{internvideo2,internvl_clip_vision,internvideo2_clip_vision}.py`:
  wrapped unconditional `flash_attn` imports in try/except — `FlashAttention`
  is only instantiated when `use_flash_attn=True`, which our config sets to
  `False` (fp32 CPU/MPS), so the package is never actually needed.
- `models/__init__.py`: made its eager imports of CLIP/audiovisual model
  variants (which pull in `flash_attn`/LLaMA/mobileclip chains we don't use)
  best-effort instead of hard failures.
- `models/criterions.py`: fixed `from ..utils.distributed import ...` /
  `from ..utils.easydict import ...` to absolute imports (`from utils....`) —
  this is the exact fix documented in the upstream `DEMO_USAGE_GUIDE.md` for
  running outside their package-install path.
- `models/backbones/bert/xbert.py`: `apply_chunking_to_forward`,
  `find_pruneable_heads_and_indices`, `prune_linear_layer` moved from
  `transformers.modeling_utils` to `transformers.pytorch_utils` in newer
  `transformers` — updated the import location (also pinned `transformers<5`,
  since `find_pruneable_heads_and_indices` was removed entirely in v5).
- `demo/pipeline_config.py`: copy of `demo/internvideo2_stage2_config.py` with
  `device`/`pretrained_path` set for local eval, `use_checkpoint`/
  `gradient_checkpointing`/`deepspeed.enable` off (training-only features), and
  `vision_encoder.pretrained=None` — the original path is a placeholder
  (`'your_model_path/1B_stage2_pt.pth'`) for a redundant intermediate
  vision-only load; the full fused checkpoint is loaded afterward via
  `pretrained_path` anyway.
- `demo/utils.py`: swapped the vendored custom `BertTokenizer` subclass for
  stock `transformers.BertTokenizer` — the vendored one predates transformers'
  internal tokenizer refactor (`self.vocab`/`get_vocab` no longer compatible);
  stock `BertTokenizer` is the same WordPiece tokenizer and works with the
  same `vocab.txt`.
- `demo/utils.py::frames2tensor`: fixed `.to(device).float()` ordering to
  `.float().to(device)` — moving a float64 numpy tensor to an MPS device
  before casting throws (MPS doesn't support float64); this is a genuine
  upstream bug, not just a CPU workaround.
