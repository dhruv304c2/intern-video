# intern-video

Re-encode a video to H.264/AAC at a given bitrate, via ffmpeg. Also splits a
video into scenes (PySceneDetect) and encodes each scene as its own clip.

Requires `ffmpeg`/`ffprobe` on `PATH` (e.g. `brew install ffmpeg`) and
`pip install -r requirements.txt` (PySceneDetect, for `ingest.py`).

### One-time setup for InternVideo2 embeddings

`core/embedder/internvideo2.py` wraps the vendored InternVideo2-Stage2 model
(`vendor/InternVideo`) to embed scenes for content-similarity matching. It
needs a checkpoint, which isn't in git - which one depends on the
`InternVideo2Embedder(variant=...)` you use (defaults to `"1b"`):
```
# variant="1b" (default) - ~2.6GB
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OpenGVLab/InternVideo2-Stage2_1B-224p-f4', filename='InternVideo2-stage2_1b-224p-f4.pt', local_dir='weights')"

# variant="6b" - ~29GB, only needed if you construct InternVideo2Embedder("6b")
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OpenGVLab/InternVideo2-Stage2_6B-224p-f4', filename='internvideo2-s2_6b-224p-f4_with_audio_encoder.pt', local_dir='weights')"
```
Only needed if you use `core/embedder/` directly, or `ingest.py` with
`--index` (which always uses the `"1b"` variant); skip it if you're only
using `core/encode.py`/`ingest.py` without `--index`.

## Usage

```
python -m core.encode <video.mp4> <output.mp4> --kbps 2500
```

As a library:
```python
from core.encode import encode_video

encode_video("video.mp4", "output.mp4", kbps=2500)
```

### Scene-split ingestion

```
python ingest.py <video.mp4> <out_dir> --kbps 2500
```
Detects scene cuts, then encodes each scene straight from the source in one
ffmpeg pass (cut + bitrate encode together, no intermediate re-encode).

As a library:
```python
from ingest import ingest_video

ingest_video(
    "video.mp4", "out_dir", kbps=2500
)  # -> list of clip paths
```

Pass `--index <dir>` (or `store=VectorStore(...)` as a library) to also
extract a few thumbnail frames per scene (`<clip>-thumb-N.jpg`, alongside the
clip), compute its rate-distortion curve, and embed it with InternVideo2 (the
only embedder - needs the checkpoint from "One-time setup" above) -
recording the curve under the `rd-curve` namespace and the embedding under
`internvideo2`, for content-similarity matching against past scenes:
```
python ingest.py <video.mp4> <out_dir> --index <index_dir>
```

### Rate-distortion curve

```python
from core.rd_curve import compute_rd_curve

compute_rd_curve(
    "scene.mp4"
)  # -> [(500, 82.1), (1000, 91.4), (2000, 96.8), (4000, 98.9), (8000, 99.6)]
```
Encodes the scene at each bitrate rung and measures VMAF against the source
(ffmpeg's `libvmaf` filter - requires an ffmpeg build with
`--enable-libvmaf`, no separate quality model to install). This is recorded
per scene for display (see "API" below) - it's not used for
content-similarity matching. InternVideo2 (below) is the only embedding used
for that.

### InternVideo2 content embedding

`core/embedder/` defines the `Embedder` protocol (one method: `embed(video_path) ->
vector`) and `InternVideo2Embedder`, its only implementation:

```python
from core.embedder import InternVideo2Embedder

InternVideo2Embedder().embed(
    "scene.mp4"
)  # -> L2-normalized joint video embedding, shape (512,)

InternVideo2Embedder(variant="6b").embed(
    "scene.mp4"
)  # same shape, larger backbone - needs the 6b checkpoint (see "One-time setup")
```
Wraps the vendored model's `get_vid_feat` - the raw video embedding, with no
text/caption comparison - so it can be dropped into the vector index under
its own namespace and matched against other scenes by content similarity.

### Vector index

Different embedders produce incompatible vector spaces, so the index is
namespaced per embedder/encoding type (one Chroma collection per namespace);
within a namespace, search is Chroma's own nearest-neighbor index.
`core/vectorstore/` defines the `VectorStore` protocol and `ChromaVectorStore`,
its only implementation:

```python
from core.vectorstore import ChromaVectorStore

store = ChromaVectorStore("index")
store.add(
    "clip-embedder",
    embedding,
    {"video": "foo.mp4", "scene": 1},
)
store.search(
    "clip-embedder", query_embedding, topk=5
)  # -> [(metadata, similarity), ...]
```

## API

```
python -m core.api <index_dir_or_url>
```
`<index_dir_or_url>` is either a filesystem path (fine for one-process use,
e.g. running everything sequentially) or an `http(s)://` URL to a running
`chroma run` server (required if ingest and serve run as separate
processes - see "Ingest + serve" below for why).

Serves `GET /videos` at `http://127.0.0.1:8000`, listing every source video
that's been ingested, with its scene clips, a few thumbnail frames, RD
curve, and its InternVideo2 embedding. `clip` and
`thumbnails` are URLs under the `/media/` mount (also served by
`core/api.py`, from `--media-root`, default `scenes` - see "Ingest + serve"
below):
```json
[
  {
    "source_video": "foo",
    "scenes": [
      {
        "scene": 1,
        "clip": "/media/foo-Scene-001.mp4",
        "start": 0.0,
        "end": 2.0,
        "thumbnails": ["/media/foo-Scene-001-thumb-1.jpg", "/media/foo-Scene-001-thumb-2.jpg", "/media/foo-Scene-001-thumb-3.jpg"],
        "rd_curve": {"kbps": [500, 1000, 2000, 4000, 8000], "vmaf": [82.1, 91.4, 96.8, 98.9, 99.6]},
        "embedding": [0.0123, -0.0456, ...],
        "has_embedding": true
      }
    ]
  }
]
```

## Frontend

`frontend/` is a separate static project (plain HTML/JS, Chart.js via CDN,
no build step) that fetches from the API above - it's a different origin/
port, so `core/api.py` enables CORS for it.
```
frontend/serve.sh start
```
Serves the UI at `http://127.0.0.1:5500`, listing videos and rendering each
scene's RD curve as a chart. Requires `api.py` (above) running separately.

## Ingest + serve, running both at once

Chroma's on-disk store isn't safe for one process to read while a *different*
process writes to it - it can corrupt the collection it's reading. `chroma
run` (its own server) fixes this by making one process the sole owner of the
store, so start it first:
```
./chroma-server.sh start   # owns index/, serves it over HTTP on :8001
./ingest.sh start          # picks up the first video in vids/, ingests + embeds it
./serve.sh start           # runs the API against chroma-server.sh (frontend/serve.sh start separately for the UI)
```
`ingest.sh` and `serve.sh` both talk to `chroma-server.sh` over HTTP, so
refresh the frontend as scenes land.

Every script above (plus `frontend/serve.sh`) takes `start`/`stop` -
`start` backgrounds the process and writes its pid to `.pids/<name>.pid`
(logs alongside it at `.pids/<name>.log`); `stop` reads that pid file and
kills it. `start` refuses to double-launch if one is already running;
`stop` on an already-stopped one is a no-op.
```
./serve.sh stop
./ingest.sh stop
./chroma-server.sh stop
```

To start over, stop `chroma-server.sh` first (it holds `index/` open), then:
```
./clean-db.sh   # wipes index/ and scenes/; refuses to run while chroma-server.sh is up
```

## Test

```
python tests/test_encode.py
python tests/test_ingest.py
python tests/test_vectorstore.py
python tests/test_rd_curve.py
python tests/test_embed.py
python tests/test_api.py
```

## Quality gate

`.githooks/pre-commit` runs `ruff format --check`, `ruff check`, and
`basedpyright` (type checker - config in `pyproject.toml`) over
`core/`/`ingest.py`/`tests/`, and blocks the commit if any of them fail.
Opt in once per clone:
```
git config core.hooksPath .githooks
```
