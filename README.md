# intern-video

Re-encode a video to H.264/AAC at a given bitrate, via ffmpeg. Also splits a
video into scenes (PySceneDetect) and encodes each scene as its own clip.

Requires `ffmpeg`/`ffprobe` on `PATH` (e.g. `brew install ffmpeg`) and
`pip install -r requirements.txt` (PySceneDetect, for `encoder.ingest`).

### One-time setup for InternVideo2 embeddings

`embed.py` wraps the vendored InternVideo2-Stage2-1B model
(`vendor/InternVideo`) to embed scenes for content-similarity matching. It
needs its ~2.6GB checkpoint, which isn't in git:
```
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OpenGVLab/InternVideo2-Stage2_1B-224p-f4', filename='InternVideo2-stage2_1b-224p-f4.pt', local_dir='weights')"
```
Only needed if you use `embed.py` or `--embed`; skip it otherwise.

## Usage

```
python -m encoder.encode <video.mp4> <output.mp4> --kbps 2500
```

As a library:
```python
from encoder.encode import encode_video
encode_video("video.mp4", "output.mp4", kbps=2500)
```

### Scene-split ingestion

```
python -m encoder.ingest <video.mp4> <out_dir> --kbps 2500
```
Detects scene cuts, then encodes each scene straight from the source in one
ffmpeg pass (cut + bitrate encode together, no intermediate re-encode).

As a library:
```python
from encoder.ingest import ingest_video
ingest_video("video.mp4", "out_dir", kbps=2500)  # -> list of clip paths
```

Pass `--index <dir>` (or `store=VectorStore(...)` as a library) to also
extract a few thumbnail frames per scene (`<clip>-thumb-N.jpg`, alongside the
clip) and compute its rate-distortion curve, recording the curve in the
vector index under the `rd-curve` namespace:
```
python -m encoder.ingest <video.mp4> <out_dir> --index <index_dir>
```

Add `--embed` (requires `--index`, and the checkpoint from "One-time setup"
above) to also embed each scene with InternVideo2 and record it under the
`internvideo2` namespace, for content-similarity matching against past scenes:
```
python -m encoder.ingest <video.mp4> <out_dir> --index <index_dir> --embed
```

### Rate-distortion curve

```python
from encoder.rd_curve import compute_rd_curve
compute_rd_curve("scene.mp4")  # -> [(500, 0.94), (1000, 0.97), (2000, 0.99), (4000, 0.995), (8000, 0.998)]
```
Encodes the scene at each bitrate rung and measures SSIM against the source
(ffmpeg's built-in `ssim` filter - no external quality model needed). The
SSIM values (not the kbps rungs, which are fixed/known) are what gets stored
as the vector: two scenes that compress similarly land close together, so
matching a new scene against this namespace surfaces past scenes with a
similar bitrate/quality tradeoff.

### InternVideo2 content embedding

```python
from embed import embed_video
embed_video("scene.mp4")  # -> L2-normalized joint video embedding, shape (512,)
```
Wraps the vendored model's `get_vid_feat` - the raw video embedding, with no
text/caption comparison - so it can be dropped into the vector index under
its own namespace and matched against other scenes by content similarity.

### Vector index

Different embedders produce incompatible vector spaces, so the index is
namespaced per embedder/encoding type (one Chroma collection per namespace);
within a namespace, search is Chroma's own nearest-neighbor index.

```python
from vectorstore import VectorStore

store = VectorStore("index")
store.add("clip-embedder", embedding, {"video": "foo.mp4", "scene": 1})
store.search("clip-embedder", query_embedding, topk=5)  # -> [(metadata, similarity), ...]
```

## API

```
python api.py <index_dir_or_url>
```
`<index_dir_or_url>` is either a filesystem path (fine for one-process use,
e.g. running everything sequentially) or an `http(s)://` URL to a running
`chroma run` server (required if ingest and serve run as separate
processes - see "Ingest + serve" below for why).

Serves `GET /videos` at `http://127.0.0.1:8000`, listing every source video
that's been ingested, with its scene clips, a few thumbnail frames, RD
curve, and whether an InternVideo2 embedding was stored. `clip` and
`thumbnails` are URLs under the `/media/` mount (also served by `api.py`,
from `--media-root`, default `scenes` - see "Ingest + serve" below):
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
        "rd_curve": {"kbps": [500, 1000, 2000, 4000, 8000], "ssim": [0.94, 0.97, 0.99, 0.995, 0.998]},
        "has_embedding": true
      }
    ]
  }
]
```

## Frontend

`frontend/` is a separate static project (plain HTML/JS, Chart.js via CDN,
no build step) that fetches from the API above - it's a different origin/
port, so `api.py` enables CORS for it.
```
frontend/serve.sh
```
Serves the UI at `http://127.0.0.1:5500`, listing videos and rendering each
scene's RD curve as a chart. Requires `api.py` (above) running separately.

## Ingest + serve, running both at once

Chroma's on-disk store isn't safe for one process to read while a *different*
process writes to it - it can corrupt the collection it's reading. `chroma
run` (its own server) fixes this by making one process the sole owner of the
store, so run it first:
```
./chroma-server.sh   # owns index/, serves it over HTTP on :8001
./ingest.sh          # picks up the first video in vids/, ingests it with --embed
./serve.sh           # runs the API against chroma-server.sh (frontend/serve.sh separately for the UI)
```
Three separate terminals - `ingest.sh` and `serve.sh` both talk to
`chroma-server.sh` over HTTP, so refresh the frontend as scenes land.

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
