# intern-video

Re-encode a video to H.264/AAC at a given bitrate, via ffmpeg. Also splits a
video into scenes (PySceneDetect) and encodes each scene as its own clip.

Requires `ffmpeg`/`ffprobe` on `PATH` (e.g. `brew install ffmpeg`) and
`pip install -r requirements.txt` (PySceneDetect, for `core/ingest.py`/`main.py`).

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
Only needed if you use `core/embedder/` directly, or `main.py` (which
always indexes, using the `"1b"` variant); skip it if you're only using
`core/encode.py`/`core/ingest.py` as a library without an `indexer`.

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

`core/ingest.py` is a pure library (`split_scenes`, `ingest_video`) -
`main.py` is the CLI entry point: it builds the default `Indexer`/`Retriever`
pair (see "Pipeline" below) against `ChromaVectorStore(".cache/index")` and
runs ingestion through it, writing scene clips to `.cache/scenes`:
```
python main.py <video.mp4>
```
Detects scene cuts, then encodes each scene straight from the source in one
ffmpeg pass (cut + bitrate encode together, no intermediate re-encode), then
extracts a few thumbnail frames per scene (`<clip>-thumb-N.jpg`, alongside
the clip) and, for each ann, embeds the scene and records it under that
ann's namespace, for lookups against past scenes: `"internvideo2"`
(`InternVideo2Embedder` - needs the checkpoint from "One-time setup" above)
and `"rd-curve"` (`RdCurveEmbedder` - see "Content embedders" below).

As a library (`indexer` is optional - omit it to skip indexing):
```python
from core.ingest import ingest_video

ingest_video(
    "video.mp4", "out_dir", kbps=2500
)  # -> list of clip paths
```
With your own anns (an `core.indexing.Indexer(anns)` - see "Anns" below):
```python
from core.embedder import InternVideo2Embedder, RdCurveEmbedder
from core.indexing import Ann, Indexer
from core.vectorstore import ChromaVectorStore
from core.ingest import ingest_video

store = ChromaVectorStore("index_dir")
ingest_video(
    "video.mp4",
    "out_dir",
    indexer=Indexer(
        [
            Ann("internvideo2", InternVideo2Embedder(), store),
            Ann("rd-curve", RdCurveEmbedder(), store),
        ]
    ),
)
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
`--enable-libvmaf`, no separate quality model to install). `RdCurveEmbedder`
(below) wraps this as a mock ann for lookups against past scenes'
rate-distortion shape.

### Content embedders

`core/embedder/` defines the `Embedder` protocol (one method: `embed(video_path) ->
vector`) and two implementations:

```python
from core.embedder import InternVideo2Embedder, RdCurveEmbedder

InternVideo2Embedder().embed(
    "scene.mp4"
)  # -> L2-normalized joint video embedding, shape (512,)

InternVideo2Embedder(variant="6b").embed(
    "scene.mp4"
)  # same shape, larger backbone - needs the 6b checkpoint (see "One-time setup")

RdCurveEmbedder().embed(
    "scene.mp4"
)  # -> VMAF at each of DEFAULT_KBPS_RUNGS, shape (5,)
```
`InternVideo2Embedder` wraps the vendored model's `get_vid_feat` - the raw
video embedding, with no text/caption comparison. `RdCurveEmbedder` is a
mock embedder: it just runs `compute_rd_curve` (above) and returns the VMAF
values as the vector, so a scene's rate-distortion curve can be indexed the
same way as a real content embedding. Either can be dropped into the vector
index under its own namespace (see "Anns" below) and matched against other
scenes by similarity.

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

### Anns

`core/indexing/` ties a namespace to the embedder and store its vectors
belong to - an `Ann(namespace, embedder, store)` - so ingestion can index a
scene into any number of these "ann tables" without knowing which embedders
or stores they use. `Ann.record(clip_path, metadata)` embeds and stores in
one call:

```python
from core.embedder import InternVideo2Embedder, RdCurveEmbedder
from core.indexing import Ann
from core.vectorstore import ChromaVectorStore

store = ChromaVectorStore("index_dir")
anns = [
    Ann("internvideo2", InternVideo2Embedder(), store),
    Ann("rd-curve", RdCurveEmbedder(), store),
]
anns[0].record("scene.mp4", {"scene": 1})  # embeds + stores under "internvideo2"
```
Anns can share a store (as above, one Chroma index with two namespaces) or
use different stores entirely - an `Indexer` (below) only cares that each
ann can `record()` a clip.

### Pipeline

`Indexer(anns)`/`Retriever(anns)` (`core/indexing/index.py`,
`core/retrieval.py`) so callers don't pass `anns` around by hand:
`indexer.index_scene(clip)` / `retriever.retrieve(clip_path, topk=5)`.
`build_pipeline(store)` builds the default `Indexer`/`Retriever` pair off
`core.indexing.default_anns(store)`, sharing the same anns so what's indexed
is exactly what's searched:

```python
from core.retrieval import build_pipeline
from core.vectorstore import ChromaVectorStore
from core.ingest import ingest_video

store = ChromaVectorStore("index_dir")
indexer, retriever = build_pipeline(store)
outputs = ingest_video("video.mp4", "out_dir", indexer=indexer)
retriever.retrieve(outputs[0], topk=5)  # -> {"internvideo2": [...], "rd-curve": [...]}
```

## Cleanup

```
./clean-db.sh   # wipes .cache/index/ and .cache/scenes/
```

## Test

```
python tests/test_encode.py
python tests/test_ingest.py
python tests/test_vectorstore.py
python tests/test_rd_curve.py
python tests/test_embed.py
python tests/test_retrieval.py
```

## Quality gate

`.githooks/pre-commit` runs `ruff format --check`, `ruff check`, and
`basedpyright` (type checker - config in `pyproject.toml`) over
`core/`/`main.py`/`tests/`, and blocks the commit if any of them fail.
Opt in once per clone:
```
git config core.hooksPath .githooks
```
