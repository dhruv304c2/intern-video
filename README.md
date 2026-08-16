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
`core/encode.py`/`core/ingest.py` as a library without a retriever.

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

`core/ingest.py` is a pure library (`split_scenes`) - `main.py` is the CLI
entry point: it builds an `RRFRetriever` (see "Multi-collection retrieval"
below) over an `"internvideo2"` collection against
`ChromaVectorStore(".cache/index")`, then runs `RDRecallTest` (see
"RD-curve recall" below) over an index/query CSV pair of YouTube URLs and
prints the mean score - `RDRecallTest` parses both CSVs, then downloads,
splits, and indexes the index-set URLs, and downloads, splits (but doesn't
index) the query-set URLs:
```
python main.py <index_videos.csv> <query_videos.csv> --max-scenes 50
```
`--max-scenes` (default 50) caps how many scenes each video is split into -
see `split_scenes` below. Or, against the sample sets in `datasets/`:
```
./run-recall.sh                 # MAX_SCENES defaults to 50
MAX_SCENES=100 ./run-recall.sh  # override via env var
./run-recall.sh --bypass-cache  # extra args are forwarded to main.py
```
Detects scene cuts, then encodes each scene straight from the source in one
ffmpeg pass (cut + bitrate encode together, no intermediate re-encode), then
extracts a few thumbnail frames per scene (`<clip>-thumb-N.jpg`, alongside
the clip) and embeds it with `InternVideo2Embedder` (needs the checkpoint
from "One-time setup" above) for lookups against past scenes.

As a library (indexing is optional - skip the `retriever.index_scene` loop to
just split scenes):
```python
from core.ingest import split_scenes

clips = split_scenes(
    "video.mp4", "out_dir", kbps=2500
)  # -> list of SceneClip
```
With your own collections (see "Collections" and "Multi-collection
retrieval" below):
```python
from core.embedder import InternVideo2Embedder, RdCurveEmbedder
from core.indexing import Collection
from core.retrieval import RRFRetriever
from core.vectorstore import ChromaVectorStore
from core.ingest import split_scenes

store = ChromaVectorStore("index_dir")
internvideo2 = InternVideo2Embedder()
rd_curve = RdCurveEmbedder()
retriever = RRFRetriever(
    [
        Collection.symmetric("internvideo2", internvideo2, store),
        Collection.symmetric("rd-curve", rd_curve, store),
    ]
)
for clip in split_scenes("video.mp4", "out_dir"):
    retriever.index_scene(clip)
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
(below) wraps this as a mock collection for lookups against past scenes'
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
index under its own namespace (see "Collections" below) and matched against other
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

### Collections

`core/indexing/` ties a namespace to the embedders and store its vectors
belong to - a `Collection(namespace, index_embedder, query_embedder,
store)` - so ingestion can index a scene into any number of these
collections without knowing which embedders or stores they use. Separate
index/query embedders support asymmetric encoders (e.g. a dual-encoder with
distinct passage/query models); `Collection.symmetric(namespace, embedder,
store)` fills both with the same embedder, for the common case where
indexing and querying share an embedding space, as `InternVideo2Embedder`
and `RdCurveEmbedder` do here. `Collection.record(clip_path, metadata)`
embeds (with the index embedder) and stores in one call:

```python
from core.embedder import InternVideo2Embedder, RdCurveEmbedder
from core.indexing import Collection
from core.vectorstore import ChromaVectorStore

store = ChromaVectorStore("index_dir")
internvideo2 = InternVideo2Embedder()
rd_curve = RdCurveEmbedder()
collections = [
    Collection.symmetric("internvideo2", internvideo2, store),
    Collection.symmetric("rd-curve", rd_curve, store),
]
collections[0].record("scene.mp4", {"scene": 1})  # embeds + stores under "internvideo2"
```
Collections can share a store (as above, one Chroma index with two
namespaces) or use different stores entirely - each collection only needs
to support `record()`/`search()` independently.

### Multi-collection retrieval

`core/retrieval.py` defines `RRFRetriever(collections)` - the single
entry point for indexing into, and searching across, multiple collections
at once, so callers don't loop over `collections` by hand:

```python
from core.retrieval import RRFRetriever

retriever = RRFRetriever(collections)
retriever.index_scene(clip)  # records clip into every collection
```
`retrieve()` searches every collection and fuses their rankings via
[Reciprocal Rank
Fusion](https://en.wikipedia.org/wiki/Learning_to_rank#Reciprocal_rank_fusion) -
`score(clip) = sum(1 / (RRF_K + rank + 1))` over every collection the clip
appears in - so a clip that's a strong match in either an `"internvideo2"`
or `"rd-curve"` search (or both) ranks highly, without needing to normalize
similarity scores across the two (incompatible) vector spaces:

```python
retriever.retrieve(
    "scene.mp4", topk=5
)  # -> [(clip_path, metadata, fused_score), ...] sorted descending
```

### RD-curve recall

`recall/rd_recall.py` defines `RDRecallTest(retriever, loader)`: does
content-similarity retrieval also surface RD-curve-similar scenes?
`run(index_csv, query_csv)` parses two CSVs of video URLs (one per row,
first column): the index set is downloaded, split into scenes, and indexed
into `retriever` (a private `_index_videos()` step - each URL is recorded
in `cache_path` (default `.cache/recall/indexed_urls.json`) once indexed,
so a later `run()` skips re-downloading/re-splitting/re-indexing it; pass
`bypass_cache=True` - or `./run-recall.sh --bypass-cache` - to force a
full re-index); the query set is
downloaded and split but never indexed, so a query's neighbors are always
scenes from a *different* video, never its own - keeping index and query
disjoint. For each query scene clip, `score()` retrieves its topk
content-similarity neighbors (via `retriever.retrieve`, self excluded) and
compares each neighbor's rate-distortion curve against the query's,
scoring by 1 minus their mean absolute VMAF difference. A neighbor's curve
comes from its stored metadata (`build_scene_meta` precomputes `rd_curve`
at index time), not by re-encoding its clip file - which may already be
gone, since every indexed clip's raw `.mp4` is deleted right after
indexing (thumbnails and the RD curve are kept, everything the report and
scoring need). A query clip is only deleted after it's scored, once its
own thumbnails have been extracted. `run()` also writes an HTML report
(thumbnails + RD-curve comparison per query/neighbor pair - see
`recall/report.py`) to `report_path` (default
`.cache/recall/report.html`). Build `retriever` from content collections
only (e.g. just `"internvideo2"`, not `"rd-curve"`), or RD-curve
similarity would leak into neighbor selection:

```python
from core.loader import YtDlpLoader
from recall.rd_recall import RDRecallTest

test = RDRecallTest(retriever, YtDlpLoader())
test.score("scene.mp4", topk=5)  # -> mean RD-curve similarity to its topk content neighbors, for an already-indexed clip
test.run("index_videos.csv", "query_videos.csv", topk=5)  # -> parses both CSVs, indexes the first, scores the second, mean over every query scene
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
python tests/test_recall.py
```

## Quality gate

`.githooks/pre-commit` runs `ruff format --check`, `ruff check`, and
`basedpyright` (type checker - config in `pyproject.toml`) over
`core/`/`main.py`/`tests/`, and blocks the commit if any of them fail.
Opt in once per clone:
```
git config core.hooksPath .githooks
```
