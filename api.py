"""JSON API listing ingested videos and their scene data.

Reads whatever `encoder.ingest` has written into a vector store (see
vectorstore.VectorStore): the "rd-curve" namespace (always present when
ingestion was run with --index) and the "internvideo2" namespace (present
only when ingestion also used --embed). Groups scene entries by their
source video for display. The frontend/ directory is a separate project
that consumes this API over HTTP - see frontend/app.js.
"""

import argparse
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from vectorstore import VectorStore


def build_videos(
    store: VectorStore, media_root: str
) -> list[dict]:
    """Group every ingested scene by source video, joining in its RD curve and embedding status.

    `media_root` is the directory clips/thumbnails were written under (see
    encoder.ingest) - clip/thumbnail paths are rewritten relative to it, as
    URLs under the /media/ mount `create_app` serves that same directory at.
    """

    def media_url(path: str) -> str | None:
        rel = os.path.relpath(path, media_root)
        return (
            None
            if rel.startswith("..")
            else f"/media/{rel}"
        )

    embedded_clips = {
        meta["clip"]
        for meta, _ in store.list_all("internvideo2")
    }

    videos: dict[str, dict] = {}
    for meta, vector in store.list_all("rd-curve"):
        video = videos.setdefault(
            meta["source_video"],
            {
                "source_video": meta["source_video"],
                "scenes": [],
            },
        )
        video["scenes"].append(
            {
                "scene": meta["scene"],
                "clip": media_url(meta["clip"]),
                "start": meta["start"],
                "end": meta["end"],
                "thumbnails": [
                    media_url(t)
                    for t in meta.get("thumbnails", [])
                ],
                "rd_curve": {
                    "kbps": meta["kbps_rungs"],
                    "ssim": vector,
                },
                "has_embedding": meta["clip"]
                in embedded_clips,
            }
        )

    result = sorted(
        videos.values(), key=lambda v: v["source_video"]
    )
    for video in result:
        video["scenes"].sort(key=lambda s: s["scene"])
    return result


def create_app(
    store: VectorStore, media_root: str = "scenes"
) -> FastAPI:
    media_root = os.path.abspath(media_root)
    app = FastAPI()
    # frontend/ is a separate project served from its own origin/port.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
    )

    @app.get("/videos")
    def list_videos():
        return build_videos(store, media_root)

    # check_dir=False: media_root may not exist yet on a fresh
    # checkout, before the first ingest.sh run.
    app.mount(
        "/media",
        StaticFiles(directory=media_root, check_dir=False),
    )
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "index",
        help="vector store root to read from - a filesystem "
        "path, or an http(s):// URL to a running `chroma run` "
        "server (required if ingest runs as a separate process)",
    )
    parser.add_argument(
        "--media-root",
        default="scenes",
        help="directory ingest wrote clips/thumbnails under (see ingest.sh)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(
        f"API: http://{args.host}:{args.port} "
        "(run frontend/serve.sh separately for the UI)",
        flush=True,
    )

    app = create_app(
        VectorStore(args.index), args.media_root
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
