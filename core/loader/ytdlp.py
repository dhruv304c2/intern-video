"""yt-dlp-backed VideoLoader (see protocol.py) - downloads a YouTube (or any other yt-dlp-supported site) video to local disk.

Caches by URL under `out_dir` - a repeat `load()` call for the same URL and
`out_dir` returns the previously downloaded file directly, no network call.
"""

import glob
import hashlib
import os

import yt_dlp


class YtDlpLoader:
    def load(self, url: str, out_dir: str) -> str:
        """Download `url` into `out_dir` via yt-dlp and return the downloaded file's local path."""
        os.makedirs(out_dir, exist_ok=True)
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        cached = glob.glob(
            os.path.join(out_dir, f"{key}.*")
        )
        if cached:
            return cached[0]

        opts = {
            "outtmpl": os.path.join(
                out_dir, f"{key}.%(ext)s"
            ),
            "format": "mp4/bestvideo+bestaudio/best",
            "quiet": True,
            "noprogress": True,
        }
        with yt_dlp.YoutubeDL(
            opts  # pyright: ignore[reportArgumentType]
        ) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
