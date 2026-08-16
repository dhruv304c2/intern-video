"""Local-filesystem VideoLoader (see protocol.py) - treats `url` as an already-local file path, no download."""

import os


class LocalVideoLoader:
    def load(self, url: str, out_dir: str) -> str:
        """Return `url` unchanged (it's already a local path) after checking it exists - `out_dir` is unused."""
        if not os.path.isfile(url):
            raise FileNotFoundError(url)
        return url
