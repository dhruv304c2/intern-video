"""Video loader - backend-agnostic contract for fetching a video from a remote source onto local disk."""

from typing import Protocol


class VideoLoader(Protocol):
    def load(self, url: str, out_dir: str) -> str:
        """Download the video at `url` into `out_dir` and return its local file path."""
        ...
