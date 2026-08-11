"""Split videos in vids/ into scenes, classify each scene's bitrate category, and build an HTML report."""
import argparse
import json
import os
import shutil
import time
from dataclasses import asdict

import cv2

from videotag.categorize import Categorization, categorize
from videotag.report import ReportEntry, build_html_report
from videotag.source_bitrate import actual_bitrate_kbps, packet_sizes
from videotag.split_scenes import split_into_scenes

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDS_DIR = os.path.join(REPO_ROOT, "vids")
SCENES_DIR = os.path.join(REPO_ROOT, "scenes")
REPORT_PATH = os.path.join(REPO_ROOT, "report.html")

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")

_THUMB_HEIGHT = 320  # ponytail: report displays frames at 160px tall; 2x for retina, downscale rest to keep report.html small


def _sample_frames(video_path: str, count: int = 2) -> list[bytes]:
    """Grab `count` frames spread through the clip, as JPEG bytes, for the report thumbnails."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(count):
        idx = int(total * (i + 1) / (count + 1)) if total > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            scale = _THUMB_HEIGHT / h
            frame = cv2.resize(frame, (round(w * scale), _THUMB_HEIGHT))
            frames.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    cap.release()
    return frames


def _cache_paths(clip: str) -> tuple[str, list[str]]:
    return clip + ".cache.json", [f"{clip}.{i}.jpg" for i in range(2)]


def _load_cached_entry(
    clip: str, start: float, end: float, actual_kbps: float, source_video: str
) -> ReportEntry | None:
    cache_json, frame_paths = _cache_paths(clip)
    if not os.path.isfile(cache_json) or not all(os.path.isfile(p) for p in frame_paths):
        return None
    with open(cache_json) as f:
        data = json.load(f)
    frames = [open(p, "rb").read() for p in frame_paths]
    return ReportEntry(
        video_name=os.path.basename(clip),
        categorization=Categorization(**data["categorization"]),
        frames=frames,
        latency_seconds=data["latency_seconds"],
        start_seconds=start,
        duration_seconds=end - start,
        actual_bitrate_kbps=actual_kbps,
        source_video=source_video,
    )


def _compute_entry(clip: str, start: float, end: float, actual_kbps: float, source_video: str) -> ReportEntry:
    t0 = time.monotonic()
    result = categorize(clip)
    latency = time.monotonic() - t0
    frames = _sample_frames(clip)

    cache_json, frame_paths = _cache_paths(clip)
    with open(cache_json, "w") as f:
        json.dump({"categorization": asdict(result), "latency_seconds": latency}, f)
    for path, frame in zip(frame_paths, frames):
        with open(path, "wb") as f:
            f.write(frame)

    return ReportEntry(
        video_name=os.path.basename(clip),
        categorization=result,
        frames=frames,
        latency_seconds=latency,
        start_seconds=start,
        duration_seconds=end - start,
        actual_bitrate_kbps=actual_kbps,
        source_video=source_video,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean", action="store_true", help="wipe scenes/ (splits + classification cache) before running"
    )
    args = parser.parse_args()
    if args.clean:
        shutil.rmtree(SCENES_DIR, ignore_errors=True)

    videos = [f for f in os.listdir(VIDS_DIR) if f.lower().endswith(VIDEO_EXTS)]
    if not videos:
        raise SystemExit(f"no videos found in {VIDS_DIR}")

    entries = []
    for video in videos:
        video_path = os.path.join(VIDS_DIR, video)
        print(f"splitting {video} into scenes...")
        clips = split_into_scenes(video_path, SCENES_DIR)
        print(f"  {len(clips)} scenes")
        packets = packet_sizes(video_path)

        for clip, start, end in clips:
            actual_kbps = actual_bitrate_kbps(packets, start, end)
            entry = _load_cached_entry(clip, start, end, actual_kbps, video)
            cached = entry is not None
            if entry is None:
                entry = _compute_entry(clip, start, end, actual_kbps, video)
            entries.append(entry)
            c = entry.categorization
            print(
                f"  {'[cached] ' if cached else ''}{os.path.basename(clip)}: tier={c.tier} "
                f"predicted={c.preferred_bitrate_kbps}kbps actual={actual_kbps:.0f}kbps "
                f"latency={entry.latency_seconds:.2f}s"
            )

    build_html_report(entries, REPORT_PATH)
    print(f"wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
