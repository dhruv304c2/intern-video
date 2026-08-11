"""Split a video into per-scene clips, so each scene can be classified independently."""
import json
import os

from scenedetect import ContentDetector, detect, split_video_ffmpeg


def split_into_scenes(video_path: str, out_dir: str) -> list[tuple[str, float, float]]:
    """Detect scene cuts in `video_path` and write one clip per scene into `out_dir`.

    Returns (clip_path, start_seconds, end_seconds) per scene, in order. Cached via
    a manifest file next to the clips - re-running with the same video/out_dir skips
    both scene detection and the ffmpeg re-encode.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    manifest_path = os.path.join(out_dir, f"{base}.manifest.json")

    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest and all(os.path.isfile(c["clip"]) for c in manifest):
            return [(c["clip"], c["start"], c["end"]) for c in manifest]

    scene_list = detect(video_path, ContentDetector())
    split_video_ffmpeg(
        video_path,
        scene_list,
        output_dir=out_dir,
        output_file_template=f"{base}-Scene-$SCENE_NUMBER.mp4",
    )
    clips = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith(base) and f.endswith(".mp4")
    )
    result = [(clip, start.get_seconds(), end.get_seconds()) for clip, (start, end) in zip(clips, scene_list)]
    with open(manifest_path, "w") as f:
        json.dump([{"clip": c, "start": s, "end": e} for c, s, e in result], f)
    return result


if __name__ == "__main__":
    import sys

    clips = split_into_scenes(sys.argv[1], sys.argv[2])
    print(f"wrote {len(clips)} scene clips to {sys.argv[2]}")
