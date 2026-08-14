"""Re-encode a video to H.264 at a given bitrate, via ffmpeg."""

import argparse
import os
import subprocess


def encode_video(
    video_path: str,
    output_path: str,
    kbps: int = 2500,
    start: float | None = None,
    end: float | None = None,
) -> str:
    """Re-encode `video_path` to H.264/AAC at `kbps` and write it to `output_path`.

    `start`/`end` (seconds, absolute in the source) optionally trim to a sub-clip -
    used to cut a scene and encode it in a single ffmpeg pass.
    """
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += [
        "-c:v",
        "libx264",
        "-b:v",
        f"{kbps}k",
        "-maxrate",
        f"{kbps}k",
        "-bufsize",
        f"{kbps * 2}k",
        "-c:a",
        "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", help="path to a source video file"
    )
    parser.add_argument(
        "output", help="path to write the encoded output"
    )
    parser.add_argument(
        "--kbps",
        type=int,
        default=2500,
        help="target video bitrate in kbps",
    )
    args = parser.parse_args()

    encode_video(args.video, args.output, kbps=args.kbps)
    print(
        f"wrote {args.output} ({os.path.getsize(args.output)} bytes)"
    )


if __name__ == "__main__":
    main()
