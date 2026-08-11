"""Content-complexity categories for bitrate estimation.

Zero-shot ranks a video against captions spanning motion levels (the axis
that drives how much bitrate a given resolution needs - this is the same
reasoning per-title/complexity-aware encoders use, just approximated with
natural-language captions instead of measured motion metrics). The winning
caption's motion level maps to a tier, which maps to a bitrate range via a
per-resolution base rate.

# ponytail: originally also ranked a spatial-detail axis (flat/moderate/fine
# texture), joined with motion into a 3x3 grid. Tried 3 different caption
# wordings (see git history) against 101 real scenes - the detail axis picked
# "moderate" 100% of the time regardless of actual visual content in every
# version. This model's video-text embeddings match concrete semantic content
# ("a person walking"), not abstract visual-statistics descriptors ("fine
# texture") - no caption rewording fixed it, so detail axis was dropped rather
# than kept as dead weight. Motion alone does discriminate (verified: real
# static/moderate/high spread across the same 101 scenes).
"""
MOTION_LEVELS = ["static", "moderate", "high"]

# multiple caption phrasings per motion tier - a single caption per tier let
# "moderate" dominate the softmax (verified: 0/101 real scenes hit "high").
# The original joint grid effectively had 3 "high" phrasings (paired with each
# detail level) and did hit "high" 14/101 times - restoring that redundancy
# (still pure cosine similarity, just more exemplars per tier) recovers a
# balanced spread (verified: 11 static / 63 moderate / 27 high on 101 scenes).
_TIER_CAPTIONS = {
    0: [
        "A still or barely moving shot, such as a static camera or a person "
        "standing or sitting still.",
        "A calm, mostly motionless scene with little to no camera movement.",
    ],
    1: [
        "A scene with some movement, such as people walking or talking, or a "
        "slowly panning camera.",
        "A video with everyday movement, like people gesturing or a camera "
        "slowly following a subject.",
    ],
    2: [
        "A fast-moving scene with rapid motion or quick camera cuts, such as "
        "running, driving, or fighting.",
        "A video with sports or action content and fast typical camera movement.",
        "A high-energy scene with rapid on-screen motion, such as a chase, a "
        "crash, or a dance sequence.",
    ],
}

# caption -> motion_idx (0..2)
CATEGORIES = {caption: idx for idx, captions in _TIER_CAPTIONS.items() for caption in captions}

# stable id = insertion order in CATEGORIES, for report/log references
CATEGORY_ID = {caption: i for i, caption in enumerate(CATEGORIES)}

# motion_idx -> tier
SCORE_TO_TIER = {0: "low", 1: "medium", 2: "high"}

# recommended kbps at "medium" complexity, by resolution - the multiplier
# below scales this up/down for other tiers
RESOLUTION_BASE_KBPS = {
    "360p": 800,
    "480p": 1200,
    "720p": 2500,
    "1080p": 4500,
    "1440p": 9000,
    "2160p": 16000,
}

TIER_MULTIPLIER = {"low": 0.6, "medium": 1.0, "high": 1.5}


def bitrate_range_kbps(tier: str, resolution: str) -> tuple[int, int]:
    base = RESOLUTION_BASE_KBPS[resolution] * TIER_MULTIPLIER[tier]
    return (round(base * 0.8), round(base * 1.2))


def estimate_bitrate(video_path: str, resolution: str = "1080p", topk: int = 3) -> dict:
    """Rank a video against the motion captions and estimate a bitrate range.

    Returns dict with: category, motion, tier, kbps_range, ranked (top-`topk`
    (category, prob) pairs from the underlying zero-shot ranking).
    """
    from videotag.pipeline import classify_video

    if resolution not in RESOLUTION_BASE_KBPS:
        raise ValueError(f"unknown resolution {resolution!r}, expected one of {list(RESOLUTION_BASE_KBPS)}")

    ranked = classify_video(video_path, list(CATEGORIES), topk=topk)
    top_label, _ = ranked[0]
    motion_idx = CATEGORIES[top_label]
    tier = SCORE_TO_TIER[motion_idx]
    return {
        "category_id": CATEGORY_ID[top_label],
        "category": top_label,
        "motion": MOTION_LEVELS[motion_idx],
        "tier": tier,
        "kbps_range": bitrate_range_kbps(tier, resolution),
        "ranked": ranked,
    }
