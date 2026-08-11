__all__ = []

# ponytail: these CLIP/audiovisual variants pull in CUDA-only deps (flash_attn,
# LLaMA/mobileclip chains) and, for internvideo2_clip, a relative-import bug
# outside this package. None are needed for Stage2 zero-shot (BERT text encoder,
# imported directly from models.backbones.*) — skip whichever fails to import.
for _name, _attr in [
    ("internvideo2_clip", "InternVideo2_CLIP"),
    ("internvideo2_clip_small", "InternVideo2_CLIP_small"),
    ("internvideo2_stage2_visual", "InternVideo2_Stage2_visual"),
    ("internvideo2_stage2_audiovisual", "InternVideo2_Stage2_audiovisual"),
]:
    try:
        _mod = __import__(f"{__name__}.{_name}", fromlist=[_attr])
        globals()[_attr] = getattr(_mod, _attr)
        __all__.append(_attr)
    except Exception:
        pass
del _name, _attr
