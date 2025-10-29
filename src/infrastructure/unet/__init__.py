from .diffusers_video_unet import DiffusersVideoUNetBackbone

def create_diffusers_video_unet(*args, **kwargs):  # lazy import to avoid heavy deps at import time
    from .factory import create_diffusers_video_unet as _impl
    return _impl(*args, **kwargs)


def create_diffusers_video_unet_from_pretrained(*args, **kwargs):  # lazy import to avoid heavy deps at import time
    from .factory import create_diffusers_video_unet_from_pretrained as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "DiffusersVideoUNetBackbone",
    "create_diffusers_video_unet",
    "create_diffusers_video_unet_from_pretrained",
]


