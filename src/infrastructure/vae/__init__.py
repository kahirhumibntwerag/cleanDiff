from .diffusers_vae import DiffusersVAEBackbone

def create_diffusers_vae_from_pretrained(*args, **kwargs):  # lazy import
    from .factory import create_diffusers_vae_from_pretrained as _impl
    return _impl(*args, **kwargs)


def create_diffusers_vae(*args, **kwargs):  # lazy import
    from .factory import create_diffusers_vae as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "DiffusersVAEBackbone",
    "create_diffusers_vae_from_pretrained",
    "create_diffusers_vae",
]


