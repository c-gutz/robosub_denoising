from .base import Denoiser

# Backend modules import their model libraries lazily, so listing them here
# doesn't require every library to be installed.
DENOISERS = {
    "deepfilternet3": "denoising.deepfilternet:DeepFilterNetDenoiser",
}


def load_denoiser(name: str, **kwargs) -> Denoiser:
    """Instantiate a denoiser by name, e.g. ``load_denoiser("deepfilternet3")``."""
    import importlib

    try:
        target = DENOISERS[name]
    except KeyError:
        raise ValueError(f"Unknown denoiser {name!r}; choose from {sorted(DENOISERS)}") from None
    module_name, cls_name = target.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)
