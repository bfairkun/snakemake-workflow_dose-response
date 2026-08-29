"""Bayesian dose-response models for expression and splicing.

Submodules resolve lazily, which keeps `dose_response.plotting` and its matplotlib
dependency out of compute environments that only fit models.
"""
import importlib

__version__ = "0.1.0"

_SUBMODULES = ("covariates", "filters", "priors", "fitting", "io", "summarize",
               "plotting", "models", "cli")


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_SUBMODULES))


__all__ = [*_SUBMODULES, "__version__"]
