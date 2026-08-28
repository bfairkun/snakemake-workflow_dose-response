"""Bayesian dose-response models for expression and splicing.

Submodules are resolved lazily so that importing the package costs nothing and so that
`dose_response.plotting` (which needs matplotlib) is never pulled into a compute env that
only fits models. Access them as attributes:

    import dose_response as dr
    dr.priors.get_prior_dist(...)
    from dose_response import plotting as drplot   # matplotlib only loaded here
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
