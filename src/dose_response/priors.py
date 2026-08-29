"""Prior distribution construction from CLI specifications."""
from collections import defaultdict

import pymc as pm

__all__ = ["FAMILIES", "family_help", "get_prior_dist", "parse_priors"]


FAMILIES = {
    "Normal": (pm.Normal, ("mu", "sigma")),
    "StudentT": (pm.StudentT, ("nu", "mu", "sigma")),
    "LogNormal": (pm.LogNormal, ("mu", "sigma")),
    "Gamma": (pm.Gamma, ("alpha", "beta")),
    "Uniform": (pm.Uniform, ("lower", "upper")),
    "HalfNormal": (pm.HalfNormal, ("sigma",)),
    "HalfCauchy": (pm.HalfCauchy, ("beta",)),
    "Beta": (pm.Beta, ("alpha", "beta")),
    "Exponential": (pm.Exponential, ("lam",)),
}


def family_help():
    return "; ".join(f"{k} {' '.join(v[1])}" for k, v in FAMILIES.items())


def get_prior_dist(family, params, name, dims=None):
    if family not in FAMILIES:
        raise ValueError(f"Unknown prior family {family!r}. Available: {family_help()}")
    dist, names = FAMILIES[family]
    if len(params) > len(names):
        raise ValueError(f"{family} takes at most {len(names)} parameters "
                         f"({' '.join(names)}), got {len(params)}: {list(params)}")
    kwargs = dict(zip(names, params))
    if dims:
        kwargs["dims"] = dims
    return dist(name, **kwargs)


def parse_priors(args):
    priors = defaultdict(dict)
    default_priors = {}
    if hasattr(args, "prior") and args.prior:
        for prior in args.prior:
            param, treatment, family, *params = prior
            priors[param][treatment] = (family, [float(p) for p in params])
    if hasattr(args, "prior_default") and args.prior_default:
        for prior in args.prior_default:
            param, family, *params = prior
            default_priors[param] = (family, [float(p) for p in params])
    return priors, default_priors
