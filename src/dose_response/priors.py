"""Prior construction and validation.

Moved verbatim from scripts/BayesianDoseResponse_ByBatch.py; behaviour is unchanged.
"""
import pymc as pm

__all__ = ["get_prior_dist", "parse_priors", "validate_treatment_specific_priors"]


# --- Flexible prior parsing and mapping ---
def get_prior_dist(family, params, name, dims=None):
    dist_map = {
        "Normal": pm.Normal,
        "Gamma": pm.Gamma,
        "Uniform": pm.Uniform,
        "HalfNormal": pm.HalfNormal,
        "HalfCauchy": pm.HalfCauchy,
        # Add more as needed
    }
    if family not in dist_map:
        raise ValueError(f"Unknown prior family: {family}")
    dist = dist_map[family]
    kwargs = {"dims": dims} if dims else {}
    return dist(name, *params, **kwargs)

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

def validate_treatment_specific_priors(args, model_num):
    allowed = TREATMENT_INDEXED_PARAMS[model_num]
    if hasattr(args, "prior") and args.prior:
        for prior in args.prior:
            param, treatment, *_ = prior
            if treatment != "ALL" and param not in allowed:
                raise ValueError(
                    f"Parameter '{param}' is not indexed by treatment in model {model_num}, "
                    f"so you cannot specify a treatment-specific prior for it (got --prior {param} {treatment} ...)."
                )
