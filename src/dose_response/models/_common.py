"""Pieces shared by more than one model."""
import logging

import pymc as pm

from ..covariates import DEFAULT_COVARIATE_PRIOR_SD, parse_covariate_priors
from ..priors import get_prior_dist

__all__ = ["_covariate_offsets"]


def _covariate_offsets(args, cov_spec, X_treated, X_untreated):
    """Build the `X . beta` offset terms, or (None, None) if no covariates are in use.

    Must be called inside a `with pm.Model()` block. Returns per-observation offsets for the
    treated and untreated likelihoods built from the SAME `beta` -- that shared coefficient is
    what makes beta identifiable: the dose-0 contrast never touches the dose-response curve,
    so it pins beta, which then carries onto the treated samples.

    Returning (None, None) rather than a zero term is deliberate: callers guard on it so that
    with no covariates NO new random variable is created and the model graph is unchanged,
    which keeps existing fits exactly reproducible.
    """
    if cov_spec is None or X_treated is None or X_treated.shape[1] == 0:
        return None, None

    cov_priors = parse_covariate_priors(getattr(args, "covariate_prior", None))
    unknown = [c for c in cov_priors if c not in cov_spec.columns]
    if unknown:
        logging.getLogger(__name__).warning(
            "--covariate_prior given for column(s) %s which are not in the design "
            "(dropped for this series, or misspelled); those priors are ignored.", unknown
        )

    # Per-covariate scalars stacked into a vector, mirroring how slope/logEC50 are built
    # per treatment. Keeping them as separate scalars allows a different prior family per
    # covariate, which a single vector RV could not express.
    beta_list = []
    for c in cov_spec.columns:
        if c in cov_priors:
            family, params_ = cov_priors[c]
            beta_list.append(get_prior_dist(family, params_, f"beta_{c}"))
        else:
            beta_list.append(pm.Normal(f"beta_{c}", mu=0.0, sigma=DEFAULT_COVARIATE_PRIOR_SD))
    beta = pm.Deterministic("beta", pm.math.stack(beta_list), dims="covariate")

    Xt = pm.Data("X_treated", X_treated, dims=("obs_treated", "covariate"))
    Xu = pm.Data("X_untreated", X_untreated, dims=("obs_untreated", "covariate"))
    return pm.math.dot(Xt, beta), pm.math.dot(Xu, beta)
