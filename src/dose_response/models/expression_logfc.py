"""Model 1: log2 fold-change expression, untreated level pinned at 0.

Moved verbatim from scripts/BayesianDoseResponse_ByBatch.py; behaviour is unchanged.
"""
import numpy as np
import pymc as pm

from ..covariates import design_for_feature
from ..priors import get_prior_dist, parse_priors
from ._common import _covariate_offsets

__all__ = ["fit_gene_expression_model"]


def fit_gene_expression_model(data, samples=1000, args=None):
    data = data.copy()   # `data` holds one feature's rows, subset by the caller

    is_treated = data["dose"].notna() & (data["dose"] != 0)
    treated_data = data[is_treated].copy()
    treated_data["treatment"] = treated_data["treatment"].astype("category")
    treated_data["treatment_ID"] = treated_data["treatment"].cat.codes
    treatments = treated_data["treatment"].cat.categories

    # Prepare arrays for treated and untreated
    log10_dose_treated = np.log10(treated_data["dose"])
    treatment_treated = treated_data["treatment_ID"].values
    y_treated = treated_data["y"].values

    is_untreated = ~is_treated
    y_untreated = data.loc[is_untreated, "y"].values

    coords = {
        "treatment": treatments,
        "obs_treated": np.arange(len(y_treated)),
        "obs_untreated": np.arange(len(y_untreated))
    }

    priors, default_priors = parse_priors(args)

    with pm.Model(coords=coords) as model:
        log10_dose = pm.Data("log10_dose", log10_dose_treated, dims="obs_treated")
        treatment_idx = pm.Data("treatment_idx", treatment_treated, dims="obs_treated")
        y_treated_data = pm.Data("y_treated", y_treated, dims="obs_treated")
        y_untreated_data = pm.Data("y_untreated", y_untreated, dims="obs_untreated")

        # Flexible priors for upper
        if "upper" in priors and "ALL" in priors["upper"]:
            family, params_ = priors["upper"]["ALL"]
            upper = get_prior_dist(family, params_, "upper")
        elif "upper" in default_priors:
            family, params_ = default_priors["upper"]
            upper = get_prior_dist(family, params_, "upper")
        else:
            upper = pm.Normal('upper', mu=0, sigma=3.0)

        # logEC50 default prior centered at midpoint of each treatment's assayed log10-dose range.
        # sigma=1.5 keeps the same width as the old prior — only the center moves.
        logEC50_mu_data = {}
        for t in treatments:
            log_doses = np.log10(
                treated_data[treated_data["treatment"] == t]["dose"].astype(float).values
            )
            logEC50_mu_data[t] = (log_doses.min() + log_doses.max()) / 2.0

        # Flexible priors for slope and logEC50 (per-treatment)
        slope_list = []
        logEC50_list = []
        for i, t in enumerate(treatments):
            # Slope
            if "slope" in priors and t in priors["slope"]:
                family, params_ = priors["slope"][t]
                slope_list.append(get_prior_dist(family, params_, f"slope_{t}"))
            elif "slope" in default_priors:
                family, params_ = default_priors["slope"]
                slope_list.append(get_prior_dist(family, params_, f"slope_{t}"))
            else:
                slope_list.append(pm.Gamma(f"slope_{t}", alpha=4, beta=1.5))
            # logEC50
            if "logEC50" in priors and t in priors["logEC50"]:
                family, params_ = priors["logEC50"][t]
                logEC50_list.append(get_prior_dist(family, params_, f"logEC50_{t}"))
            elif "logEC50" in default_priors:
                family, params_ = default_priors["logEC50"]
                logEC50_list.append(get_prior_dist(family, params_, f"logEC50_{t}"))
            else:
                logEC50_list.append(pm.Normal(f"logEC50_{t}", mu=logEC50_mu_data[t], sigma=1.5))
        slope = pm.Deterministic("slope", pm.math.stack(slope_list), dims="treatment")
        logEC50 = pm.Deterministic("logEC50", pm.math.stack(logEC50_list), dims="treatment")

        sigma = pm.HalfNormal('sigma', sigma=1)

        slope_t = slope[treatment_idx]
        logEC50_t = logEC50[treatment_idx]
        y_treated_mu = upper / (1 + pm.math.exp(-slope_t * (log10_dose - logEC50_t)))

        pm.Normal('y_treated_mu', mu=y_treated_mu, sigma=sigma, observed=y_treated_data, dims="obs_treated")
        pm.Normal('y_untreated_mu', mu=0, sigma=sigma, observed=y_untreated_data, dims="obs_untreated")

        pm.Deterministic('ED2x', logEC50 - (1 / slope) * pm.math.log(pm.math.abs(upper) - 1), dims="treatment")

        idata = pm.sample(samples, tune=1000, target_accept=0.95, random_seed=42, cores=1)

    return idata, model
