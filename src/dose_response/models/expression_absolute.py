"""Model 3: absolute log2 abundance with a free baseline."""
import numpy as np
import pymc as pm

from ..covariates import design_for_feature
from ..priors import get_prior_dist, parse_priors
from ._common import _covariate_offsets

__all__ = ["fit_expression_absolute_model"]


def fit_expression_absolute_model(data, samples=1000, args=None):
    """Model 3: dose-response on ABSOLUTE log2 abundance, with a free intercept.

    Same log-logistic curve as model 1, but `y` is absolute log2 abundance (e.g. log2 TMM-CPM
    straight from the feature-by-sample table via `no_transform.R`) instead of a log2 fold
    change against a subtracted baseline. Consequences:

      * The untreated level `lower` is a free parameter rather than the constant 0, so the
        uncertainty in the baseline is propagated instead of being asserted to be zero. Under
        model 1 that error is shared by every treated observation, which correlates residuals
        in a way the likelihood assumes away and understates posterior width on upper/logEC50.
      * The dose-0 samples become real observations with their own residuals.
      * There is a reference level for optional covariates to shift from.

    Parameterized as (`lower`, `Delta`) with `Delta = upper - lower`:

        y_treated   = lower + Delta * sigmoid(slope_t * (x - logEC50_t)) [+ X . beta]
        y_untreated = lower                                             [+ X . beta]

    `Delta` is exactly the quantity model 1 calls `upper` (model 1 pins lower == 0), so the
    default prior Normal(0, 3) is carried over unchanged and effect sizes stay comparable
    across the two models. `upper` is also reported as a Deterministic for readability.
    """
    data = data.copy()   # `data` holds one feature's rows, subset by the caller

    is_treated = data["dose"].notna() & (data["dose"] != 0)
    treated_data = data[is_treated].copy()
    treated_data["treatment"] = treated_data["treatment"].astype("category")
    treated_data["treatment_ID"] = treated_data["treatment"].cat.codes
    treatments = treated_data["treatment"].cat.categories

    log10_dose_treated = np.log10(treated_data["dose"])
    treatment_treated = treated_data["treatment_ID"].values
    y_treated = treated_data["y"].values

    is_untreated = ~is_treated
    y_untreated = data.loc[is_untreated, "y"].values

    cov_spec = getattr(args, "cov_spec", None) if args is not None else None
    X_treated, X_untreated = design_for_feature(cov_spec, data)

    coords = {
        "treatment": treatments,
        "obs_treated": np.arange(len(y_treated)),
        "obs_untreated": np.arange(len(y_untreated))
    }
    if cov_spec is not None and cov_spec.n_covariates > 0:
        coords["covariate"] = list(cov_spec.columns)

    priors, default_priors = parse_priors(args)

    # Prior location for `lower` comes from this feature's own control mean, following the
    lower_mu_data = float(np.mean(y_untreated)) if len(y_untreated) else 0.0

    with pm.Model(coords=coords) as model:
        log10_dose = pm.Data("log10_dose", log10_dose_treated, dims="obs_treated")
        treatment_idx = pm.Data("treatment_idx", treatment_treated, dims="obs_treated")
        y_treated_data = pm.Data("y_treated", y_treated, dims="obs_treated")
        y_untreated_data = pm.Data("y_untreated", y_untreated, dims="obs_untreated")

        # Delta = upper - lower. Same meaning (and same default prior) as model 1's `upper`.
        if "Delta" in priors and "ALL" in priors["Delta"]:
            family, params_ = priors["Delta"]["ALL"]
            Delta = get_prior_dist(family, params_, "Delta")
        elif "Delta" in default_priors:
            family, params_ = default_priors["Delta"]
            Delta = get_prior_dist(family, params_, "Delta")
        else:
            Delta = pm.Normal("Delta", mu=0, sigma=3.0)

        # Free intercept: the gene's absolute untreated abundance.
        if "lower" in priors and "ALL" in priors["lower"]:
            family, params_ = priors["lower"]["ALL"]
            lower = get_prior_dist(family, params_, "lower")
        elif "lower" in default_priors:
            family, params_ = default_priors["lower"]
            lower = get_prior_dist(family, params_, "lower")
        else:
            lower = pm.Normal("lower", mu=lower_mu_data, sigma=5.0)

        # logEC50 default prior centered at midpoint of each treatment's assayed log10-dose range.
        logEC50_mu_data = {}
        for t in treatments:
            log_doses = np.log10(
                treated_data[treated_data["treatment"] == t]["dose"].astype(float).values
            )
            logEC50_mu_data[t] = (log_doses.min() + log_doses.max()) / 2.0

        # slope stays per-treatment, as in model 1: gene-level expression aggregates multiple
        slope_list = []
        logEC50_list = []
        for i, t in enumerate(treatments):
            if "slope" in priors and t in priors["slope"]:
                family, params_ = priors["slope"][t]
                slope_list.append(get_prior_dist(family, params_, f"slope_{t}"))
            elif "slope" in default_priors:
                family, params_ = default_priors["slope"]
                slope_list.append(get_prior_dist(family, params_, f"slope_{t}"))
            else:
                slope_list.append(pm.Gamma(f"slope_{t}", alpha=4, beta=1.5))
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

        Xb_treated, Xb_untreated = _covariate_offsets(args, cov_spec, X_treated, X_untreated)
        offset_treated = lower if Xb_treated is None else lower + Xb_treated
        offset_untreated = lower if Xb_untreated is None else lower + Xb_untreated

        slope_t = slope[treatment_idx]
        logEC50_t = logEC50[treatment_idx]
        y_treated_mu = offset_treated + Delta / (1 + pm.math.exp(-slope_t * (log10_dose - logEC50_t)))

        pm.Normal('y_treated_mu', mu=y_treated_mu, sigma=sigma, observed=y_treated_data, dims="obs_treated")
        pm.Normal('y_untreated_mu', mu=offset_untreated, sigma=sigma, observed=y_untreated_data, dims="obs_untreated")

        # Absolute upper asymptote, for readability alongside the relative effect size.
        pm.Deterministic("upper", lower + Delta)

        # Same definition as model 1: the dose at which the change from baseline is 2-fold.
        pm.Deterministic('ED2x', logEC50 - (1 / slope) * pm.math.log(pm.math.abs(Delta) - 1), dims="treatment")

        idata = pm.sample(samples, tune=1000, target_accept=0.95, random_seed=42, cores=1)

    return idata, model
