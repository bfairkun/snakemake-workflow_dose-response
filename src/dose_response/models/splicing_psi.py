"""Model 2: splicing PSI, beta-binomial, logistic applied in PSI space."""
import numpy as np
import pymc as pm

from ..covariates import design_for_feature
from ..priors import get_prior_dist, parse_priors
from ._common import _covariate_offsets

__all__ = ["fit_splicing_model"]


def fit_splicing_model(data, samples=1000, args=None):
    data = data.copy()   # `data` holds one feature's rows, subset by the caller

    is_treated = data["dose"].notna() & (data["dose"] != 0)
    treated_data = data[is_treated].copy()
    treated_data["treatment"] = treated_data["treatment"].astype("category")
    treated_data["treatment_ID"] = treated_data["treatment"].cat.codes
    treatments = treated_data["treatment"].cat.categories

    log10_dose_treated = np.log10(treated_data["dose"].astype(float))
    treatment_treated = treated_data["treatment_ID"].values
    y_treated = treated_data["y"].values.astype(int)
    n_treated = treated_data["n"].values.astype(int)

    is_untreated = ~is_treated
    untreated_data = data[is_untreated].copy()
    y_untreated = untreated_data["y"].values.astype(int)
    n_untreated = untreated_data["n"].values.astype(int)

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

    with pm.Model(coords=coords) as model:
        log10_dose = pm.Data("log10_dose", log10_dose_treated, dims="obs_treated")
        treatment_idx = pm.Data("treatment_idx", treatment_treated, dims="obs_treated")
        y_treated_data = pm.Data("y_treated", y_treated, dims="obs_treated")
        n_treated_data = pm.Data("n_treated", n_treated, dims="obs_treated")
        y_untreated_data = pm.Data("y_untreated", y_untreated, dims="obs_untreated")
        n_untreated_data = pm.Data("n_untreated", n_untreated, dims="obs_untreated")

        # Flexible priors for lower, upper, slope, phi, logEC50
        # lower
        if "lower" in priors and "ALL" in priors["lower"]:
            family, params_ = priors["lower"]["ALL"]
            lower = get_prior_dist(family, params_, "lower")
        elif "lower" in default_priors:
            family, params_ = default_priors["lower"]
            lower = get_prior_dist(family, params_, "lower")
        else:
            lower = pm.Uniform("lower", lower=0, upper=1)
        # upper
        delta_logit = None   # set only on the default path; used by the covariate branch below
        if "upper" in priors and "ALL" in priors["upper"]:
            family, params_ = priors["upper"]["ALL"]
            upper = get_prior_dist(family, params_, "upper")
        elif "upper" in default_priors:
            family, params_ = default_priors["upper"]
            upper = get_prior_dist(family, params_, "upper")
        else:
            # Logit-delta reparameterization: upper = sigmoid(logit(lower) + delta_logit)
            # delta_logit ~ N(0, 2.5) shrinks upper toward lower in absence of data;
            # sigma=2.5 on the logit scale allows large splicing effects (e.g. 1%→99% PSI).
            delta_logit = pm.Normal("delta_logit", mu=0, sigma=2.5)
            upper = pm.Deterministic(
                "upper",
                pm.math.sigmoid(pm.math.log(lower / (1 - lower)) + delta_logit)
            )
        # slope
        if "slope" in priors and "ALL" in priors["slope"]:
            family, params_ = priors["slope"]["ALL"]
            slope = get_prior_dist(family, params_, "slope")
        elif "slope" in default_priors:
            family, params_ = default_priors["slope"]
            slope = get_prior_dist(family, params_, "slope")
        else:
            slope = pm.Gamma("slope", alpha=4, beta=1.5)
        # phi
        if "phi" in priors and "ALL" in priors["phi"]:
            family, params_ = priors["phi"]["ALL"]
            phi = get_prior_dist(family, params_, "phi")
        elif "phi" in default_priors:
            family, params_ = default_priors["phi"]
            phi = get_prior_dist(family, params_, "phi")
        else:
            phi = pm.Gamma("phi", alpha=2, beta=0.2)
        # logEC50 (per-treatment) — default prior centered at midpoint of each treatment's
        # assayed log10-dose range, so the prior is anchored within the observable range
        # regardless of how different potencies are across treatments.
        logEC50_mu_data = {}
        for t in treatments:
            log_doses = np.log10(
                treated_data[treated_data["treatment"] == t]["dose"].astype(float).values
            )
            logEC50_mu_data[t] = (log_doses.min() + log_doses.max()) / 2.0

        logEC50_list = []
        for i, t in enumerate(treatments):
            if "logEC50" in priors and t in priors["logEC50"]:
                family, params_ = priors["logEC50"][t]
                logEC50_list.append(get_prior_dist(family, params_, f"logEC50_{t}"))
            elif "logEC50" in default_priors:
                family, params_ = default_priors["logEC50"]
                logEC50_list.append(get_prior_dist(family, params_, f"logEC50_{t}"))
            else:
                logEC50_list.append(pm.Normal(f"logEC50_{t}", mu=logEC50_mu_data[t], sigma=1.0))
        logEC50 = pm.Deterministic("logEC50", pm.math.stack(logEC50_list), dims="treatment")

        Xb_treated, Xb_untreated = _covariate_offsets(args, cov_spec, X_treated, X_untreated)

        if Xb_treated is None:
            psi_treated_mu = lower + (upper - lower) / (
                1 + pm.math.exp(-slope * (log10_dose - logEC50[treatment_idx]))
            )
            psi_untreated_mu = lower
        else:
            eta_lower = pm.math.log(lower / (1 - lower))
            delta_logit_eff = (
                delta_logit if delta_logit is not None
                else pm.math.log(upper / (1 - upper)) - eta_lower
            )
            lower_obs = pm.math.sigmoid(eta_lower + Xb_treated)
            upper_obs = pm.math.sigmoid(eta_lower + Xb_treated + delta_logit_eff)
            psi_treated_mu = lower_obs + (upper_obs - lower_obs) / (
                1 + pm.math.exp(-slope * (log10_dose - logEC50[treatment_idx]))
            )
            psi_untreated_mu = pm.math.sigmoid(eta_lower + Xb_untreated)

        pm.Deterministic("psi_treated_mu", psi_treated_mu, dims="obs_treated")

        # Beta-Binomial parameters
        alpha_treated = psi_treated_mu * phi
        beta_treated = (1 - psi_treated_mu) * phi
        alpha_untreated = psi_untreated_mu * phi
        beta_untreated = (1 - psi_untreated_mu) * phi

        # Likelihoods
        pm.BetaBinomial(
            "y_treated_mu",
            alpha=alpha_treated,
            beta=beta_treated,
            n=n_treated_data,
            observed=y_treated_data,
            dims="obs_treated"
        )
        pm.BetaBinomial(
            "y_untreated_mu",
            alpha=alpha_untreated,
            beta=beta_untreated,
            n=n_untreated_data,
            observed=y_untreated_data,
            dims="obs_untreated"
        )

        MaxDeltaPSI = pm.Deterministic("MaxDeltaPSI", lower - upper)

        # ED_5dPSI: log10 dose at which PSI increases by 0.05 above lower
        ED_5dPSI = pm.Deterministic(
            'ED_5dPSI',
            logEC50 - (1 / slope) * pm.math.log((upper - lower) / 0.05 - 1),
            dims="treatment"
        )

        # ED2x_odds: log10 dose at which odds of inclusion double/halved relative to untreated
        # Compute f: 2 if upper > lower, 0.5 if upper < lower
        f = pm.math.switch(upper > lower, 2.0, 0.5)
        # Compute y_star (the target PSI for 2x odds)
        y_star = (f * lower) / (1 - lower + f * lower)
        # Compute the log10 dose for 2x odds
        ED2x_odds = pm.Deterministic(
            "ED2x_odds",
            logEC50 - (1 / slope) * pm.math.log((upper - lower) / (y_star - lower) - 1),
            dims="treatment"
        )
        idata = pm.sample(samples, tune=1000, target_accept=0.95, random_seed=42, cores=1)
    return idata, model
