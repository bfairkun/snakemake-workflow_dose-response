"""Model 4: splicing PSI, beta-binomial, logistic applied in log2-odds space."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..covariates import design_for_feature
from ..priors import get_prior_dist, parse_priors

__all__ = ["fit_splicing_log2odds", "LN2", "LOG2_10", "SPAN_FLOOR"]

LN2, LOG2_10 = np.log(2.0), np.log2(10.0)

# Below this the span carries no slope information, so floor it when deriving rate from hill.
SPAN_FLOOR = 0.25

DEFAULTS = {
    "baseline_log2odds": ("Normal", None),          # mu filled from the observed controls
    "span_log2odds": ("StudentT", [3.0, 0.0, 6.0]),
    "hill": ("LogNormal", [np.log(1.5), 0.35]),
    "phi": ("Gamma", [2.0, 0.2]),
    "beta_log2odds": ("Normal", [0.0, 4.0]),
    "logEC50": ("Normal", None),                    # mu filled per arm from its dose range
}
LOGEC50_SD = 3.0


def fit_splicing_log2odds(data, samples=1000, args=None):
    data = data.copy()
    is_treated = data["dose"].notna() & (data["dose"] != 0)
    treated = data[is_treated].copy()
    treated["treatment"] = treated["treatment"].astype("category")
    treated["treatment_ID"] = treated["treatment"].cat.codes
    treatments = list(treated["treatment"].cat.categories)
    untreated = data[~is_treated].copy()

    x_treated = np.log10(treated["dose"].astype(float)).values
    y_treated = treated["y"].values.astype(int)
    n_treated = treated["n"].values.astype(int)
    y_untreated = untreated["y"].values.astype(int)
    n_untreated = untreated["n"].values.astype(int)

    cov_spec = getattr(args, "cov_spec", None) if args is not None else None
    X_treated, X_untreated = design_for_feature(cov_spec, data)
    has_cov = cov_spec is not None and cov_spec.n_covariates > 0

    p0 = ((y_untreated.sum() + 0.5) / (n_untreated.sum() + 1.0) if len(y_untreated)
          else (y_treated.sum() + 0.5) / (n_treated.sum() + 1.0))
    baseline_mu = float(np.log(p0 / (1 - p0)) / LN2)
    ec50_mu = [float((np.log10(treated.loc[treated.treatment == t, "dose"].astype(float)).min()
                      + np.log10(treated.loc[treated.treatment == t, "dose"].astype(float)).max()) / 2)
               for t in treatments]
    x_top = np.array([float(np.log10(treated.loc[treated.treatment == t, "dose"].astype(float)).max())
                      for t in treatments])
    X_by_arm = (np.vstack([cov_spec.matrix(treated.loc[treated.treatment == t, "sample"].tolist())
                           .mean(axis=0) for t in treatments]) if has_cov else None)

    coords = {"treatment": treatments,
              "obs_treated": np.arange(len(y_treated)),
              "obs_untreated": np.arange(len(y_untreated))}
    if has_cov:
        coords["covariate"] = list(cov_spec.columns)

    priors, default_priors = parse_priors(args)

    def scalar(name, fallback_params=None):
        if name in priors and "ALL" in priors[name]:
            fam, pars = priors[name]["ALL"]
        elif name in default_priors:
            fam, pars = default_priors[name]
        else:
            fam, pars = DEFAULTS[name][0], DEFAULTS[name][1] or fallback_params
        return get_prior_dist(fam, pars, name)

    with pm.Model(coords=coords) as model:
        log10_dose = pm.Data("log10_dose", x_treated, dims="obs_treated")
        treatment_idx = pm.Data("treatment_idx", treated["treatment_ID"].values,
                                dims="obs_treated")
        y_treated_data = pm.Data("y_treated", y_treated, dims="obs_treated")
        n_treated_data = pm.Data("n_treated", n_treated, dims="obs_treated")
        y_untreated_data = pm.Data("y_untreated", y_untreated, dims="obs_untreated")
        n_untreated_data = pm.Data("n_untreated", n_untreated, dims="obs_untreated")

        baseline = scalar("baseline_log2odds", [baseline_mu, 3.0])
        span = scalar("span_log2odds")
        hill = scalar("hill")
        phi = scalar("phi")

        plateau = pm.Deterministic("plateau_log2odds", baseline + span)
        rate = pm.Deterministic(
            "rate", 4 * LOG2_10 * hill / pm.math.sqrt(span ** 2 + SPAN_FLOOR ** 2))

        offset_treated = offset_untreated = 0.0
        offset_by_arm = np.zeros(len(treatments))
        if has_cov:
            beta_list = []
            for c in cov_spec.columns:
                if "beta_log2odds" in priors and c in priors["beta_log2odds"]:
                    fam, pars = priors["beta_log2odds"][c]
                else:
                    fam, pars = DEFAULTS["beta_log2odds"]
                beta_list.append(get_prior_dist(fam, pars, f"beta_log2odds_{c}"))
            beta = pm.Deterministic("beta_log2odds", pm.math.stack(beta_list), dims="covariate")
            offset_treated = pm.math.dot(
                pm.Data("X_treated", X_treated, dims=("obs_treated", "covariate")), beta)
            offset_untreated = pm.math.dot(
                pm.Data("X_untreated", X_untreated, dims=("obs_untreated", "covariate")), beta)
            offset_by_arm = pm.math.dot(
                pm.Data("X_by_arm", X_by_arm, dims=("treatment", "covariate")), beta)

        floor_by_arm = baseline + offset_by_arm
        psi_floor_arm = pm.math.sigmoid(floor_by_arm * LN2)
        psi_plateau = pm.Deterministic("plateau_PSI", pm.math.sigmoid(plateau * LN2))

        # logEC50 is the PSI-halfway dose; the log2-odds-halfway point is a different dose,
        # reached by a unit-Jacobian shift so no density correction is needed either way.
        psi_mid = (psi_floor_arm + psi_plateau) / 2.0
        eta_mid = pm.math.log(psi_mid / (1 - psi_mid)) / LN2
        frac = pm.math.clip((eta_mid - floor_by_arm) / (plateau - floor_by_arm), 1e-9, 1 - 1e-9)
        halfway_shift = pm.math.log(frac / (1 - frac)) / rate

        ec50_list = []
        for t, mu in zip(treatments, ec50_mu):
            if "logEC50" in priors and t in priors["logEC50"]:
                fam, pars = priors["logEC50"][t]
            elif "logEC50" in default_priors:
                fam, pars = default_priors["logEC50"]
            else:
                fam, pars = "Normal", [mu, LOGEC50_SD]
            ec50_list.append(get_prior_dist(fam, pars, f"logEC50_{t}"))
        logEC50 = pm.Deterministic("logEC50", pm.math.stack(ec50_list), dims="treatment")
        logEC50_log2odds = pm.Deterministic("logEC50_log2odds", logEC50 - halfway_shift,
                                           dims="treatment")

        floor_treated = baseline + offset_treated
        eta_treated = floor_treated + (plateau - floor_treated) / (
            1 + pm.math.exp(-rate * (log10_dose - logEC50_log2odds[treatment_idx])))
        psi_treated = pm.math.sigmoid(eta_treated * LN2)
        psi_untreated = pm.math.sigmoid((baseline + offset_untreated) * LN2)
        pm.Deterministic("psi_treated", psi_treated, dims="obs_treated")

        pm.BetaBinomial("y_treated_mu", alpha=psi_treated * phi, beta=(1 - psi_treated) * phi,
                        n=n_treated_data, observed=y_treated_data, dims="obs_treated")
        pm.BetaBinomial("y_untreated_mu", alpha=psi_untreated * phi,
                        beta=(1 - psi_untreated) * phi, n=n_untreated_data,
                        observed=y_untreated_data, dims="obs_untreated")

        span_by_arm = pm.Deterministic("span_by_arm_log2odds", plateau - floor_by_arm,
                                       dims="treatment")
        pm.Deterministic("span_sign_min", pm.math.min(span_by_arm * pt.sign(span)))
        pm.Deterministic("baseline_PSI", pm.math.sigmoid(baseline * LN2))
        pm.Deterministic("span_PSI", psi_plateau - pm.math.sigmoid(baseline * LN2))
        span_arm = psi_plateau - psi_floor_arm

        eta_top = floor_by_arm + (plateau - floor_by_arm) / (
            1 + pm.math.exp(-rate * (x_top - logEC50_log2odds)))
        dpsi_top = pm.Deterministic(
            "dPSI_at_maxdose", pm.math.sigmoid(eta_top * LN2) - psi_floor_arm, dims="treatment")
        pm.Deterministic("frac_realized", dpsi_top / span_arm, dims="treatment")

        idata = pm.sample(samples, tune=1000, target_accept=0.95, random_seed=42, cores=1)
    return idata, model
