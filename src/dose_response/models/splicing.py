"""Splicing dose-response models: a 2x2 over response scale and covariate target."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..covariates import design_for_feature
from ..priors import get_prior_dist, parse_priors

__all__ = ["fit_splicing", "SCALES", "COVARIATE_TARGETS", "SPAN_FLOOR_LOG2ODDS",
           "SPAN_FLOOR_PSI"]

LN2, LN10, LOG2_10 = np.log(2.0), np.log(10.0), np.log2(10.0)

SCALES = ("psi", "log2odds")
COVARIATE_TARGETS = ("vertical", "sharedceiling")

# Below these the span carries no slope information, so floor it when deriving rate from hill.
SPAN_FLOOR_LOG2ODDS = 0.25
SPAN_FLOOR_PSI = 0.01

DEFAULTS = {
    "baseline_log2odds": ("Normal", None),
    "span_log2odds": ("StudentT", [3.0, 0.0, 6.0]),
    "hill": ("LogNormal", [np.log(1.5), 0.35]),
    "phi": ("Gamma", [2.0, 0.2]),
    "beta_log2odds": ("Normal", [0.0, 4.0]),
}
LOGEC50_SD = 3.0


def fit_splicing(data, samples=1000, args=None, scale="log2odds",
                 covariate_target="sharedceiling"):
    """Fit one junction.

    scale             "psi" applies the logistic to PSI, "log2odds" to log2 of the
                      inclusion/exclusion odds.
    covariate_target  "vertical" shifts floor and ceiling together; "sharedceiling" shifts
                      the floor only, leaving one ceiling for every sample and arm.

    logEC50 is the sampled per-arm parameter and centres the sigmoid on whichever scale this
    model acts on, so it is the native location. logEC50_PSI is always the dose at which PSI
    sits halfway between floor and ceiling, and is the cross-model currency. In the PSI-space
    models the two coincide.

    Named asymptotes are reference-level (covariate = 0); per-arm readouts use each arm's own
    offset.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    if covariate_target not in COVARIATE_TARGETS:
        raise ValueError(f"covariate_target must be one of {COVARIATE_TARGETS}, "
                         f"got {covariate_target!r}")

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
    dose_of = lambda t: np.log10(treated.loc[treated.treatment == t, "dose"].astype(float))
    ec50_mu = [float((dose_of(t).min() + dose_of(t).max()) / 2) for t in treatments]
    x_top = np.array([float(dose_of(t).max()) for t in treatments])
    X_by_arm = (np.vstack([cov_spec.matrix(treated.loc[treated.treatment == t, "sample"].tolist())
                           .mean(axis=0) for t in treatments]) if has_cov else None)

    coords = {"treatment": treatments,
              "obs_treated": np.arange(len(y_treated)),
              "obs_untreated": np.arange(len(y_untreated))}
    if has_cov:
        coords["covariate"] = list(cov_spec.columns)

    priors, default_priors = parse_priors(args)

    def scalar(name, fallback=None):
        if name in priors and "ALL" in priors[name]:
            fam, pars = priors[name]["ALL"]
        elif name in default_priors:
            fam, pars = default_priors[name]
        else:
            fam, pars = DEFAULTS[name][0], DEFAULTS[name][1] or fallback
        return get_prior_dist(fam, pars, name)

    sig = pm.math.sigmoid
    to_psi = lambda eta: sig(eta * LN2)
    to_log2odds = lambda psi: pm.math.log(psi / (1 - psi)) / LN2

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

        ceiling_shift = offset_treated if covariate_target == "vertical" else 0.0
        ceiling_shift_arm = offset_by_arm if covariate_target == "vertical" else 0.0

        # reference-level asymptotes, on both scales, always corresponding
        psi_base_ref = pm.Deterministic("baseline_PSI", to_psi(baseline))
        psi_plat_ref = pm.Deterministic("plateau_PSI", to_psi(plateau))
        span_psi_ref = pm.Deterministic("span_PSI", psi_plat_ref - psi_base_ref)

        if scale == "log2odds":
            rate = pm.Deterministic(
                "rate", 4 * LOG2_10 * hill
                / pm.math.sqrt(span ** 2 + SPAN_FLOOR_LOG2ODDS ** 2))
        else:
            m = (psi_base_ref + psi_plat_ref) / 2.0
            rate = pm.Deterministic(
                "rate", 4 * m * (1 - m) * LN10 * hill
                / pm.math.sqrt(span_psi_ref ** 2 + SPAN_FLOOR_PSI ** 2))

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

        floor_arm = baseline + offset_by_arm
        ceil_arm = plateau + ceiling_shift_arm
        psi_floor_arm, psi_ceil_arm = to_psi(floor_arm), to_psi(ceil_arm)
        span_arm_psi = psi_ceil_arm - psi_floor_arm
        span_arm = pm.Deterministic("span_by_arm_log2odds", ceil_arm - floor_arm,
                                    dims="treatment")
        pm.Deterministic("span_sign_min", pm.math.min(span_arm * pt.sign(span)))

        floor_obs = baseline + offset_treated
        ceil_obs = plateau + ceiling_shift

        if scale == "log2odds":
            eta = floor_obs + (ceil_obs - floor_obs) / (
                1 + pm.math.exp(-rate * (log10_dose - logEC50[treatment_idx])))
            psi_treated = to_psi(eta)
            psi_top = to_psi(floor_arm + span_arm / (
                1 + pm.math.exp(-rate * (x_top - logEC50))))
            # PSI-halfway dose: invert the log2-odds curve at the log2-odds of the PSI midpoint
            eta_psi_mid = to_log2odds((psi_floor_arm + psi_ceil_arm) / 2.0)
            frac = pm.math.clip((eta_psi_mid - floor_arm) / span_arm, 1e-9, 1 - 1e-9)
            pm.Deterministic("logEC50_PSI",
                             logEC50 + pm.math.log(frac / (1 - frac)) / rate,
                             dims="treatment")
        else:
            base_obs, ceil_obs_psi = to_psi(floor_obs), to_psi(ceil_obs)
            psi_treated = base_obs + (ceil_obs_psi - base_obs) / (
                1 + pm.math.exp(-rate * (log10_dose - logEC50[treatment_idx])))
            psi_top = psi_floor_arm + span_arm_psi / (
                1 + pm.math.exp(-rate * (x_top - logEC50)))
            # the sigmoid is already centred on the PSI midpoint here
            pm.Deterministic("logEC50_PSI", logEC50 * 1.0, dims="treatment")

        psi_untreated = to_psi(baseline + offset_untreated)
        pm.Deterministic("psi_treated", psi_treated, dims="obs_treated")

        pm.BetaBinomial("y_treated_mu", alpha=psi_treated * phi, beta=(1 - psi_treated) * phi,
                        n=n_treated_data, observed=y_treated_data, dims="obs_treated")
        pm.BetaBinomial("y_untreated_mu", alpha=psi_untreated * phi,
                        beta=(1 - psi_untreated) * phi, n=n_untreated_data,
                        observed=y_untreated_data, dims="obs_untreated")

        dpsi_top = pm.Deterministic("dPSI_at_maxdose", psi_top - psi_floor_arm,
                                    dims="treatment")
        pm.Deterministic("frac_realized", dpsi_top / span_arm_psi, dims="treatment")
        pm.Deterministic(
            "logEC_dPSI05",
            logEC50 - (1 / rate) * pm.math.log(pm.math.abs(span_arm_psi) / 0.05 - 1),
            dims="treatment")
        f2 = pm.math.switch(span_arm_psi > 0, 2.0, 0.5)
        odds_target = f2 * psi_floor_arm / (1 - psi_floor_arm)
        psi_target = odds_target / (1 + odds_target)
        r2 = pm.math.clip((psi_target - psi_floor_arm) / span_arm_psi, 1e-9, 1 - 1e-9)
        pm.Deterministic("logEC2x_odds", logEC50 + pm.math.log(r2 / (1 - r2)) / rate,
                         dims="treatment")

        idata = pm.sample(samples, tune=1000, target_accept=0.95, random_seed=42, cores=1)
    return idata, model
