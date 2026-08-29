"""Model 2: splicing PSI, beta-binomial, logistic applied in PSI space."""
import numpy as np
import pymc as pm

from ..covariates import design_for_feature
from ..priors import get_prior_dist, parse_priors

__all__ = ["fit_splicing_psi", "SPAN_FLOOR_PSI"]

LN2, LN10, LOG2_10 = np.log(2.0), np.log(10.0), np.log2(10.0)

# PSI-scale analogue of the log2-odds span floor: below this the span carries no slope
# information, so floor it when deriving rate from hill.
SPAN_FLOOR_PSI = 0.01

DEFAULTS = {
    "baseline_log2odds": ("Normal", None),
    "span_log2odds": ("StudentT", [3.0, 0.0, 6.0]),
    "hill": ("LogNormal", [np.log(1.5), 0.35]),
    "phi": ("Gamma", [2.0, 0.2]),
    "beta_log2odds": ("Normal", [0.0, 4.0]),
}
LOGEC50_SD = 3.0


def fit_splicing_psi(data, samples=1000, args=None):
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

        # The covariate shifts the whole curve here, both asymptotes by the same amount.
        psi_base_ref = pm.math.sigmoid(baseline * LN2)
        psi_plat_ref = pm.math.sigmoid(plateau * LN2)
        span_psi_ref = psi_plat_ref - psi_base_ref
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

        base_treated = pm.math.sigmoid((baseline + offset_treated) * LN2)
        plat_treated = pm.math.sigmoid((plateau + offset_treated) * LN2)
        psi_treated = base_treated + (plat_treated - base_treated) / (
            1 + pm.math.exp(-rate * (log10_dose - logEC50[treatment_idx])))
        psi_untreated = pm.math.sigmoid((baseline + offset_untreated) * LN2)
        pm.Deterministic("psi_treated", psi_treated, dims="obs_treated")

        pm.BetaBinomial("y_treated_mu", alpha=psi_treated * phi, beta=(1 - psi_treated) * phi,
                        n=n_treated_data, observed=y_treated_data, dims="obs_treated")
        pm.BetaBinomial("y_untreated_mu", alpha=psi_untreated * phi,
                        beta=(1 - psi_untreated) * phi, n=n_untreated_data,
                        observed=y_untreated_data, dims="obs_untreated")

        base_arm = pm.Deterministic("baseline_PSI",
                                    pm.math.sigmoid((baseline + offset_by_arm) * LN2),
                                    dims="treatment")
        plat_arm = pm.Deterministic("plateau_PSI",
                                    pm.math.sigmoid((plateau + offset_by_arm) * LN2),
                                    dims="treatment")
        span_psi = pm.Deterministic("span_PSI", plat_arm - base_arm, dims="treatment")

        sig_top = 1 / (1 + pm.math.exp(-rate * (x_top - logEC50)))
        dpsi_top = pm.Deterministic("dPSI_at_maxdose", span_psi * sig_top, dims="treatment")
        pm.Deterministic("frac_realized", dpsi_top / span_psi, dims="treatment")

        # Location readouts. logEC50 is already the PSI-halfway dose; the log2-odds-halfway
        # point is a different dose.
        eta_mid = (pm.math.log(((base_arm + plat_arm) / 2)
                               / (1 - (base_arm + plat_arm) / 2)) / LN2)
        floor_arm2 = baseline + offset_by_arm
        frac = pm.math.clip((eta_mid - floor_arm2) / (plateau - baseline), 1e-9, 1 - 1e-9)
        pm.Deterministic("logEC50_log2odds",
                         logEC50 - pm.math.log(frac / (1 - frac)) / rate, dims="treatment")
        pm.Deterministic(
            "logEC_dPSI05",
            logEC50 - (1 / rate) * pm.math.log(pm.math.abs(span_psi) / 0.05 - 1),
            dims="treatment")
        f2 = pm.math.switch(span_psi > 0, 2.0, 0.5)
        odds_target = f2 * base_arm / (1 - base_arm)
        psi_target = odds_target / (1 + odds_target)
        r2 = pm.math.clip((psi_target - base_arm) / span_psi, 1e-9, 1 - 1e-9)
        pm.Deterministic("logEC2x_odds",
                         logEC50 + pm.math.log(r2 / (1 - r2)) / rate, dims="treatment")
        pm.Deterministic("hill_check", rate * span_psi_ref / (4 * m * (1 - m) * LN10))

        idata = pm.sample(samples, tune=1000, chains=4, cores=1, random_seed=42,
                          target_accept=0.9, return_inferencedata=True)
    return idata, model
