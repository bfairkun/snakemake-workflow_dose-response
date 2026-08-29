"""Model 4: splicing PSI, beta-binomial, logistic applied in log2-odds space."""
import numpy as np
import pymc as pm

from ..covariates import DEFAULT_COVARIATE_PRIOR_SD, design_for_feature

__all__ = ["fit_model4b", "LN2", "LOG2_10", "AMP_FLOOR2"]

LN2, LOG2_10 = np.log(2.0), np.log2(10.0)

AMP_FLOOR2 = 0.25


def fit_model4b(data, samples=1000, cov_spec=None, seed=42,
                H_prior=(np.log(1.5), 0.35), Delta2_sd=8.0, a2_sd=3.0,
                beta2_sd=4.0, loc_sd=1.0, location_prior="ECdPSI50"):
    data = data.copy()
    is_t = data["dose"].notna() & (data["dose"] != 0)
    td = data[is_t].copy()
    td["treatment"] = td["treatment"].astype("category"); td["tid"] = td["treatment"].cat.codes
    trts = list(td["treatment"].cat.categories)
    ud = data[~is_t].copy()
    x_t = np.log10(td["dose"].astype(float)).values
    y_t, n_t = td["y"].values.astype(int), td["n"].values.astype(int)
    y_u, n_u = ud["y"].values.astype(int), ud["n"].values.astype(int)
    Xt, Xu = design_for_feature(cov_spec, data)
    has_cov = cov_spec is not None and cov_spec.n_covariates > 0

    # baseline anchored on the observed control odds, in log2-odds
    p0 = (y_u.sum()+0.5)/(n_u.sum()+1.0) if len(y_u) else (y_t.sum()+0.5)/(n_t.sum()+1.0)
    a2_mu = float(np.log(p0/(1-p0))/LN2)
    # Prior sits on the PSI-halfway dose, not the log2-odds-halfway point; see docs/models.qmd.
    ec_mu = [float((np.log10(td.loc[td.treatment==t,"dose"].astype(float)).min() +
                    np.log10(td.loc[td.treatment==t,"dose"].astype(float)).max())/2) for t in trts]
    X_arm = None
    if has_cov:
        X_arm = np.vstack([cov_spec.matrix(td.loc[td.treatment==t,"sample"].tolist()).mean(axis=0)
                           for t in trts])

    coords = {"treatment": trts, "obs_treated": np.arange(len(y_t)),
              "obs_untreated": np.arange(len(y_u))}
    if has_cov: coords["covariate"] = list(cov_spec.columns)

    with pm.Model(coords=coords) as m:
        xd  = pm.Data("log10_dose", x_t, dims="obs_treated")
        tix = pm.Data("treatment_idx", td["tid"].values, dims="obs_treated")

        a2     = pm.Normal("a2", mu=a2_mu, sigma=a2_sd)            # floor, doublings of odds
        Delta2 = pm.Normal("Delta2", mu=0.0, sigma=Delta2_sd)      # signed span, doublings
        H      = pm.LogNormal("H", mu=H_prior[0], sigma=H_prior[1])
        phi    = pm.Gamma("phi", alpha=2, beta=0.2)
        amp_eff = pm.math.sqrt(Delta2**2 + AMP_FLOOR2**2)
        k  = pm.Deterministic("k", 4*LOG2_10*H/amp_eff)            # slope on the log10-dose axis
        U2 = pm.Deterministic("U2", a2 + Delta2)                   # SHARED asymptote

        off_t = off_u = 0.0; off_arm = np.zeros(len(trts))
        if has_cov:
            beta2 = pm.Deterministic("beta2", pm.math.stack(
                [pm.Normal(f"beta2_{c}", mu=0.0, sigma=beta2_sd) for c in cov_spec.columns]),
                dims="covariate")
            off_t   = pm.math.dot(pm.Data("X_treated", Xt, dims=("obs_treated","covariate")), beta2)
            off_u   = pm.math.dot(pm.Data("X_untreated", Xu, dims=("obs_untreated","covariate")), beta2)
            off_arm = pm.math.dot(pm.Data("X_arm", X_arm, dims=("treatment","covariate")), beta2)

        A2_arm = a2 + off_arm

        # Unit-Jacobian shift, so no density correction whichever location is sampled.
        psi_mid  = (pm.math.sigmoid(A2_arm*LN2) + pm.math.sigmoid(U2*LN2))/2.0
        eta2_mid = pm.math.log(psi_mid/(1-psi_mid))/LN2
        f_mid = pm.math.clip((eta2_mid - A2_arm)/(U2 - A2_arm), 1e-9, 1-1e-9)
        delta_half = pm.math.log(f_mid/(1-f_mid))/k

        if location_prior == "ECdPSI50":
            ecd = pm.Deterministic("EC_dPSI50Max", pm.math.stack(
                [pm.Normal(f"EC_dPSI50Max_{t}", mu=mu, sigma=loc_sd)
                 for t, mu in zip(trts, ec_mu)]), dims="treatment")
            e = pm.Deterministic("logEC50", ecd - delta_half, dims="treatment")
        elif location_prior == "logEC50":
            e = pm.Deterministic("logEC50", pm.math.stack(
                [pm.Normal(f"logEC50_{t}", mu=mu, sigma=loc_sd)
                 for t, mu in zip(trts, ec_mu)]), dims="treatment")
            pm.Deterministic("EC_dPSI50Max", e + delta_half, dims="treatment")
        else:
            raise ValueError(location_prior)
        pm.Deterministic("delta_half", delta_half, dims="treatment")

        A2_t, A2_u = a2 + off_t, a2 + off_u
        eta2_t = A2_t + (U2 - A2_t)/(1 + pm.math.exp(-k*(xd - e[tix])))
        psi_t  = pm.math.sigmoid(eta2_t*LN2)
        psi_u  = pm.math.sigmoid(A2_u*LN2)                          # sigmoid(k*-inf)=0 at dose 0
        pm.Deterministic("psi_treated_mu", psi_t, dims="obs_treated")

        pm.BetaBinomial("y_treated_mu", alpha=psi_t*phi, beta=(1-psi_t)*phi,
                        n=pm.Data("n_treated", n_t, dims="obs_treated"),
                        observed=y_t, dims="obs_treated")
        pm.BetaBinomial("y_untreated_mu", alpha=psi_u*phi, beta=(1-psi_u)*phi,
                        n=pm.Data("n_untreated", n_u, dims="obs_untreated"),
                        observed=y_u, dims="obs_untreated")

        # ---- derived, per arm ------------------------------------------------------------
        amp2   = pm.Deterministic("amp2", U2 - A2_arm, dims="treatment")   # doublings, signed
        pm.Deterministic("min_amp_signed", pm.math.min(amp2*pm.math.sgn(Delta2)))
        pm.Deterministic("Emax", 2.0**Delta2)
        pm.Deterministic("psi_floor", pm.math.sigmoid(A2_arm*LN2), dims="treatment")
        psi_asym = pm.Deterministic("psi_asymptote", pm.math.sigmoid(U2*LN2))   # shared
        pm.Deterministic("MaxDeltaPSI", psi_asym - pm.math.sigmoid(A2_arm*LN2), dims="treatment")

        idata = pm.sample(samples, tune=1000, chains=4, cores=1, random_seed=seed,
                          target_accept=0.9, return_inferencedata=True)
    return idata, m
