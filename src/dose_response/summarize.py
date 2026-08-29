"""Per-treatment R^2 for each outcome type."""
import numpy as np
import pymc as pm

__all__ = ["r2_by_treatment_expression", "r2_by_treatment_splicing"]


def r2_by_treatment_expression(idata, model):
    # All variable names are hardcoded for the expression model
    with model:
        ppc = pm.sample_posterior_predictive(idata, var_names=["y_treated_mu"], random_seed=42)
    y_pred_ppc = ppc.posterior_predictive['y_treated_mu'].mean(axis=(0,1)).values
    y_obs = idata.observed_data['y_treated_mu'].values
    treatment_idx = idata.constant_data["treatment_idx"].values
    treatment_names = list(idata.posterior.coords["treatment"].values)
    r2_dict = {}
    for i, t in enumerate(treatment_names):
        mask = treatment_idx == i
        if np.sum(mask) > 1:
            y_obs_t = y_obs[mask]
            y_pred_t = y_pred_ppc[mask]
            ss_res = np.sum((y_obs_t - y_pred_t) ** 2)
            ss_tot = np.sum((y_obs_t - np.mean(y_obs_t)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            r2 = np.nan
        r2_dict[t] = r2
    return r2_dict

def r2_by_treatment_splicing(idata, model):
    # Posterior mean PSI for each treated sample
    psi_pred = idata.posterior["psi_treated_mu"].mean(dim=("chain", "draw")).values
    # Use y and n from constant_data for observed PSI
    y_obs = idata.constant_data['y_treated'].values
    n_obs = idata.constant_data['n_treated'].values
    psi_obs = y_obs / n_obs
    treatment_idx = idata.constant_data["treatment_idx"].values
    treatment_names = list(idata.posterior.coords["treatment"].values)
    r2_dict = {}
    for i, t in enumerate(treatment_names):
        mask = treatment_idx == i
        if np.sum(mask) > 1:
            psi_obs_t = psi_obs[mask]
            psi_pred_t = psi_pred[mask]
            ss_res = np.sum((psi_obs_t - psi_pred_t) ** 2)
            ss_tot = np.sum((psi_obs_t - np.mean(psi_obs_t)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            r2 = np.nan
        r2_dict[t] = r2
    return r2_dict
