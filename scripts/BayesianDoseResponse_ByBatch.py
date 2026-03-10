import argparse
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pickle
import sys
import logging
from scipy.stats import spearmanr
from collections import defaultdict

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

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Fit Bayesian Dose Response Model for Gene Expression by Batch")
    parser.add_argument(
        '--model', type=int, required=True, choices=[1, 2],
        help="Which model to use:\n\n1: Intended to model expression. The outcome variable y represents the log2 fold change in expression. The model uses a three-parameter log-logistic dose–response function to predict y, where the slope and the EC50 (the dose at which half the maximal effect is observed) vary by treatment, while the upper asymptote (maximum effect) is shared across treatments.\n\n2: Intended to model splicing. The outcome is a count of inclusion reads y out of total reads n for each observation. We model this using a beta-binomial likelihood to account for overdispersion, where the mean inclusion proportion (PSI) is linked to dose using a four-parameter log-logistic function. In this model, the EC50 varies by treatment, while the upper and lower asymptotes and the slope are shared across treatments."
    )
    parser.add_argument('--input', required=True, help="Batch input file with data. Required columns: featureID, dose, treatment, columns for outcome variables (e.g., y for model 1; y and n for model 2). If dose is 0, the sample is considered untreated.")
    parser.add_argument('--output_pkl', required=True, help="Output pickle file")
    parser.add_argument('--output_tsv', required=True, help="Output summary tsv file")
    parser.add_argument('--featureIDsToProcess', nargs='+', default=None, help="Optional: Only process these featureIDs (space-separated list or use multiple times).")
    parser.add_argument('--samples', type=int, default=1000, help="Number of samples to draw from the posterior")
    parser.add_argument('--AbsSpearmanPreFilter', type=float, default=0.4, help="Minimum |Spearman correlation| between dose and y to attempt fitting")
    parser.add_argument(
        '--PosteriorFilter', nargs=4, action='append',
        metavar=('param', 'fraction', 'low', 'high'),
        help=(
            "Posterior filter criteria, e.g. --PosteriorFilter slope 0.95 1 2. "
            "Can be specified multiple times for the same parameter for multi-interval (two-sided) filtering, "
            "e.g. --PosteriorFilter lower 0.95 -10 -1 --PosteriorFilter lower 0.95 1 10"
        )
    )
    parser.add_argument(
        '--PreFilterByNumberReasonableObservedOutcomes', nargs=4, action='append',
        metavar=('var', 'min_count', 'low', 'high'),
        help=(
            "Pre-filter: skip modeling if fewer than min_count observed values of var are within [low, high]. "
            "E.g. --PreFilterByNumberReasonableObservedOutcomes y 3 -2 2"
        )
    )
    parser.add_argument(
        '--prior', nargs='+', action='append',
        metavar='PRIOR_SPEC',
        help=(
            "Set prior for a parameter for a specific treatment: "
            "--prior logEC50 Branaplam Normal 2.5 1.0 "
            "--prior slope Risdiplam Gamma 4.0 1.5\n"
            "To set a prior for all treatments, use --prior_default."
        )
    )
    parser.add_argument(
        '--prior_default', nargs='+', action='append',
        metavar='PRIOR_DEFAULT_SPEC',
        help=(
            "Set default prior for a parameter for all treatments (unless overridden by --prior): "
            "--prior_default logEC50 Normal 3.5 1.5"
        )
    )
    parser.add_argument('--verbose', action='store_true', help="Enable verbose/debug logging")
    return parser.parse_args(args)

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )
    # Turn down logging for noisy packages
    for noisy_pkg in ["pymc", "arviz", "pytensor", "numba"]:
        logging.getLogger(noisy_pkg).setLevel(logging.WARNING)

# --- Model functions using flexible priors ---

def fit_gene_expression_model(df, feature, samples=1000, args=None):
    data = df[df["featureID"] == feature].copy()
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
                logEC50_list.append(pm.Normal(f"logEC50_{t}", mu=3.5, sigma=1.5))
        slope = pm.Deterministic("slope", pm.math.stack(slope_list), dims="treatment")
        logEC50 = pm.Deterministic("logEC50", pm.math.stack(logEC50_list), dims="treatment")

        sigma = pm.HalfNormal('sigma', sigma=1)

        slope_t = slope[treatment_idx]
        logEC50_t = logEC50[treatment_idx]
        y_treated_mu = upper / (1 + pm.math.exp(-slope_t * (log10_dose - logEC50_t)))

        pm.Normal('y_treated_mu', mu=y_treated_mu, sigma=sigma, observed=y_treated_data, dims="obs_treated")
        pm.Normal('y_untreated_mu', mu=0, sigma=sigma, observed=y_untreated_data, dims="obs_untreated")

        pm.Deterministic('ED2x', logEC50 - (1 / slope) * pm.math.log(pm.math.abs(upper) - 1), dims="treatment")

        idata = pm.sample(samples, tune=2000, target_accept=0.95, random_seed=42, cores=1)

    return idata, model

def fit_splicing_model(df, feature, samples=1000, args=None):
    data = df[df["featureID"] == feature].copy()
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
        if "upper" in priors and "ALL" in priors["upper"]:
            family, params_ = priors["upper"]["ALL"]
            upper = get_prior_dist(family, params_, "upper")
        elif "upper" in default_priors:
            family, params_ = default_priors["upper"]
            upper = get_prior_dist(family, params_, "upper")
        else:
            upper = pm.Uniform("upper", lower=0, upper=1)
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
        # logEC50 (per-treatment)
        logEC50_list = []
        for i, t in enumerate(treatments):
            if "logEC50" in priors and t in priors["logEC50"]:
                family, params_ = priors["logEC50"][t]
                logEC50_list.append(get_prior_dist(family, params_, f"logEC50_{t}"))
            elif "logEC50" in default_priors:
                family, params_ = default_priors["logEC50"]
                logEC50_list.append(get_prior_dist(family, params_, f"logEC50_{t}"))
            else:
                logEC50_list.append(pm.Normal(f"logEC50_{t}", mu=3.5, sigma=1.5))
        logEC50 = pm.Deterministic("logEC50", pm.math.stack(logEC50_list), dims="treatment")

        # Model mean PSI for treated
        psi_treated_mu = lower + (upper - lower) / (
            1 + pm.math.exp(-slope * (log10_dose - logEC50[treatment_idx]))
        )
        pm.Deterministic("psi_treated_mu", psi_treated_mu, dims="obs_treated")

        # Model mean PSI for untreated (dose=0)
        psi_untreated_mu = lower

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
        idata = pm.sample(samples, tune=2000, target_accept=0.95, random_seed=42, cores=1)
    return idata, model

### will need to add a function for fitting splicing model, that similarly returns idata and model objects

def check_prefilter_by_number(df, feature, prefilters):
    """
    prefilters: list of (var, min_count, low, high)
    Passes if for each var, at least one interval is satisfied for the required min_count.
    """
    filter_dict = defaultdict(list)
    for var, min_count, low, high in prefilters:
        filter_dict[var].append((int(min_count), float(low), float(high)))

    for var, intervals in filter_dict.items():
        vals = df[df["featureID"] == feature][var]
        passed_any = False
        actuals = []
        for min_count, low, high in intervals:
            count_in_range = ((vals >= low) & (vals <= high)).sum()
            actuals.append(f"{count_in_range} in [{low}, {high}]")
            if count_in_range >= min_count:
                passed_any = True
        if not passed_any:
            intervals_str = " or ".join([f"[{low}, {high}] (at least {min_count})" for min_count, low, high in intervals])
            actuals_str = "; ".join(actuals)
            return False, f"Did not fit; {var}: {actuals_str}; required {intervals_str}"
    return True, "Pass"


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

def check_posterior_filters(idata, filters):
    """
    filters: list of (param, fraction, low, high)
    For each param, combine all intervals, and require that the fraction of posterior samples
    in the union of all intervals is at least the specified threshold.
    Returns (True, "Pass") if all filters pass, else (False, reason)
    """
    filter_dict = defaultdict(list)
    for param, fraction, low, high in filters:
        filter_dict[param].append((float(fraction), float(low), float(high)))

    for param, intervals in filter_dict.items():
        arr = idata.posterior[param].values.flatten()
        # Union of all intervals
        mask = np.zeros_like(arr, dtype=bool)
        required_fraction = None
        for fraction, low, high in intervals:
            mask |= ((arr >= low) & (arr <= high))
            if required_fraction is None:
                required_fraction = fraction
            elif required_fraction != fraction:
                raise ValueError(f"Multiple different fractions specified for {param} in posterior filter.")
        frac_in_range = np.mean(mask)
        if frac_in_range < required_fraction:
            intervals_str = " or ".join([f"[{low}, {high}]" for _, low, high in intervals])
            return False, (
                f"Did not fit; {param}: only {frac_in_range:.2f} in {intervals_str} (required {required_fraction})"
            )
    return True, "Pass"

MODEL_CONFIG = {
    1: {  # Expression model
        "fit_func": fit_gene_expression_model,
        "spearman_func": lambda t_data: t_data["y"],
        "summary_vars_scalar": ["upper", "sigma"],
        "summary_vars_treatment": ["slope", "logEC50", "ED2x"],
        "ppc_var": "y_treated_mu",
        "obs_var": "y_treated_data",
        "treatment_idx_var": "treatment_idx",
        "r2_func": r2_by_treatment_expression,
    },
    2: {  # Splicing model
        "fit_func": fit_splicing_model,
        "spearman_func": lambda t_data: t_data["y"] / t_data["n"],
        "summary_vars_scalar": ["upper", "lower", "slope", "phi", "MaxDeltaPSI"],
        "summary_vars_treatment": ["logEC50", "ED_5dPSI", "ED2x_odds"],
        "ppc_var": "y_treated_mu",
        "obs_var": "y_treated_data",
        "treatment_idx_var": "treatment_idx",
        "r2_func": r2_by_treatment_splicing,
    }
}


# Define which parameters are allowed to have treatment-specific priors for each model
TREATMENT_INDEXED_PARAMS = {
    1: {"logEC50", "slope"},  # Example: model 1 allows both logEC50 and slope to be treatment-specific
    2: {"logEC50"},           # Example: model 2 only allows logEC50 to be treatment-specific
}

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

def main(args=None):
    args = parse_args(args)
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate treatment-specific priors
    validate_treatment_specific_priors(args, args.model)

    logger.info(f"Reading input: {args.input}")
    df = pd.read_csv(args.input, sep="\t")
    if args.featureIDsToProcess is not None:
        logger.info(f"Filtering to features: {args.featureIDsToProcess}")
        df = df[df["featureID"].isin(args.featureIDsToProcess)]
    features_in_batch = df["featureID"].unique().tolist()
    logger.info(f"Features to process: {features_in_batch}")

    batch_idatas = {}
    summary_records = []
    filters = args.PosteriorFilter if args.PosteriorFilter else []
    prefilters = args.PreFilterByNumberReasonableObservedOutcomes if args.PreFilterByNumberReasonableObservedOutcomes else []

    for i, feature in enumerate(features_in_batch):
        logger.info(f"Processing feature: {feature}")
        feature_data = df[df["featureID"] == feature]

        # Calculate Spearman for each treatment (excluding treatments where all dose==0)
        spearman_dict = {}
        nonzero_treatments = [
            t for t in feature_data["treatment"].unique()
            if (feature_data[feature_data["treatment"] == t]["dose"] != 0).any()
        ]
        spearman_func = MODEL_CONFIG[args.model]["spearman_func"]
        for t in nonzero_treatments:
            t_data = feature_data[feature_data["treatment"] == t]
            if t_data["dose"].nunique() > 1:
                rho, _ = spearmanr(t_data["dose"], spearman_func(t_data))
            else:
                rho = np.nan
            spearman_dict[t] = rho

        row = {"feature": feature}
        for t, r in spearman_dict.items():
            row[f"spearman_{t}"] = r

        # Pre-filter by number of reasonable observed outcomes
        if prefilters:
            passed, reason = check_prefilter_by_number(df, feature, prefilters)
            logger.debug(f"Prefilter for {feature}: {passed}, {reason}")
            if not passed:
                row["status"] = reason
                summary_records.append(row)
                logger.info(f"Feature {feature} filtered out by prefilter: {reason}")
                continue

        treated = feature_data[feature_data["dose"] > 0]
        if treated.shape[0] < 3:
            row["status"] = "Did not fit; too few treated points"
            summary_records.append(row)
            logger.info(f"Feature {feature} filtered out: too few treated points")
            continue

        spearman_values = [abs(r) for r in spearman_dict.values() if not np.isnan(r)]
        if spearman_values:
            max_abs_spearman = np.nanmax(spearman_values)
        else:
            max_abs_spearman = np.nan
        logger.debug(f"Feature {feature} max_abs_spearman: {max_abs_spearman}")

        if np.isnan(max_abs_spearman) or max_abs_spearman < args.AbsSpearmanPreFilter:
            row["status"] = "Did not fit; low correlation"
            summary_records.append(row)
            logger.info(f"Feature {feature} filtered out: low correlation")
            continue

        try:
            fit_func = MODEL_CONFIG[args.model]["fit_func"]
            logger.info(f"Calling fit_func for {feature}")
            idata, model = fit_func(df, feature, samples=args.samples, args=args)
        except Exception as e:
            row["status"] = f"Model fit error: {e}"
            summary_records.append(row)
            logger.error(f"Model fit error for {feature}: {e}")
            continue

        passes, msg = check_posterior_filters(idata, filters)
        logger.debug(f"Posterior filter for {feature}: {passes}, {msg}")

        row["status"] = msg if not passes else "Success"

        rhat = az.rhat(idata)
        for var in MODEL_CONFIG[args.model]["summary_vars_scalar"]:
            row[f"{var}_mean"] = idata.posterior[var].mean().item()
            row[f"{var}_95hdi_lower"] = float(az.hdi(idata.posterior[var], hdi_prob=0.95)[var].sel(hdi="lower"))
            row[f"{var}_95hdi_upper"] = float(az.hdi(idata.posterior[var], hdi_prob=0.95)[var].sel(hdi="higher"))
            row[f"{var}_rhat"] = float(rhat[var].values)

        for var in MODEL_CONFIG[args.model]["summary_vars_treatment"]:
            means = idata.posterior[var].mean(dim=("chain", "draw"))
            hdi = az.hdi(idata.posterior[var], hdi_prob=0.95)
            for i, t in enumerate(idata.posterior.coords["treatment"].values):
                row[f"{var}_{t}_mean"] = means[i].item()
                row[f"{var}_{t}_95hdi_lower"] = hdi[var].sel(hdi="lower").values[i]
                row[f"{var}_{t}_95hdi_upper"] = hdi[var].sel(hdi="higher").values[i]
                row[f"{var}_{t}_rhat"] = float(rhat[var].values[i])

        r2_func = MODEL_CONFIG[args.model]["r2_func"]
        r2_dict = r2_func(idata, model)
        for t, r2 in r2_dict.items():
            row[f"posterior_predictive_R2_{t}"] = r2

        summary_records.append(row)

        if passes:
            batch_idatas[feature] = idata
            logger.info(f"Feature {feature} PASSED and added to batch_idatas")
        else:
            logger.info(f"Feature {feature} did NOT pass posterior filter: {msg}")

        logger.info(f"Done with feature: {i}:{feature} ({row['status']})")

    logger.info(f"Success status models to pickle: {list(batch_idatas.keys())}")
    with open(args.output_pkl, "wb") as f:
        pickle.dump(batch_idatas, f)

    pd.DataFrame(summary_records).to_csv(args.output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    if hasattr(sys, 'ps1'):
        main("--model 2 --input ExperimentGeneralizedDoseResponseModelling/Data/LeafcutterCounts/Exp11_TidyDataForModelling.tsv.gz --output_pkl scratch/Batch_0.results.pkl --output_tsv scratch/Batch_0.results.tsv.gz --PreFilterByNumberReasonableObservedOutcomes n 5 10 100000 --PreFilterByNumberReasonableObservedOutcomes y 3 3 100000 --featureIDsToProcess chr1:100007156:100009288:clu_7103_+ chr1:100007156:100011365:clu_7103_+ chr1:111454356:111456227:clu_7290_+ chr1:114410316:114421436:clu_2741_- chr10:5277294:5282776:clu_11419_- --PosteriorFilter MaxDeltaPSI 0.95 -1 -0.05 --PosteriorFilter MaxDeltaPSI 0.95 0.05 1".split(' '))
        main("--model 2 --input ExperimentGeneralizedDoseResponseModelling/Data/LeafcutterCounts/Exp11_TidyDataForModelling.tsv.gz --output_pkl scratch/Batch_0.results.pkl --output_tsv scratch/Batch_0.results.tsv.gz --PreFilterByNumberReasonableObservedOutcomes n 5 10 100000 --PreFilterByNumberReasonableObservedOutcomes y 3 3 100000 --featureIDsToProcess chr20:4691989:4699211:clu_56221_+ chr2:190977025:190978528:clu_50845_- chr6:110211859:110212133:clu_75667_+ --PosteriorFilter MaxDeltaPSI 0.95 -1 -0.05 --PosteriorFilter MaxDeltaPSI 0.95 0.05 1".split(' '))
        main("--model 1 --input ExperimentGeneralizedDoseResponseModelling/Data/GeneExpressionlog2FC/Exp11_TidyDataForModelling.tsv.gz --output_pkl scratch/Batch_0.results.pkl --output_tsv scratch/Batch_0.results.tsv.gz --featureIDsToProcess ACOT9 SMN2 TMEM158".split(' '))


    else:
        main()

#--prior_default logEC50 Uniform 5 6