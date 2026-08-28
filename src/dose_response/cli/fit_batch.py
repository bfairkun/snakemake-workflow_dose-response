"""CLI: fit one batch of features.

`main` is left whole rather than split into an io module: Phase A of the refactor is a
mechanical move, and carving up the driver would add risk for no analytical gain.

Moved verbatim from scripts/BayesianDoseResponse_ByBatch.py; behaviour is unchanged.
"""
import argparse
import logging
import pickle
import sys
from collections import defaultdict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import spearmanr

from ..covariates import (
    DEFAULT_COVARIATE_PRIOR_SD,
    CovariateDesignError,
    design_for_feature,
    parse_covariate_priors,
    prepare_covariates,
)
from ..filters import check_posterior_filters, check_prefilter_by_number
from ..fitting import (
    COVARIATE_INDEXED_SUMMARY_VARS,
    COVARIATE_SUPPORTED_MODELS,
    MODEL_CONFIG,
    TREATMENT_INDEXED_PARAMS,
    validate_covariate_args,
    validate_treatment_specific_priors,
)
from ..models.expression_absolute import fit_expression_absolute_model
from ..models.expression_logfc import fit_gene_expression_model
from ..models.splicing_psi import fit_splicing_model
from ..priors import get_prior_dist, parse_priors
from ..summarize import r2_by_treatment_expression, r2_by_treatment_splicing

__all__ = ["main", "parse_args", "setup_logging"]


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Fit Bayesian Dose Response Model for Gene Expression by Batch")
    parser.add_argument(
        '--model', type=int, required=True, choices=[1, 2, 3],
        help="Which model to use:\n\n1: Intended to model expression. The outcome variable y represents the log2 fold change in expression. The model uses a three-parameter log-logistic dose–response function to predict y, where the slope and the EC50 (the dose at which half the maximal effect is observed) vary by treatment, while the upper asymptote (maximum effect) is shared across treatments.\n\n2: Intended to model splicing. The outcome is a count of inclusion reads y out of total reads n for each observation. We model this using a beta-binomial likelihood to account for overdispersion, where the mean inclusion proportion (PSI) is linked to dose using a four-parameter log-logistic function. In this model, the EC50 varies by treatment, while the upper and lower asymptotes and the slope are shared across treatments.\n\n3: Intended to model expression on an ABSOLUTE log2 abundance scale (e.g. log2 TMM-CPM) rather than a log2 fold change. Identical curve shape to model 1, but the untreated level is a free parameter `lower` instead of being pinned at 0, so baseline uncertainty is propagated instead of assumed away, and `Delta` (= upper - lower) carries the effect size. Note `Delta` in model 3 is the same quantity model 1 calls `upper`, so their priors are directly comparable. Supports optional sample x covariate terms."
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
    # --- Optional sample x covariate terms (models 2 and 3) ---------------------------
    # The covariate matrix is always OPTIONAL and always ADDITIONAL. The baseline/intercept
    # is a parameter of the model itself (`lower`), never a covariate column -- do not supply
    # an all-ones column, it is rejected as collinear with `lower`.
    parser.add_argument(
        '--covariates', default=None,
        help=(
            "Optional TSV of sample x covariate values (one row per sample, a 'sample' column "
            "plus one column per covariate). Covariates enter as a VERTICAL offset only "
            "(added to log2 abundance for model 3, to logit(PSI) for model 2), applied to the "
            "dose-0 observations as well as the treated ones -- the control contrast is what "
            "identifies the coefficients. Not supported for model 1, which has no free intercept."
        )
    )
    parser.add_argument(
        '--covariate_cols', nargs='+', default=None,
        help="Which columns of --covariates to use (default: all non-'sample' columns)."
    )
    parser.add_argument(
        '--covariate_prior', nargs='+', action='append', metavar='COVARIATE_PRIOR_SPEC',
        help=(
            "Set the prior for one covariate's coefficient, e.g. "
            "--covariate_prior IFNa Normal 0 2.0. Default: Normal(0, 3), matching the default "
            "prior on Delta, since both are effects in log2 units. "
            "Prior scales are per covariate rather than shared."
        )
    )
    parser.add_argument(
        '--scale_covariates', action='store_true',
        help=(
            "Z-scale CONTINUOUS covariate columns (>2 distinct values). Off by default so that "
            "beta stays directly interpretable as the effect in log2 (or logit-PSI) units. "
            "Binary 0/1 indicators are never scaled even when this is set: dividing by "
            "sqrt(p(1-p)) would make the coefficient depend on design balance, which differs "
            "between series, so betas would stop being comparable across series."
        )
    )
    parser.add_argument(
        '--max_covariate_fraction', type=float, default=0.25,
        help=(
            "Refuse to fit if the number of covariates exceeds this fraction of the samples in "
            "the series (default 0.25). Every covariate costs one parameter PER FEATURE and "
            "features are fit independently, so no strength is borrowed across genes."
        )
    )
    parser.add_argument(
        '--covariate_target', default='vertical', choices=['vertical'],
        help=(
            "Where covariates enter. Only 'vertical' is supported. A horizontal (logEC50) "
            "covariate is deliberately not offered: EC50 is the estimand, and logEC50 is "
            "already indexed by treatment, so a treatment-aligned covariate on it would be "
            "exactly collinear with the existing per-treatment parameters."
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

def main(args=None):
    args = parse_args(args)
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate treatment-specific priors
    validate_treatment_specific_priors(args, args.model)
    validate_covariate_args(args, args.model)

    logger.info(f"Reading input: {args.input}")
    df = pd.read_csv(args.input, sep="\t")
    if args.featureIDsToProcess is not None:
        logger.info(f"Filtering to features: {args.featureIDsToProcess}")
        df = df[df["featureID"].isin(args.featureIDsToProcess)]
    features_in_batch = df["featureID"].unique().tolist()
    # sort=False keeps each group in the incoming row order, which the batch splitter has
    # already arranged as controls first, then treatment, then dose.
    feature_groups = dict(tuple(df.groupby("featureID", sort=False)))
    logger.info(f"Features to process: {features_in_batch}")

    # Build and validate the covariate design ONCE per batch. Validating against the batch
    # rather than the global covariate file is deliberate: series are fit separately, and
    # approaches do not see the same samples (exclude_expression drops degraded libraries from
    # expression but not splicing), so which columns are identifiable is a per-batch question.
    args.cov_spec = None
    if getattr(args, "covariates", None):
        if "sample" not in df.columns:
            raise CovariateDesignError(
                f"--covariates was given but the tidy data {args.input!r} has no 'sample' "
                "column to join on. All built-in transforms emit one."
            )
        args.cov_spec = prepare_covariates(
            args.covariates,
            args.covariate_cols,
            df,
            scale_covariates=args.scale_covariates,
            max_covariate_fraction=args.max_covariate_fraction,
        )

    batch_idatas = {}
    summary_records = []
    filters = args.PosteriorFilter if args.PosteriorFilter else []
    prefilters = args.PreFilterByNumberReasonableObservedOutcomes if args.PreFilterByNumberReasonableObservedOutcomes else []

    for i, feature in enumerate(features_in_batch):
        logger.info(f"Processing feature: {feature}")
        feature_data = feature_groups[feature]

        # Spearman per treatment, skipping treatments that are controls-only.
        spearman_dict = {}
        spearman_func = MODEL_CONFIG[args.model]["spearman_func"]
        for t, t_data in feature_data.groupby("treatment", sort=False):
            if not (t_data["dose"] != 0).any():
                continue                                  # treatment is controls-only
            if t_data["dose"].nunique() > 1:
                rho, _ = spearmanr(t_data["dose"], spearman_func(t_data))
            else:
                rho = np.nan
            spearman_dict[t] = rho

        row = {"feature": feature}
        if args.cov_spec is not None:
            # Record the design so a fit is self-describing when read back months later.
            row["covariates_used"] = args.cov_spec.describe()
        for t, r in spearman_dict.items():
            row[f"spearman_{t}"] = r

        # Pre-filter by number of reasonable observed outcomes
        if prefilters:
            passed, reason = check_prefilter_by_number(feature_data, prefilters)
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
            idata, model = fit_func(feature_data, samples=args.samples, args=args)
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

        # Covariate coefficients, when covariates are in use. Absent from the posterior
        # otherwise, so this loop is a no-op for the default (covariate-free) runs.
        for var in COVARIATE_INDEXED_SUMMARY_VARS.get(args.model, []):
            if var not in idata.posterior:
                continue
            means = idata.posterior[var].mean(dim=("chain", "draw"))
            hdi = az.hdi(idata.posterior[var], hdi_prob=0.95)
            for i, c in enumerate(idata.posterior.coords["covariate"].values):
                row[f"{var}_{c}_mean"] = means[i].item()
                row[f"{var}_{c}_95hdi_lower"] = hdi[var].sel(hdi="lower").values[i]
                row[f"{var}_{c}_95hdi_upper"] = hdi[var].sel(hdi="higher").values[i]
                row[f"{var}_{c}_rhat"] = float(rhat[var].values[i])

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
    main()
