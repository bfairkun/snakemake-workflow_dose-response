"""CLI: fit one batch of features."""
import argparse
import logging
import pickle
import sys
from collections import Counter, defaultdict

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
from ..filters import check_posterior_filters, check_prefilter_by_number, observed_abs_effect
from ..matrix_input import assemble_long, chunk_bounds, count_features
from ..fitting import (
    COVARIATE_INDEXED_SUMMARY_VARS,
    COVARIATE_SUPPORTED_MODELS,
    MODEL_CONFIG,
    MODEL_NAMES,
    TREATMENT_INDEXED_PARAMS,
    resolve_model,
    validate_covariate_args,
    validate_treatment_specific_priors,
)
from ..models.expression_absolute import fit_expression_absolute
from ..models.expression_logfc import fit_expression_logfc
from ..models.splicing_log2odds import fit_splicing_log2odds
from ..models.splicing_psi import fit_splicing_psi
from ..priors import get_prior_dist, parse_priors
from ..summarize import r2_by_treatment_expression, r2_by_treatment_splicing

__all__ = ["main", "parse_args", "setup_logging"]


EPILOG = """
MODELS
  Each fits a log-logistic curve in log10(dose), independently per feature, indexed by
  treatment arm so several arms of a series are fit jointly.

  Expression:
    expression_logfc      log2 fold-change; untreated level pinned at 0
    expression_absolute   absolute log2 abundance; free baseline

  Splicing -- a 2x2 over the response scale and what the covariate shifts. All four emit the
  same column set, and the `model` column of the output records which one ran.

    splicing_psi_vertical             logistic on PSI;       covariate shifts floor + ceiling
    splicing_psi_sharedceiling        logistic on PSI;       covariate shifts the floor only
    splicing_log2odds_vertical        logistic on log2-odds; covariate shifts floor + ceiling
    splicing_log2odds_sharedceiling   logistic on log2-odds; covariate shifts the floor only

  splicing_psi and splicing_log2odds are aliases for the first and last. Integer aliases
  1-6 also resolve.

  logEC50 is the native location: the log2-odds midpoint in the log2-odds models, the PSI
  midpoint in the PSI models. logEC50_PSI is always the PSI-halfway dose and is the column to
  use when comparing across models.

  Full specification with equations and priors: docs/models.qmd in this repo.

INPUT
  A long TSV, one row per feature x sample. Required: featureID, treatment, dose, plus the
  outcome columns for the chosen model -- `y` for the expression models, `y` and `n` for the
  splicing models. dose == 0 marks a sample as untreated; those rows get their own likelihood
  term and are what identify the baseline (and any covariate offset).

FILTERS
  Pre-fit filters are cheap and run before sampling:

    --AbsSpearmanPreFilter 0.4
        skip a feature unless |Spearman(dose, outcome)| reaches this in some arm

    --PreFilterByNumberReasonableObservedOutcomes VAR MIN_COUNT LOW HIGH
        require at least MIN_COUNT observations with VAR inside [LOW, HIGH]. Repeatable; all
        must pass. Bounds are in the units of VAR, so for log2 data they are log2 units.

  Post-fit filters drop a feature after sampling unless a posterior quantity clears a
  credible-interval test:

    --PosteriorFilter PARAM FRACTION LOW HIGH
        keep only if FRACTION of the posterior for PARAM lies inside [LOW, HIGH]. Repeatable;
        any one passing is enough, so use a pair for a two-sided effect.

  For splicing, filter on the change the assayed doses actually demonstrate rather than on the
  extrapolated asymptote:

    --PosteriorFilter dPSI_at_maxdose 0.95 0.1 1
    --PosteriorFilter dPSI_at_maxdose 0.95 -1 -0.1

PRIORS
  --prior PARAM TREATMENT FAMILY [ARGS...]     override for one arm ("ALL" for every arm)
  --prior_default PARAM FAMILY [ARGS...]       override the default for a parameter

    --prior_default hill LogNormal 0.405 0.35
    --prior logEC50 Branaplam Normal 1.0 0.5

  Families and their parameters, in order:
    Normal mu sigma; StudentT nu mu sigma; LogNormal mu sigma; Gamma alpha beta;
    Uniform lower upper; HalfNormal sigma; HalfCauchy beta; Beta alpha beta;
    Exponential lam

COVARIATES
  --covariates FILE
      Separate TSV, one row per sample, one column per covariate. Enters as a vertical offset
      applied to the treated AND the dose-0 rows with the same coefficient -- that shared
      coefficient across both likelihood terms is what identifies it.

      Consequences worth knowing before use:
        * the covariate must vary among the dose-0 samples, or it is not identified;
        * an all-ones column is rejected -- the baseline is a model parameter, not a covariate;
        * a column constant within a series is dropped, so one global file can serve every
          series and be inert where it does not apply;
        * coefficients are unscaled by default, so beta stays in the outcome's own units.

  --covariate_cols A B C        use only these columns
  --covariate_prior COL FAMILY [ARGS...]
  --scale_covariates            centre and scale continuous columns (indicators never scaled)
  --max_covariate_fraction 0.25 refuse if covariates exceed this fraction of the sample count

EXAMPLE
  dose-response-fit --model splicing_log2odds \
      --input DataBatched/Exp2/0.tsv.gz \
      --output_pkl out.pkl --output_tsv out.tsv.gz \
      --covariates config/dose_response_covariates.tsv \
      --PreFilterByNumberReasonableObservedOutcomes n 5 10 100000 \
      --PosteriorFilter dPSI_at_maxdose 0.95 0.1 1
"""


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="dose-response-fit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Fit a Bayesian dose-response model to one batch of features.",
        epilog=EPILOG,
    )
    parser.add_argument(
        '--model', required=True, metavar="MODEL",
        help="Model name, or its integer alias: "
             + ", ".join(f"{n} ({k})" for k, n in sorted(MODEL_NAMES.items(),
                                                         key=lambda kv: kv[1])))
    parser.add_argument('--inputlong', '--input', dest='inputlong', help="Batch input file with data. Required columns: featureID, dose, treatment, columns for outcome variables (y for the expression models; y and n for the splicing models). If dose is 0, the sample is considered untreated.")
    parser.add_argument('--output_pkl', required=True, help="Output pickle file")
    parser.add_argument('--output_tsv', required=True, help="Output summary tsv file")
    parser.add_argument('--matrix', nargs=2, action='append', metavar=('OUTCOME', 'PATH'), default=None, help="Feature-by-sample matrix supplying one outcome, e.g. --matrix y JuncCounts.bed.gz --matrix n Denom.bed.gz. Repeatable; 'y' is required and the splicing models also need 'n'. Two layouts are accepted and detected automatically from --design: a BED6+ file (six leading columns, as produced upstream) or a bare matrix of a featureID column plus one column per sample. Genome coordinates are never used, so they are not required. Mutually exclusive with --inputlong.")
    parser.add_argument('--design', default=None, help="Sample-level TSV with columns sample, treatment, dose -- one row per sample in this series. Required with --matrix. This is what selects the matrix columns, so restricting a series to its own samples needs nothing else.")
    parser.add_argument('--sorted', dest='sorted_input', action='store_true', help="The matrices are sorted by featureID and have a .fidx sidecar; seek to the chunk instead of reading the whole matrix. Off by default, in which case the matrix is read into memory and sliced, which is correct whatever its order. Opt-in because an unsorted or stale-indexed matrix would otherwise silently yield the wrong rows.")
    parser.add_argument('--feature_col', default=None, help="Name of the feature column in the matrices. Inferred as featureID, else name, else junc, else the first non-sample column.")
    parser.add_argument('--chunks', nargs=2, type=int, metavar=('N', 'M'), default=None, help="Process only chunk N of M, splitting features into M contiguous chunks with the remainder spread over the leading chunks. N=0 writes just the header, so concatenating the outputs of chunks 0..M reproduces an unchunked run exactly.")
    parser.add_argument('--featureIDsToProcess', nargs='+', default=None, help="Optional: Only process these featureIDs (space-separated list or use multiple times).")
    parser.add_argument('--samples', type=int, default=1000, help="Number of samples to draw from the posterior")
    parser.add_argument('--MaxFitErrors', type=int, default=0, help="Exit non-zero if more than this many features raise an exception during fitting. Such errors are usually environmental (most often the PyTensor compile cache being deleted mid-run) rather than properties of the data, and would otherwise be recorded silently in the per-feature 'status' column while the job still exits 0. Set to -1 to tolerate any number.")
    parser.add_argument('--AbsSpearmanPreFilter', type=float, default=0.4, help="Minimum |Spearman correlation| between dose and y to attempt fitting")
    parser.add_argument('--MinObservedAbsEffect', type=float, default=None, help="Skip fitting a feature whose OBSERVED outcome barely moves: the largest |mean(outcome at an arm's top dose) - mean(outcome in a dose-0 group)|, maximised over treated arms and control groups, must reach this value. Units follow the model's outcome (PSI for the splicing models, log2 for the expression models). This is a necessary condition for a demonstrated-change posterior filter such as --PosteriorFilter dPSI_at_maxdose 0.95 0.1 1, since the fitted effect at the top dose is anchored on these same observations; it exists because such features are both discarded and disproportionately slow to sample (an unidentified posterior forces a tiny step size). A feature whose baseline or top dose is unmeasurable is always kept. Off by default.")
    parser.add_argument(
        '--PosteriorFilter', nargs=4, action='append',
        metavar=('param', 'fraction', 'low', 'high'),
        help=(
            "Posterior filter criteria, e.g. --PosteriorFilter dPSI_at_maxdose 0.95 0.1 1. "
            "Can be specified multiple times for the same parameter for multi-interval (two-sided) filtering, "
            "e.g. --PosteriorFilter span_log2 0.95 -100 -1 --PosteriorFilter span_log2 0.95 1 100"
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
            "--prior logEC50 Risdiplam Normal 2.0 0.5\n"
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
            "(added to log2 abundance in the expression models, to log2-odds in the splicing models), applied to the "
            "dose-0 observations as well as the treated ones -- the control contrast is what "
            "identifies the coefficients. Not supported for expression_logfc, which pins the untreated level at 0 and so has no free baseline."
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
            "--covariate_prior IFNa Normal 0 2.0. Default Normal(0, 4) in the splicing models, Normal(0, 3) in expression_absolute. "
            "Prior scales are per covariate rather than shared."
        )
    )
    parser.add_argument(
        '--scale_covariates', action='store_true',
        help=(
            "Z-scale CONTINUOUS covariate columns (>2 distinct values). Off by default so that "
            "beta stays directly interpretable: log2 units in the expression models, doublings of the inclusion/exclusion odds in the splicing models. "
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

def validate_input_args(args):
    if bool(args.matrix) == bool(args.inputlong):
        raise ValueError("Give exactly one of --inputlong or --matrix.")
    if args.matrix:
        outcomes = [o for o, _ in args.matrix]
        if "y" not in outcomes:
            raise ValueError(f"--matrix must supply 'y'; got {outcomes}.")
        if len(set(outcomes)) != len(outcomes):
            raise ValueError(f"--matrix outcomes must be unique; got {outcomes}.")
        needs_n = MODEL_CONFIG[args.model]["spearman_func"] is not None and args.model in (2, 4, 5, 6)
        if needs_n and "n" not in outcomes:
            raise ValueError(
                f"model {MODEL_CONFIG[args.model]['name']!r} is a splicing model and needs a "
                f"denominator, so --matrix n <path> is required; got {outcomes}."
            )
        if not args.design:
            raise ValueError("--design is required with --matrix.")
    elif args.sorted_input or args.feature_col:
        raise ValueError("--sorted and --feature_col only apply to --matrix input.")


def write_header_only(args, logger):
    """Emit chunk 0: the summary header, an empty pickle, and nothing else."""
    import pickle

    treatments = []
    if args.design:
        design = pd.read_csv(args.design, sep="\t")
        treated = design[pd.to_numeric(design["dose"], errors="coerce").fillna(0) > 0]
        treatments = sorted(treated["treatment"].astype(str).unique())

    cfg = MODEL_CONFIG[args.model]
    cols = ["feature", "model", "covariates_used"]
    cols += [f"spearman_{t}" for t in treatments] + ["status"]
    for var in cfg["summary_vars_scalar"]:
        cols += [f"{var}_{suffix}" for suffix in ("mean", "95hdi_lower", "95hdi_upper", "rhat")]
    for var in cfg["summary_vars_treatment"]:
        for t in treatments:
            cols += [f"{var}_{t}_{suffix}" for suffix in ("mean", "95hdi_lower", "95hdi_upper", "rhat")]
    cols += [f"posterior_predictive_R2_{t}" for t in treatments]

    pd.DataFrame(columns=cols).to_csv(args.output_tsv, sep="\t", index=False)
    with open(args.output_pkl, "wb") as fh:
        pickle.dump({}, fh)
    logger.info(f"chunk 0: wrote header with {len(cols)} columns and an empty pickle")


def load_input(args, logger):
    """Return the long table, from either --inputlong or --matrix + --design."""
    source_start, source_count = 0, None
    if args.matrix:
        design = pd.read_csv(args.design, sep="\t")
        matrices = {outcome: path for outcome, path in args.matrix}
        total = count_features(next(iter(matrices.values())))
        if args.chunks:
            n, m = args.chunks
            source_start, source_count = chunk_bounds(total, n, m)
        else:
            source_count = total
        logger.info(f"Reading rows [{source_start}, {source_start + source_count}) of {total} from "
                    f"{len(matrices)} matri{'x' if len(matrices) == 1 else 'ces'}"
                    f"{' via the sorted index' if args.sorted_input else ''}")
        return assemble_long(matrices, design, source_start, source_count,
                             sorted_access=args.sorted_input, feature_col=args.feature_col)

    logger.info(f"Reading input: {args.inputlong}")
    df = pd.read_csv(args.inputlong, sep="\t")
    if args.chunks:
        n, m = args.chunks
        features = df["featureID"].drop_duplicates().tolist()
        start, count = chunk_bounds(len(features), n, m)
        keep = set(features[start:start + count])
        df = df[df["featureID"].isin(keep)]
    return df


def main(args=None):
    args = parse_args(args)
    args.model = resolve_model(args.model)
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate treatment-specific priors
    validate_treatment_specific_priors(args, args.model)
    validate_covariate_args(args, args.model)
    validate_input_args(args)

    # Chunk 0 carries the header and no rows, so that concatenating chunks 0..M reproduces an
    # unchunked run. It needs no data at all, hence the early exit.
    if args.chunks and args.chunks[0] == 0:
        write_header_only(args, logger)
        return

    df = load_input(args, logger)
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
                "--covariates was given but the input has no 'sample' "
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

        row = {"feature": feature, "model": MODEL_CONFIG[args.model]["name"]}
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

        if args.MinObservedAbsEffect is not None:
            effect = observed_abs_effect(feature_data, treated,
                                         MODEL_CONFIG[args.model]["spearman_func"])
            if np.isfinite(effect) and effect < args.MinObservedAbsEffect:
                row["status"] = (f"Did not fit; observed |effect| at top dose {effect:.4f} "
                                 f"< {args.MinObservedAbsEffect}")
                summary_records.append(row)
                logger.info(f"Feature {feature} filtered out: observed effect {effect:.4f}")
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

    errored = [r for r in summary_records
               if str(r.get("status", "")).startswith("Model fit error")]
    if errored and args.MaxFitErrors >= 0 and len(errored) > args.MaxFitErrors:
        signatures = Counter(str(r["status"])[:200] for r in errored)
        logger.error(
            f"{len(errored)}/{len(summary_records)} features failed to fit with an "
            f"exception, above the --MaxFitErrors threshold of {args.MaxFitErrors}. "
            "Distinct signatures:"
        )
        for sig, n in signatures.most_common():
            logger.error(f"  [{n}x] {sig}")
        sys.exit(1)


if __name__ == "__main__":
    main()
