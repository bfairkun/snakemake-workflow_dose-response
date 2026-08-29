"""Design-matrix construction and validation for optional sample x covariate terms.

Covariates live in a separate TSV, one row per sample, and enter the models as a vertical
offset applied to the treated and dose-0 likelihoods alike. Deliberately free of PyMC so the
validation rules stay cheap to unit-test. See docs/models.qmd.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Indicators are never scaled: that would make beta depend on design balance.
_MAX_DISTINCT_FOR_BINARY = 2

# Warn when a covariate is nearly indistinguishable from the dose design itself.
_DOSE_COLLINEARITY_WARN = 0.8

# Matches the default prior on the span; both are an effect in log2 units.
DEFAULT_COVARIATE_PRIOR_SD = 3.0


class CovariateDesignError(ValueError):
    """Raised when a covariate design matrix cannot be used safely.

    Always a hard failure: a silently-unidentified parameter in a per-feature fit that runs
    across 200 batches is far more expensive to discover later than a crash now.
    """


@dataclass
class CovariateSpec:
    """A validated covariate design, fixed for one batch (i.e. one series x approach)."""

    frame: pd.DataFrame                      # index = sample, columns = kept covariates (unscaled)
    columns: List[str]                       # final kept column order
    center: Dict[str, float] = field(default_factory=dict)   # per-column, 0.0 if not scaled
    scale: Dict[str, float] = field(default_factory=dict)    # per-column, 1.0 if not scaled
    dropped: Dict[str, str] = field(default_factory=dict)    # column -> reason
    warnings: List[str] = field(default_factory=list)

    @property
    def n_covariates(self) -> int:
        return len(self.columns)

    def describe(self) -> str:
        """One-line, log/TSV-friendly summary so a fit is self-describing when read back."""
        parts = []
        for c in self.columns:
            if self.scale.get(c, 1.0) != 1.0 or self.center.get(c, 0.0) != 0.0:
                parts.append(f"{c}(center={self.center[c]:.4g},scale={self.scale[c]:.4g})")
            else:
                parts.append(c)
        out = "+".join(parts) if parts else "none"
        if self.dropped:
            out += " | dropped:" + ",".join(f"{k}({v})" for k, v in self.dropped.items())
        return out

    def matrix(self, samples: Sequence[str]) -> np.ndarray:
        """Return the (n_obs, K) design matrix for the given per-observation sample IDs."""
        if not self.columns:
            return np.zeros((len(samples), 0), dtype=float)
        sub = self.frame.loc[list(samples), self.columns].to_numpy(dtype=float)
        for j, c in enumerate(self.columns):
            sub[:, j] = (sub[:, j] - self.center.get(c, 0.0)) / self.scale.get(c, 1.0)
        return sub


def load_covariate_table(path: str, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read the covariate TSV and return it indexed by sample.

    Requires a `sample` column. If `cols` is given, only those columns are kept (and all must
    exist); otherwise every non-`sample` column is used.
    """
    df = pd.read_csv(path, sep="\t")
    if "sample" not in df.columns:
        raise CovariateDesignError(
            f"Covariate file {path!r} has no 'sample' column; found {list(df.columns)}. "
            "The file must have one row per sample and one column per covariate."
        )
    if df["sample"].duplicated().any():
        dupes = sorted(df.loc[df["sample"].duplicated(), "sample"].unique())
        raise CovariateDesignError(
            f"Covariate file {path!r} has duplicate sample rows: {dupes}. Expected one row per sample."
        )
    df = df.set_index("sample")
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise CovariateDesignError(
                f"Requested covariate column(s) {missing} not in {path!r}; "
                f"available: {list(df.columns)}"
            )
        df = df[list(cols)]
    return df


def prepare_covariates(
    path: str,
    cols: Optional[Sequence[str]],
    batch_df: pd.DataFrame,
    scale_covariates: bool = False,
    max_covariate_fraction: Optional[float] = None,
) -> CovariateSpec:
    """Validate covariates against the samples actually present in this batch.

    `batch_df` is the tidy batch table (columns: sample, dose, treatment, ...). Validation is
    done against the batch rather than the global covariate file on purpose: series are fit
    separately and approaches do not see the same samples (e.g. `exclude_expression` removes
    degraded libraries from expression but not splicing), so which columns are identifiable
    is a per-(series x approach) question.

    Raises CovariateDesignError on any condition that would leave a parameter unidentified.
    """
    raw = load_covariate_table(path, cols)

    samples = pd.unique(batch_df["sample"])
    missing = [s for s in samples if s not in raw.index]
    if missing:
        raise CovariateDesignError(
            f"{len(missing)} sample(s) in the data have no row in the covariate file {path!r}: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}. "
            "Complete covariates are required -- with only a handful of samples per series, "
            "silently dropping one is worse than failing."
        )

    frame = raw.loc[samples]

    # --- non-numeric / missing values -------------------------------------------------
    non_numeric = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise CovariateDesignError(
            f"Covariate column(s) {non_numeric} are not numeric. Dummy-code k-level factors "
            "as k-1 indicator columns in the TSV yourself (and do not add an intercept column)."
        )
    na_cols = [c for c in frame.columns if frame[c].isna().any()]
    if na_cols:
        raise CovariateDesignError(
            f"Covariate column(s) {na_cols} contain missing values for samples in this batch. "
            "Complete covariates are required."
        )

    dose = pd.to_numeric(batch_df["dose"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    per_obs_samples = batch_df["sample"].to_numpy()
    is_control = dose == 0

    kept: List[str] = []
    dropped: Dict[str, str] = {}
    warns: List[str] = []

    control_samples = pd.unique(per_obs_samples[is_control])
    treated_samples = pd.unique(per_obs_samples[~is_control])

    for c in frame.columns:
        vals = frame[c].to_numpy(dtype=float)
        distinct = np.unique(vals)

        # An all-ones column is a user trying to supply an intercept. `lower` is already a
        # first-class model parameter, so this would be exactly collinear with it.
        if distinct.size == 1:
            if np.isclose(distinct[0], 1.0):
                raise CovariateDesignError(
                    f"Covariate column {c!r} is all ones. Do not supply an intercept/offset "
                    "column: the baseline (`lower`) is already a free parameter of the model, "
                    "and an all-ones column would be exactly collinear with it."
                )
            dropped[c] = f"constant({distinct[0]:g}) in this series"
            continue

        # The identification rule, made operational. beta is identified by the control
        # contrast; if the covariate does not vary among the dose-0 samples, that contrast
        # does not exist and beta is left leaning on the treated arm's baseline plateau.
        ctrl_vals = frame.loc[control_samples, c].to_numpy(dtype=float)
        if ctrl_vals.size == 0:
            raise CovariateDesignError(
                "This batch contains no dose-0 (control) observations at all, so no covariate "
                "coefficient can be identified. Check that the batch is non-empty and that the "
                "series' control samples survived any upstream filtering."
            )
        if np.unique(ctrl_vals).size == 1:
            raise CovariateDesignError(
                f"Covariate column {c!r} does not vary among the dose-0 (control) samples of "
                "this series. The control contrast is what identifies its coefficient, so "
                "without it beta would be identified only through the treated arm's baseline "
                "plateau -- exactly the leak-prone direction this design avoids. "
                "Either drop the column for this series, or add controls that vary in it."
            )

        kept.append(c)

    if not kept:
        # Not an error: the covariate file is global but series are fit separately, so a
        # column that is meaningful for one series (e.g. IFNa) is legitimately constant in
        # another (every Bo series is noStim). Fall back to a covariate-free fit -- which
        # builds the identical model graph -- but say so loudly, and record it in the summary
        # TSV via describe(), so a per-series difference in the fitted model is never silent.
        msg = (
            "no usable covariate columns for this series (all constant); "
            f"falling back to a covariate-free fit. Dropped: {dropped}"
        )
        logger.warning("Covariates: %s", msg)
        return CovariateSpec(
            frame=frame, columns=[], center={}, scale={}, dropped=dropped, warnings=[msg]
        )

    # --- budget -----------------------------------------------------------------------
    n_samples = len(samples)
    if max_covariate_fraction is not None:
        budget = max_covariate_fraction * n_samples
        if len(kept) > budget:
            raise CovariateDesignError(
                f"{len(kept)} covariate(s) requested but this series has only {n_samples} "
                f"samples; --max_covariate_fraction {max_covariate_fraction} allows at most "
                f"{int(budget)}. Every covariate adds one parameter PER FEATURE, and features "
                "are fit independently, so there is no strength borrowed across genes."
            )

    # Rank of [intercept | X], deliberately excluding treatment dummies: the models have no
    # per-treatment intercept, so a stimulus covariate collinear with them is expected.
    Xfull = frame.loc[per_obs_samples, kept].to_numpy(dtype=float)
    stacked = np.column_stack([np.ones(len(per_obs_samples)), Xfull])
    rank = np.linalg.matrix_rank(stacked)
    if rank < stacked.shape[1]:
        raise CovariateDesignError(
            f"Design matrix [intercept | {kept}] is rank deficient "
            f"(rank {rank} < {stacked.shape[1]} columns). At least one covariate is a linear "
            "combination of the others (or of a constant), so its coefficient is not identified."
        )

    # A covariate that tracks dose is nearly indistinguishable from a change in the plateau.
    if treated_samples.size > 1:
        log_dose = np.log10(dose[~is_control])
        for c in kept:
            v = frame.loc[per_obs_samples[~is_control], c].to_numpy(dtype=float)
            if np.unique(v).size > 1 and np.unique(log_dose).size > 1:
                r = float(np.corrcoef(v, log_dose)[0, 1])
                if abs(r) > _DOSE_COLLINEARITY_WARN:
                    warns.append(
                        f"covariate {c!r} correlates with log10(dose) among treated samples "
                        f"(r={r:.2f}); a vertical offset that tracks dose is hard to separate "
                        "from a change in the upper asymptote"
                    )

    # --- scaling -----------------------------------------------------------------------
    center: Dict[str, float] = {c: 0.0 for c in kept}
    scale: Dict[str, float] = {c: 1.0 for c in kept}
    if scale_covariates:
        for c in kept:
            vals = frame[c].to_numpy(dtype=float)
            if np.unique(vals).size <= _MAX_DISTINCT_FOR_BINARY:
                continue  # never scale indicators; keeps beta in interpretable units
            sd = float(vals.std(ddof=1))
            if sd > 0:
                center[c] = float(vals.mean())
                scale[c] = sd

    spec = CovariateSpec(
        frame=frame, columns=kept, center=center, scale=scale, dropped=dropped, warnings=warns
    )

    logger.info("Covariates: %s (%d sample(s))", spec.describe(), n_samples)
    for w in warns:
        logger.warning("Covariate check: %s", w)
    return spec


def design_for_feature(spec: Optional[CovariateSpec], feature_df: pd.DataFrame):
    """Return (X_treated, X_untreated) for one feature, row-aligned to the model's arrays.

    Row order must match how the fitting functions split the feature's rows: treated rows are
    those with a non-null, non-zero dose, in the order they appear; untreated are the rest.
    Returns (None, None) when no covariates are in use, so callers can guard on it and build
    an unchanged model graph.
    """
    if spec is None or spec.n_covariates == 0:
        return None, None
    dose = pd.to_numeric(feature_df["dose"], errors="coerce")
    is_treated = dose.notna() & (dose != 0)
    Xt = spec.matrix(feature_df.loc[is_treated, "sample"].to_numpy())
    Xu = spec.matrix(feature_df.loc[~is_treated, "sample"].to_numpy())
    return Xt, Xu


def parse_covariate_priors(specs) -> Dict[str, tuple]:
    """Parse repeated --covariate_prior COL FAMILY P... into {column: (family, [params])}.

    Mirrors the existing --prior / --prior_default idiom in BayesianDoseResponse_ByBatch.py.
    """
    out: Dict[str, tuple] = {}
    for entry in specs or []:
        if len(entry) < 3:
            raise CovariateDesignError(
                f"--covariate_prior needs COLUMN FAMILY PARAM...; got {entry}"
            )
        col, family, *params = entry
        out[col] = (family, [float(p) for p in params])
    return out
