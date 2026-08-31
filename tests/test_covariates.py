"""Tests for dose_response.covariates.

These cover the identifiability rules, which is where the real risk lives: a silently
unidentified covariate in a per-feature fit spread over 200 batches is expensive to
discover after the fact.
"""
import numpy as np
import pandas as pd
import pytest

from dose_response.covariates import (
    CovariateDesignError,
    design_for_feature,
    parse_covariate_priors,
    prepare_covariates,
)


def test_happy_path(batch, covariates, write_tsv):
    spec = prepare_covariates(write_tsv(covariates), None, batch)
    assert spec.columns == ["IFNa", "IFNg"]
    assert spec.n_covariates == 2
    # unscaled by default, so beta stays in interpretable log2-odds units
    assert spec.scale["IFNa"] == 1.0
    assert spec.center["IFNa"] == 0.0


def test_design_split_matches_model_row_order(batch, covariates, write_tsv):
    spec = prepare_covariates(write_tsv(covariates), None, batch)
    Xt, Xu = design_for_feature(spec, batch)
    assert Xt.shape == (6, 2)          # 3 stimuli x 2 doses
    assert Xu.shape == (6, 2)          # 3 stimuli x 2 controls
    # untreated rows carry the SAME covariate values -- this is what identifies beta
    assert Xu.sum(axis=0).tolist() == [2, 2]
    assert Xt.sum(axis=0).tolist() == [2, 2]


def test_no_covariates_returns_none(batch):
    assert design_for_feature(None, batch) == (None, None)


@pytest.mark.parametrize("constant", [0, 1])
def test_constant_column_is_fatal(batch, covariates, write_tsv, constant):
    # The matrix handed to the fitter holds one series' DECLARED covariates, so a column with
    # no variation is a design error to report, not something to drop quietly. Value is
    # irrelevant: all-ones (every sample shares the condition, as in GSE304951 split by arm)
    # is as unidentified as all-zeros.
    covariates["U1CKD"] = constant
    with pytest.raises(CovariateDesignError, match="(?i)constant"):
        prepare_covariates(write_tsv(covariates), None, batch)


def test_no_declared_covariates_is_covariate_free(batch, write_tsv):
    # Header-only matrix: CreateSeriesCovariateMatrix emits this for a series with no rows.
    import pandas as pd
    spec = prepare_covariates(write_tsv(pd.DataFrame({"sample": []})), None, batch)
    assert spec.columns == []
    assert spec.dropped == {}




def test_covariate_free_series_builds_no_design(batch, write_tsv):
    """A series with no declarations is inert: no design, so the model graph is unchanged."""
    spec = prepare_covariates(write_tsv(pd.DataFrame({"sample": []})), None, batch)
    assert spec.n_covariates == 0
    assert design_for_feature(spec, batch) == (None, None)


def test_no_variance_among_controls_rejected(batch, covariates, write_tsv):
    """The core identification rule: beta is pinned by the control contrast."""
    covariates["treated_only"] = (batch["dose"] > 0).astype(int) * covariates["IFNa"]
    with pytest.raises(CovariateDesignError, match="does not vary among the dose-0"):
        prepare_covariates(write_tsv(covariates), None, batch)


@pytest.mark.parametrize("mutate,needle", [
    (lambda cov: cov.iloc[:-1],                        "no row in the covariate file"),
    (lambda cov: cov.assign(IFNa=[np.nan] + [0]*11),   "missing values"),
    (lambda cov: cov.assign(stim="IFNa"),              "not numeric"),
    (lambda cov: cov.assign(IFNa_copy=cov["IFNa"]),    "rank deficient"),
], ids=["missing_sample", "na", "non_numeric", "duplicate_column"])
def test_rejected_inputs(batch, covariates, write_tsv, mutate, needle):
    with pytest.raises(CovariateDesignError, match=needle):
        prepare_covariates(write_tsv(mutate(covariates)), None, batch)


def test_budget_enforced(batch, covariates, write_tsv):
    path = write_tsv(covariates)
    # 12 samples, fraction 0.1 -> at most 1 covariate
    with pytest.raises(CovariateDesignError, match="max_covariate_fraction"):
        prepare_covariates(path, None, batch, max_covariate_fraction=0.1)
    prepare_covariates(path, None, batch, max_covariate_fraction=0.5)   # 2 <= 6, fine


def test_missing_requested_column(batch, covariates, write_tsv):
    with pytest.raises(CovariateDesignError, match="not in"):
        prepare_covariates(write_tsv(covariates), ["nope"], batch)


def test_binary_never_scaled_continuous_is(batch, covariates, write_tsv):
    rng = np.random.default_rng(0)
    covariates["degradation"] = rng.normal(size=len(covariates)) * 2 + 5
    spec = prepare_covariates(write_tsv(covariates), None, batch, scale_covariates=True)
    assert spec.scale["IFNa"] == 1.0, "binary indicator must not be scaled"
    assert spec.center["IFNa"] == 0.0
    assert spec.scale["degradation"] != 1.0, "continuous column should be scaled"
    X = spec.matrix(batch["sample"].to_numpy())
    j = spec.columns.index("degradation")
    assert abs(X[:, j].mean()) < 1e-9, "scaled continuous column should be centered"


def test_scaling_off_by_default(batch, covariates, write_tsv):
    rng = np.random.default_rng(0)
    covariates["degradation"] = rng.normal(size=len(covariates)) * 2 + 5
    spec = prepare_covariates(write_tsv(covariates), None, batch)
    assert spec.scale["degradation"] == 1.0, "must not scale unless asked"


def test_dose_collinearity_warns_not_fatal(batch, covariates, write_tsv):
    # high at the top dose, but varying among controls so it passes the hard check
    covariates["highdose_ish"] = (batch["dose"] >= 100).astype(float)
    covariates.loc[batch["dose"] == 0, "highdose_ish"] = [0, 1, 0, 1, 0, 1]
    spec = prepare_covariates(write_tsv(covariates), None, batch)
    assert "highdose_ish" in spec.columns
    assert any("log10(dose)" in w for w in spec.warnings)


def test_describe_roundtrip(batch, covariates, write_tsv):
    spec = prepare_covariates(write_tsv(covariates), None, batch)
    d = spec.describe()
    assert "IFNa" in d and "IFNg" in d


def test_parse_covariate_priors():
    assert parse_covariate_priors([["IFNa", "Normal", "0", "2.0"]]) == {
        "IFNa": ("Normal", [0.0, 2.0])
    }
