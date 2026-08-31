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


def test_all_ones_column_dropped_not_fatal(batch, covariates, write_tsv):
    # A series in which every sample shares the condition (e.g. GSE304951_U1CKD, where all
    # samples are U1C-knockdown) is as uninformative as one where none do. A stray intercept
    # column looks identical and is equally harmless to drop.
    covariates["intercept"] = 1
    spec = prepare_covariates(write_tsv(covariates), None, batch)
    assert "intercept" in spec.dropped
    assert spec.columns == ["IFNa", "IFNg"]


def test_constant_zero_column_dropped_not_fatal(batch, covariates, write_tsv):
    covariates["U1CKD"] = 0        # exists globally but not in this series
    spec = prepare_covariates(write_tsv(covariates), None, batch)
    assert "U1CKD" in spec.dropped
    assert spec.columns == ["IFNa", "IFNg"]


def test_all_columns_dropped_falls_back_not_fatal(batch, write_tsv):
    """A global covariate file must be inert -- not fatal -- for series it does not apply to."""
    batch["sample"] = batch["sample"].str.replace("IFNa", "noStim").str.replace("IFNg", "noStim")
    batch["treatment"] = (batch["treatment"].str.replace("IFNa", "noStim")
                          .str.replace("IFNg", "noStim"))
    batch["sample"] = batch["sample"] + "_" + batch.index.astype(str)   # keep ids unique
    cov = pd.DataFrame({"sample": batch["sample"], "IFNa": 0, "IFNg": 0})
    spec = prepare_covariates(write_tsv(cov), None, batch)
    assert spec.n_covariates == 0
    assert set(spec.dropped) == {"IFNa", "IFNg"}
    # zero covariates must produce no design at all, so the model graph is unchanged
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
    covariates["U1CKD"] = 0
    spec = prepare_covariates(write_tsv(covariates), None, batch)
    d = spec.describe()
    assert "IFNa" in d
    assert "dropped:U1CKD" in d


def test_parse_covariate_priors():
    assert parse_covariate_priors([["IFNa", "Normal", "0", "2.0"]]) == {
        "IFNa": ("Normal", [0.0, 2.0])
    }
