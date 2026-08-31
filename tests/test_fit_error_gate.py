"""A batch whose features raise during fitting must fail loudly, not exit 0."""
import sys

import numpy as np
import pandas as pd
import pytest

from dose_response.cli import fit_batch


@pytest.fixture
def expression_batch(batch):
    """Two features over the miniature design, with a dose-responsive y."""
    rows = []
    for feature in ["ENSG_A", "ENSG_B"]:
        for _, r in batch.iterrows():
            y = 5.0 + 2.0 * np.log10(r.dose + 1.0)
            rows.append((feature, r["sample"], r.treatment, r.dose, y))
    return pd.DataFrame(rows, columns=["featureID", "sample", "treatment", "dose", "y"])


def _argv(monkeypatch, write_tsv, df, extra=None):
    path = write_tsv(df)
    out_pkl = str(path) + ".pkl"
    out_tsv = str(path) + ".out.tsv"
    monkeypatch.setattr(sys, "argv", [
        "fit-batch", "--input", str(path), "--output_pkl", out_pkl,
        "--output_tsv", out_tsv, "--model", "expression_logfc",
        "--AbsSpearmanPreFilter", "0.0", "--samples", "10"] + (extra or []))
    return out_tsv


def _raise_always(monkeypatch, exc):
    def boom(*a, **k):
        raise exc
    monkeypatch.setitem(fit_batch.MODEL_CONFIG[1], "fit_func", boom)


def test_exits_nonzero_when_a_feature_raises(monkeypatch, write_tsv, expression_batch):
    out_tsv = _argv(monkeypatch, write_tsv, expression_batch)
    _raise_always(monkeypatch, RuntimeError("Compilation failed (return status=1)"))
    with pytest.raises(SystemExit) as e:
        fit_batch.main()
    assert e.value.code == 1
    # the summary is still written, so the failure stays diagnosable
    got = pd.read_csv(out_tsv, sep="\t")
    assert got.status.astype(str).str.startswith("Model fit error").any()


def test_negative_threshold_tolerates_any_number(monkeypatch, write_tsv, expression_batch):
    _argv(monkeypatch, write_tsv, expression_batch, extra=["--MaxFitErrors", "-1"])
    _raise_always(monkeypatch, RuntimeError("boom"))
    fit_batch.main()  # must not raise


def test_threshold_allows_up_to_n(monkeypatch, write_tsv, expression_batch):
    _argv(monkeypatch, write_tsv, expression_batch, extra=["--MaxFitErrors", "2"])
    _raise_always(monkeypatch, RuntimeError("boom"))
    fit_batch.main()  # 2 features, threshold 2 -> tolerated


def test_clean_batch_exits_zero(monkeypatch, write_tsv, expression_batch):
    out_tsv = _argv(monkeypatch, write_tsv, expression_batch)
    fit_batch.main()
    got = pd.read_csv(out_tsv, sep="\t")
    assert not got.status.astype(str).str.startswith("Model fit error").any()
