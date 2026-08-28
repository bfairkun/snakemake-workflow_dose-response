"""Shared fixtures for the dose_response test suite."""
import pandas as pd
import pytest


@pytest.fixture
def write_tsv(tmp_path):
    """Write a DataFrame to a temp TSV and return the path (auto-cleaned by tmp_path)."""
    counter = {"n": 0}

    def _write(df):
        counter["n"] += 1
        p = tmp_path / f"cov_{counter['n']}.tsv"
        df.to_csv(p, sep="\t", index=False)
        return str(p)

    return _write


@pytest.fixture
def batch():
    """A miniature of the C2C5 design: 3 stimuli, matched controls, 2 doses each."""
    rows = []
    for stim in ["noStim", "IFNa", "IFNg"]:
        for rep in [1, 2]:
            rows.append((f"DMSO_{stim}_rep{rep}", f"DMSO_{stim}", 0.0))
        for d in [10.0, 100.0]:
            rows.append((f"C2C5_{stim}_{d:g}nM", f"C2C5_{stim}", d))
    return pd.DataFrame(rows, columns=["sample", "treatment", "dose"])


@pytest.fixture
def covariates(batch):
    """IFNa / IFNg indicators derived from the sample names, as in the real pipeline."""
    return pd.DataFrame({
        "sample": batch["sample"],
        "IFNa": batch["sample"].str.contains("IFNa").astype(int),
        "IFNg": batch["sample"].str.contains("IFNg").astype(int),
    })
