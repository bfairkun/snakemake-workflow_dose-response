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


@pytest.fixture
def splicing_batch():
    """Synthetic junction counts: 3 arms, matched controls, 6 doses, switch-like response.

    Self-contained so the suite never depends on project data.
    """
    import numpy as np
    rng = np.random.default_rng(7)
    rows = []
    for rep in range(4):
        rows.append(("DMSO_rep%d" % rep, "DMSO", 0.0, 3, 900))
    for arm, ec50 in [("drugA", 1.0), ("drugB", 2.0), ("drugC", 2.5)]:
        for d in [3.0, 10.0, 32.0, 100.0, 320.0, 1000.0]:
            eta = -9.0 + 13.0 / (1 + np.exp(-2.0 * (np.log10(d) - ec50)))   # log2-odds
            psi = 1 / (1 + np.exp(-eta * np.log(2)))
            n = 800
            rows.append((f"{arm}_{d:g}nM", arm, d, int(rng.binomial(n, psi)), n))
    return pd.DataFrame(rows, columns=["sample", "treatment", "dose", "y", "n"]).assign(
        featureID="chr1:100:200:clu_1_+")
