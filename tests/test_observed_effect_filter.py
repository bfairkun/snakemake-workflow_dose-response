"""The pre-fit screen on observed effect size."""
import numpy as np
import pandas as pd
import pytest

from dose_response.filters import observed_abs_effect

PSI = lambda g: g["y"] / g["n"]


def _batch(rows):
    return pd.DataFrame(rows, columns=["treatment", "dose", "y", "n"])


def test_flat_feature_has_no_effect():
    d = _batch([("DMSO", 0, 50, 100), ("DMSO", 0, 50, 100),
                ("drug", 10, 50, 100), ("drug", 100, 50, 100)])
    assert observed_abs_effect(d, d[d.dose > 0], PSI) == pytest.approx(0.0)


def test_responsive_feature_measures_top_dose_change():
    d = _batch([("DMSO", 0, 10, 100), ("drug", 10, 40, 100), ("drug", 100, 90, 100)])
    # top dose is 100 -> PSI 0.9, baseline 0.1
    assert observed_abs_effect(d, d[d.dose > 0], PSI) == pytest.approx(0.8)


def test_max_is_taken_over_arms():
    d = _batch([("DMSO", 0, 10, 100),
                ("flat", 10, 10, 100), ("flat", 100, 12, 100),
                ("strong", 10, 10, 100), ("strong", 100, 70, 100)])
    assert observed_abs_effect(d, d[d.dose > 0], PSI) == pytest.approx(0.6)


def test_max_over_control_groups_never_below_pooled():
    # C2C5 shape: three stimulus-specific control groups with different baselines.
    d = _batch([("DMSO_noStim", 0, 10, 100), ("DMSO_IFNa", 0, 50, 100),
                ("DMSO_IFNg", 0, 90, 100), ("C2C5_IFNa", 100, 55, 100)])
    pooled = abs(0.55 - np.mean([0.10, 0.50, 0.90]))
    got = observed_abs_effect(d, d[d.dose > 0], PSI)
    assert got >= pooled                      # never screens out more than pooling would
    assert got == pytest.approx(0.45)         # driven by the noStim group


def test_unmeasurable_returns_nan_so_caller_keeps_feature():
    d = _batch([("DMSO", 0, 0, 0), ("drug", 100, 0, 0)])
    assert np.isnan(observed_abs_effect(d, d[d.dose > 0], PSI))


def test_no_controls_returns_nan():
    d = _batch([("drug", 10, 10, 100), ("drug", 100, 90, 100)])
    assert np.isnan(observed_abs_effect(d, d[d.dose > 0], PSI))


# --- per-model default for --MinObservedAbsEffect -----------------------------------------
# The threshold is in the model's OUTCOME units, so the default lives per model rather than
# as one CLI-wide scalar: 0.10 (PSI) for the splicing models, off for the expression models
# whose outcome is log2.

def test_splicing_models_default_to_calibrated_threshold():
    from dose_response.fitting import MODEL_CONFIG
    for m, cfg in MODEL_CONFIG.items():
        if isinstance(m, int) and cfg["name"].startswith("splicing"):
            assert cfg["default_min_observed_abs_effect"] == 0.10, cfg["name"]


def test_expression_models_have_no_default():
    """0.10 is a PSI number; applying it to a log2 outcome would be a unit error."""
    from dose_response.fitting import MODEL_CONFIG
    for m, cfg in MODEL_CONFIG.items():
        if isinstance(m, int) and cfg["name"].startswith("expression"):
            assert cfg["default_min_observed_abs_effect"] is None, cfg["name"]
