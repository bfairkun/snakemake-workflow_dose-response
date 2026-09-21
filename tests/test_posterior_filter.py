"""The post-fit posterior filter, and its any-arm semantics.

Regression cover for a bug that silently discarded every drug-selective feature: the filter
used to flatten a treatment-dimensioned parameter over arms, turning "95% of posterior mass
beyond the threshold" into "what fraction of ARMS respond". With three arms and one responder
the achievable maximum was ~0.33, so no arm-selective effect could ever pass at 0.95.
"""
import numpy as np
import xarray as xr

from dose_response.filters import check_posterior_filters

DPSI = [("dPSI_at_maxdose", 0.95, -1, -0.1), ("dPSI_at_maxdose", 0.95, 0.1, 1)]


class _Idata:
    def __init__(self, ds):
        self.posterior = ds


def _arms(per_arm, name="dPSI_at_maxdose", ndraw=2000, seed=0):
    """One (mean, sd) per treatment arm -> an idata-like with dims (chain, draw, treatment)."""
    rng = np.random.default_rng(seed)
    a = np.stack([rng.normal(m, s, (1, ndraw)) for m, s in per_arm], axis=-1)
    return _Idata(xr.Dataset(
        {name: (("chain", "draw", "treatment"), a)},
        coords={"treatment": [f"arm{i}" for i in range(len(per_arm))]}))


def _scalar(value, sd, name="span_log2", ndraw=2000, seed=0):
    rng = np.random.default_rng(seed)
    return _Idata(xr.Dataset({name: (("chain", "draw"), rng.normal(value, sd, (1, ndraw)))}))


def test_single_responding_arm_passes():
    """The case the old pooled filter made impossible: one strong arm out of three."""
    passes, _ = check_posterior_filters(_arms([(0.9, 0.03), (0.01, 0.01), (0.0, 0.01)]), DPSI)
    assert passes


def test_all_arms_responding_passes():
    passes, _ = check_posterior_filters(_arms([(0.9, 0.03), (0.8, 0.03), (0.7, 0.03)]), DPSI)
    assert passes


def test_no_responding_arm_fails():
    passes, msg = check_posterior_filters(_arms([(0.01, 0.01), (0.0, 0.01), (0.0, 0.01)]), DPSI)
    assert not passes
    assert "dPSI_at_maxdose" in msg


def test_weak_effect_fails_even_though_it_is_the_best_arm():
    """Any-arm must not become "any arm, however small": 0.05 is below the 0.1 interval."""
    passes, _ = check_posterior_filters(_arms([(0.05, 0.01), (0.0, 0.01), (0.0, 0.01)]), DPSI)
    assert not passes


def test_negative_direction_passes():
    """Both intervals are honoured, so a strongly DOWN arm passes (e.g. acceptor SSE)."""
    passes, _ = check_posterior_filters(_arms([(-0.9, 0.03), (0.0, 0.01), (0.0, 0.01)]), DPSI)
    assert passes


def test_single_arm_series_unchanged():
    """Pooling over one arm was always a no-op; behaviour here must not have shifted."""
    assert check_posterior_filters(_arms([(0.9, 0.03)]), DPSI)[0]
    assert not check_posterior_filters(_arms([(0.01, 0.01)]), DPSI)[0]


def test_scalar_parameter_still_supported():
    """Params with no treatment dim (e.g. a span) take the whole-array path."""
    assert check_posterior_filters(_scalar(2.0, 0.1), [("span_log2", 0.95, 1, 100)])[0]
    assert not check_posterior_filters(_scalar(0.0, 0.1), [("span_log2", 0.95, 1, 100)])[0]


def test_failure_message_names_the_best_arm():
    """The pooled message reported an uninterpretable average; name the arm instead.

    Arms must differ in fraction for this to be meaningful: arm1 straddles the 0.1 boundary
    so it has partial mass in range, while arm0 has none. (With every arm at 0.00 the
    argmax tie-breaks arbitrarily, which is fine but tests nothing.)
    """
    idata = _arms([(0.0, 0.01), (0.10, 0.05)])
    _, msg = check_posterior_filters(idata, DPSI)
    assert "arm1" in msg and "at least one arm" in msg


def test_all_filters_must_pass():
    """Multiple params are AND-ed: a feature clearing one but not the other is rejected."""
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {"dPSI_at_maxdose": (("chain", "draw", "treatment"), rng.normal(0.9, 0.03, (1, 500, 1))),
         "span_log2":       (("chain", "draw"), rng.normal(0.0, 0.1, (1, 500)))},
        coords={"treatment": ["arm0"]})
    passes, msg = check_posterior_filters(_Idata(ds), DPSI + [("span_log2", 0.95, 1, 100)])
    assert not passes and "span_log2" in msg
