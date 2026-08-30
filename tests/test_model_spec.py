"""The model specification, as executable assertions.

Every test here is marked xfail(strict=True) while Phase A is the current state. The moment
Phase B lands, pytest reports XPASS -- which is a failure under strict -- and the marker
must be removed. That is the point: the spec cannot be quietly dropped, and it cannot be
quietly half-implemented either.

Companion to docs/models.qmd.
"""
import pathlib

import numpy as np
import pytest

LN2, LOG2_10, LN10 = np.log(2.0), np.log2(10.0), np.log(10.0)


def _rv_spec(model, name):
    """(distribution class name, [parameter values]) for a free RV.

    PyMC RV nodes are (rng, size, *dist_params), so parameters start at index 2. Note the
    stored parameterization is not always the constructor's: Gamma keeps scale, not rate.
    """
    for v in model.free_RVs:
        if v.name == name:
            cls = type(v.owner.op).__name__
            params = [float(np.asarray(i.eval()).ravel()[0]) for i in v.owner.inputs[2:]]
            return cls, params
    raise AssertionError(f"{name!r} is not a free RV; free_RVs = "
                         f"{sorted(v.name for v in model.free_RVs)}")


def _fit(which, batch, **kw):
    from dose_response.fitting import MODEL_REGISTRY
    return MODEL_REGISTRY[which]["fit_func"](batch, samples=50, **kw)


# --- naming -------------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_named_model_registry(model_key):
    """--model takes names; integers stay as aliases."""
    from dose_response.fitting import MODEL_REGISTRY
    assert model_key in MODEL_REGISTRY
    assert MODEL_REGISTRY[2] is MODEL_REGISTRY["splicing_psi"]
    assert MODEL_REGISTRY[4] is MODEL_REGISTRY["splicing_log2odds"]


def test_maxdeltapsi_is_retired(splicing_batch):
    """Retired rather than sign-flipped, so old and new files stay distinguishable."""
    idata, _ = _fit("splicing_psi", splicing_batch)
    assert "MaxDeltaPSI" not in idata.posterior
    assert "span_PSI" in idata.posterior


@pytest.mark.parametrize("model_key,expected", [
    ("splicing_psi", {"baseline_PSI", "plateau_PSI", "span_PSI", "span_log2odds",
                      "baseline_log2odds", "plateau_log2odds", "rate", "hill", "logEC50",
                      "logEC50_PSI", "logEC_dPSI05", "logEC2x_odds", "phi",
                      "dPSI_at_maxdose", "frac_realized", "psi_treated"}),
    ("splicing_log2odds", {"baseline_log2odds", "span_log2odds", "plateau_log2odds",
                           "baseline_PSI", "plateau_PSI", "span_PSI", "rate", "hill",
                           "logEC50", "logEC50_PSI", "span_by_arm_log2odds",
                           "span_sign_min", "dPSI_at_maxdose", "frac_realized",
                           "psi_treated", "phi"}),
])
def test_expected_variable_names(splicing_batch, model_key, expected):
    idata, _ = _fit(model_key, splicing_batch)
    missing = expected - set(idata.posterior.data_vars)
    assert not missing, f"missing: {sorted(missing)}"


@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_dropped_names_are_gone(splicing_batch, model_key):
    idata, _ = _fit(model_key, splicing_batch)
    for gone in ["a2", "U2", "Delta2", "H", "k", "beta2", "amp2", "min_amp_signed",
                 "EC_dPSI50Max", "delta_half", "Emax", "lower", "upper", "slope",
                 "delta_logit", "ED2x", "ED2x_odds", "ED_5dPSI", "psi_asymptote", "psi_floor"]:
        assert gone not in idata.posterior, f"{gone} should have been renamed away"


# --- priors, identical in both models ------------------------------------------------------
@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_span_prior_is_studentt_3_0_6(splicing_batch, model_key):
    _, model = _fit(model_key, splicing_batch)
    cls, params = _rv_spec(model, "span_log2odds")
    assert "StudentT" in cls, cls
    nu, mu, sigma = params[0], params[1], params[2]
    assert (nu, mu, sigma) == pytest.approx((3.0, 0.0, 6.0))


@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_hill_is_sampled_lognormal(splicing_batch, model_key):
    """hill is a free RV; rate is derived from it, not sampled."""
    _, model = _fit(model_key, splicing_batch)
    cls, params = _rv_spec(model, "hill")
    assert "LogNormal" in cls, cls
    assert params[0] == pytest.approx(np.log(1.5))
    assert params[1] == pytest.approx(0.35)
    assert "rate" not in {v.name for v in model.free_RVs}


@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_logEC50_prior_is_wide(splicing_batch, model_key):
    """sigma = 3: the PSI-halfway dose routinely lies above the top assayed dose."""
    _, model = _fit(model_key, splicing_batch)
    names = [v.name for v in model.free_RVs if v.name.startswith("logEC50_")]
    assert names, [v.name for v in model.free_RVs]
    for n in names:
        cls, params = _rv_spec(model, n)
        assert "Normal" in cls and "LogNormal" not in cls, cls
        assert params[1] == pytest.approx(3.0)


@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_phi_prior_shared(splicing_batch, model_key):
    _, model = _fit(model_key, splicing_batch)
    cls, params = _rv_spec(model, "phi")
    assert "Gamma" in cls
    alpha, scale = params            # PyMC stores scale, so beta = 1/scale
    assert (alpha, 1 / scale) == pytest.approx((2.0, 0.2))


# --- semantics ----------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_span_sign_convention(splicing_batch, model_key):
    """span = plateau - baseline. The synthetic junction rises, so span must be positive."""
    idata, _ = _fit(model_key, splicing_batch)
    p = idata.posterior
    assert float(p["span_PSI"].mean()) > 0
    assert float(p["span_log2odds"].mean()) > 0
    assert float(p["plateau_PSI"].mean()) > float(np.asarray(p["baseline_PSI"]).mean())


@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_hill_matches_max_slope(splicing_batch, model_key):
    """The hill <-> rate conversion must hold, with the model-appropriate formula."""
    idata, _ = _fit(model_key, splicing_batch)
    p = idata.posterior
    hill = np.asarray(p["hill"]).ravel()
    rate = np.asarray(p["rate"]).ravel()
    if model_key == "splicing_log2odds":
        span = np.abs(np.asarray(p["span_log2odds"]).ravel())
        implied = rate * span / (4 * LOG2_10)
    else:
        # Model 2 derives rate from the REFERENCE asymptotes, so take arm 0; the per-arm
        # values coincide with it whenever there is no covariate.
        sig = lambda z: 1 / (1 + np.exp(-z))
        base = sig(np.asarray(p["baseline_log2odds"]).ravel() * LN2)
        plat = sig(np.asarray(p["plateau_log2odds"]).ravel() * LN2)
        m = (base + plat) / 2
        implied = rate * (plat - base) / (4 * m * (1 - m) * LN10)
    assert implied == pytest.approx(hill, rel=0.02)


@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_dPSI_at_maxdose_and_frac_realized(splicing_batch, model_key):
    """Reconstruct both from the primitives rather than from each other.

    Dividing dPSI_at_maxdose by span_PSI would be a tautology and would not catch the wrong
    dose or the wrong baseline, so the curve is rebuilt here from baseline, span, rate and
    logEC50 and evaluated at the arm's own highest assayed dose.
    """
    idata, _ = _fit(model_key, splicing_batch)
    p = idata.posterior
    sig = lambda z: 1 / (1 + np.exp(-z))
    base = np.asarray(p["baseline_log2odds"]).ravel()
    span = np.asarray(p["span_log2odds"]).ravel()
    rate = np.asarray(p["rate"]).ravel()
    b, pl = sig(base * LN2), sig((base + span) * LN2)
    x_top = np.log10(splicing_batch.loc[splicing_batch.dose > 0, "dose"].max())
    for arm in map(str, p.coords["treatment"].values):
        e50 = np.asarray(p["logEC50"].sel(treatment=arm)).ravel()
        if model_key == "splicing_psi":
            psi_top = b + (pl - b) * sig(rate * (x_top - e50))
        else:
            psi_top = sig((base + span * sig(rate * (x_top - e50))) * LN2)
        want_d = psi_top - b
        got_d = np.asarray(p["dPSI_at_maxdose"].sel(treatment=arm)).ravel()
        assert got_d == pytest.approx(want_d, abs=1e-6), "wrong dose or wrong baseline"
        got_f = np.asarray(p["frac_realized"].sel(treatment=arm)).ravel()
        assert got_f == pytest.approx(want_d / (pl - b), rel=1e-6)


@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_asymptotes_are_reference_level_and_consistent(splicing_batch, model_key):
    """sigmoid(x_log2odds * ln2) == x_PSI must hold, so the two scales name one quantity."""
    idata, _ = _fit(model_key, splicing_batch)
    p = idata.posterior
    sig = lambda z: 1 / (1 + np.exp(-z))
    for odds, psi in [("baseline_log2odds", "baseline_PSI"),
                      ("plateau_log2odds", "plateau_PSI")]:
        a = sig(np.asarray(p[odds]).ravel() * LN2)
        b = np.asarray(p[psi]).ravel()
        assert a.shape == b.shape, f"{psi} must be a scalar like {odds}"
        assert a == pytest.approx(b, rel=1e-9)
    assert np.asarray(p["span_PSI"]).ravel() == pytest.approx(
        np.asarray(p["plateau_PSI"]).ravel() - np.asarray(p["baseline_PSI"]).ravel(), rel=1e-9)


@pytest.mark.parametrize("model_key", ["splicing_psi", "splicing_log2odds"])
def test_logEC50_is_native_and_logEC50_PSI_is_the_common_currency(splicing_batch, model_key):
    """logEC50 centres the sigmoid on this model's own scale. logEC50_PSI is always the
    PSI-halfway dose, so it is the column that can be compared across models."""
    idata, _ = _fit(model_key, splicing_batch)
    p = idata.posterior
    sig = lambda z: 1 / (1 + np.exp(-z))
    l2o = lambda q: np.log(q / (1 - q)) / LN2
    base = float(p["baseline_log2odds"].mean()); span = float(p["span_log2odds"].mean())
    rate = float(p["rate"].mean())
    b, pl = sig(base * LN2), sig((base + span) * LN2)
    for arm in map(str, p.coords["treatment"].values):
        e50 = float(p["logEC50"].sel(treatment=arm).mean())
        epsi = float(p["logEC50_PSI"].sel(treatment=arm).mean())
        if model_key == "splicing_psi":
            psi_of = lambda x: b + (pl - b) * sig(rate * (x - e50))
            assert epsi == pytest.approx(e50, abs=1e-9), "PSI space: the two must coincide"
            assert psi_of(e50) == pytest.approx((b + pl) / 2, abs=1e-9)
        else:
            psi_of = lambda x: sig((base + span * sig(rate * (x - e50))) * LN2)
            assert l2o(psi_of(e50)) == pytest.approx(base + span / 2, abs=1e-9), \
                "log2-odds space: logEC50 must be the log2-odds midpoint"
            assert epsi != pytest.approx(e50), "the two locations are different doses"
        assert psi_of(epsi) == pytest.approx((b + pl) / 2, abs=5e-3), \
            "logEC50_PSI must hit the PSI midpoint in every model"


def test_span_and_beta_are_signed(splicing_batch):
    """HalfNormal would forbid repressed junctions; the prior must be two-sided."""
    _, model = _fit("splicing_log2odds", splicing_batch)
    cls, _ = _rv_spec(model, "span_log2odds")
    assert "HalfNormal" not in cls


def test_dead_model_config_keys_removed():
    from dose_response.fitting import MODEL_CONFIG
    for cfg in MODEL_CONFIG.values():
        for dead in ("ppc_var", "obs_var", "treatment_idx_var"):
            assert dead not in cfg, f"{dead} is never read and should be gone"
