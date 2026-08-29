"""Tests for dose_response.plotting."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from dose_response import plotting as p


def test_lazy_matplotlib_isolation():
    """`import dose_response` must not drag matplotlib into a fitting env."""
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; import dose_response; print('matplotlib' in sys.modules)"],
        capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert r.stdout.strip() == "False", r.stdout + r.stderr


@pytest.mark.parametrize("n", [10, 100, 900, 5000])
def test_empirical_log2odds_saturates_at_read_depth_ceiling(n):
    """At PSI=1 the Haldane-corrected log2-odds tops out at log2(2n+1). This is why a span
    larger than that is unobservable however wide its prior."""
    assert float(p.empirical_log2odds(n, n)) == pytest.approx(np.log2(2 * n + 1))


def test_empirical_log2odds_is_antisymmetric():
    assert float(p.empirical_log2odds(3, 900)) == pytest.approx(-float(p.empirical_log2odds(897, 900)))


def test_marker_area_tracks_sqrt_n():
    assert p.marker_size(400) / p.marker_size(100) == pytest.approx(2.0)


def test_curve_band_evaluates_per_draw():
    """Per-draw evaluation, not at the posterior mean: with correlated inputs the two differ."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 4000)
    b = -a + rng.normal(0, 0.1, 4000)          # strongly anti-correlated
    xg = np.array([0.0, 1.0])
    mean, lo, hi = p.curve_band(lambda x, a, b: a + b + 0 * x, {"a": a, "b": b}, xg)
    assert mean == pytest.approx(np.full(2, (a + b).mean()), abs=1e-12)
    width_correct = hi - lo
    width_naive = np.percentile(b, 97.5) - np.percentile(b, 2.5)
    assert np.all(width_correct < width_naive), "anti-correlation must shrink the band"


def test_forest_flags_rows():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    p.forest(ax, ["a", "b", "c"], [1, 2, 3], [0, 1, 2], [2, 3, 4],
             flag=[False, True, False], flag_label="outside dose range")
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]
    plt.close(fig)


def test_panel_grid_marks_empty_panels():
    fig, axes = p.panel_grid(["r1", "r2"], ["c1", "c2"],
                             lambda ax, r, c: r == "r1", figsize=(4, 3))
    assert axes.shape == (2, 2)
    assert any("no data" in t.get_text() for t in axes[1, 0].texts)
    import matplotlib.pyplot as plt; plt.close(fig)
