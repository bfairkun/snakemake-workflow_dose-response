"""Figures for dose-response fits. Import explicitly; needs matplotlib."""
import numpy as np
from matplotlib.lines import Line2D

__all__ = ["empirical_log2odds", "size_legend", "shade_assayed_range", "psi_gridlines",
           "curve_band", "panel_grid", "forest"]

LN2, LN10, LOG2_10 = np.log(2.0), np.log(10.0), np.log2(10.0)
_PSI_GRID = (0.01, 0.1, 0.5, 0.9, 0.99)


def empirical_log2odds(y, n, base=2):
    """Haldane-corrected empirical log-odds of y/n, in doublings by default."""
    v = np.log((np.asarray(y) + 0.5) / (np.asarray(n) - np.asarray(y) + 0.5))
    return v / LN2 if base == 2 else v


def marker_size(n, k=6.0):
    """Marker area proportional to sqrt(n), so area tracks precision."""
    return k * np.sqrt(np.asarray(n))


def size_legend(counts=(10, 100, 1000, 5000), k=6.0, color="0.3"):
    return [Line2D([], [], ls="", marker="o", color=color,
                   ms=np.sqrt(marker_size(c, k)), label=f"n={c}") for c in counts]


def shade_assayed_range(ax, xlo, xhi, color="0.94"):
    ax.axvspan(xlo, xhi, color=color, zorder=0)


def psi_gridlines(ax, base=2, color="0.89"):
    for p in _PSI_GRID:
        v = np.log(p / (1 - p))
        ax.axhline(v / LN2 if base == 2 else v, color=color, lw=0.6, zorder=1)


def control_offset(x_min, pad=0.7):
    """Where to park dose-0 points on a log-dose axis."""
    return float(np.floor(x_min) - pad)


def curve_band(f, draws, xgrid, q=(2.5, 97.5)):
    """Posterior mean and interval of f(xgrid, **draw) evaluated per draw.

    `f` takes (xgrid[:, None], **arrays broadcast over draws) and returns an
    (len(xgrid), n_draws) array. Evaluating per draw rather than at the posterior mean
    matters whenever the transform is non-linear or its inputs are correlated.
    """
    M = f(xgrid[:, None], **draws)
    return M.mean(axis=1), np.percentile(M, q[0], axis=1), np.percentile(M, q[1], axis=1)


def panel_grid(rows, cols, draw_panel, figsize=None, sharey=True, ylabel=None, xlabel=None,
               handles=None, suptitle=None, legend_ncol=6):
    """One panel per (row, col); `draw_panel(ax, row, col)` returns True if it drew anything."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=figsize or (5 * len(cols), 3.4 * len(rows)), sharey=sharey,
                             squeeze=False)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            if not draw_panel(ax, r, c):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_xticks([])
                continue
            if j == 0 and ylabel: ax.set_ylabel(ylabel)
            if i == len(rows) - 1 and xlabel: ax.set_xlabel(xlabel)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=legend_ncol,
                   frameon=False, fontsize=8.5)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0.05 if handles else 0, 1, 0.95 if suptitle else 1])
    return fig, axes


def forest(ax, labels, mean, lo, hi, ref=0.0, flag=None, colors=None, flag_label=None):
    """Horizontal interval plot, most recent at top. `flag` red-edges rows needing caution."""
    y = np.arange(len(labels))[::-1]
    mean, lo, hi = map(np.asarray, (mean, lo, hi))
    for k in range(len(labels)):
        c = "steelblue" if colors is None else colors[k]
        ax.plot([lo[k], hi[k]], [y[k], y[k]], color=c, lw=1.8, solid_capstyle="butt")
        ax.plot([mean[k]], [y[k]], marker="o", ms=6, color=c,
                mec="crimson" if flag is not None and flag[k] else "none", mew=1.4, zorder=3)
    if ref is not None:
        ax.axvline(ref, color="k", lw=0.9, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    if flag is not None and flag_label and np.any(flag):
        ax.plot([], [], ls="", marker="o", ms=6, mfc="none", mec="crimson", mew=1.4,
                label=flag_label)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
