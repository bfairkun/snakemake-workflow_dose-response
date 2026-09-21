"""Pre-fit and post-fit feature filters."""
import logging
from collections import defaultdict

import numpy as np

__all__ = ["check_prefilter_by_number", "check_posterior_filters", "observed_abs_effect"]


def check_prefilter_by_number(feature_data, prefilters):
    """
    feature_data: rows for ONE feature (already subset by the caller)
    prefilters: list of (var, min_count, low, high)
    Passes if for each var, at least one interval is satisfied for the required min_count.
    Note intervals are in the units of `var` itself, and multiple intervals on the same var
    are OR-ed together.
    """
    filter_dict = defaultdict(list)
    for var, min_count, low, high in prefilters:
        filter_dict[var].append((int(min_count), float(low), float(high)))

    for var, intervals in filter_dict.items():
        vals = feature_data[var]
        passed_any = False
        actuals = []
        for min_count, low, high in intervals:
            count_in_range = ((vals >= low) & (vals <= high)).sum()
            actuals.append(f"{count_in_range} in [{low}, {high}]")
            if count_in_range >= min_count:
                passed_any = True
        if not passed_any:
            intervals_str = " or ".join([f"[{low}, {high}] (at least {min_count})" for min_count, low, high in intervals])
            actuals_str = "; ".join(actuals)
            return False, f"Did not fit; {var}: {actuals_str}; required {intervals_str}"
    return True, "Pass"

def check_posterior_filters(idata, filters):
    """
    filters: list of (param, fraction, low, high)
    For each param, combine all intervals, and require that the fraction of posterior samples
    in the union of all intervals is at least the specified threshold.

    ANY-ARM semantics for treatment-dimensioned parameters. Params such as dPSI_at_maxdose /
    dY_at_maxdose carry a `treatment` dim, so the test is applied per arm and the feature
    passes if ANY arm clears the threshold. A feature that responds to one drug and not the
    others is exactly the interesting case, and it must not be penalised for the arms that do
    nothing.

    This previously flattened the array over arms, which silently turned the test into
    "what fraction of ARMS respond": with three arms and one responder the achievable maximum
    was ~0.33, so no arm-selective effect could ever reach 0.95. It discarded e.g. the PRNP
    cryptic donor (pooled 0.45) and the ATG5 skipping junction (0.91) in Exp2 despite both
    being clean, strong, branaplam-selective responses. Single-arm series were unaffected,
    since pooling over one arm is a no-op.

    Returns (True, "Pass") if all filters pass, else (False, reason)
    """
    filter_dict = defaultdict(list)
    for param, fraction, low, high in filters:
        filter_dict[param].append((float(fraction), float(low), float(high)))

    for param, intervals in filter_dict.items():
        da = idata.posterior[param]
        required_fraction = None
        for fraction, _, _ in intervals:
            if required_fraction is None:
                required_fraction = fraction
            elif required_fraction != fraction:
                raise ValueError(f"Multiple different fractions specified for {param} in posterior filter.")

        # Per-arm draw matrices; a scalar param stays a single "arm" so behaviour is unchanged.
        arm_dims = [d for d in da.dims if d not in ("chain", "draw")]
        if arm_dims:
            arm_names = [str(v) for v in da.coords[arm_dims[0]].values]
            arrays = [da.isel({arm_dims[0]: i}).values.ravel() for i in range(len(arm_names))]
        else:
            arm_names, arrays = ["ALL"], [da.values.ravel()]

        fracs = []
        for arr in arrays:
            mask = np.zeros_like(arr, dtype=bool)
            for _, low, high in intervals:
                mask |= ((arr >= low) & (arr <= high))
            fracs.append(float(np.mean(mask)))

        if max(fracs) < required_fraction:
            intervals_str = " or ".join([f"[{low}, {high}]" for _, low, high in intervals])
            best = arm_names[int(np.argmax(fracs))]
            return False, (
                f"Did not fit; {param}: best arm {best} only {max(fracs):.2f} in {intervals_str} "
                f"(required {required_fraction} in at least one arm)"
            )
    return True, "Pass"

def observed_abs_effect(feature_data, treated, outcome_func):
    """Largest observed |top-dose mean - control mean| over treated arms x dose-0 groups.

    Maximising over control groups rather than pooling them keeps this a lower bound on what
    a matched-baseline model would see: a pooled mean lies between the group means, so no
    matched comparison can exceed every group-wise one. NaN when nothing is measurable, which
    callers must treat as "keep".
    """
    def _mean(g):
        v = np.asarray(outcome_func(g), dtype=float)
        v = v[np.isfinite(v)]
        return v.mean() if v.size else np.nan

    controls = feature_data[feature_data["dose"] == 0]
    baselines = [_mean(g) for _, g in controls.groupby("treatment")]
    baselines = [b for b in baselines if np.isfinite(b)]
    if not baselines:
        return np.nan

    effect = np.nan
    for _, arm in treated.groupby("treatment"):
        top = _mean(arm[arm["dose"] == arm["dose"].max()])
        if not np.isfinite(top):
            continue
        for b in baselines:
            e = abs(top - b)
            effect = e if not np.isfinite(effect) else max(effect, e)
    return effect
