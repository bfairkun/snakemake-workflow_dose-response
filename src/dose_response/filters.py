"""Pre-fit and post-fit feature filters."""
import logging
from collections import defaultdict

import numpy as np

__all__ = ["check_prefilter_by_number", "check_posterior_filters"]


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
    Returns (True, "Pass") if all filters pass, else (False, reason)
    """
    filter_dict = defaultdict(list)
    for param, fraction, low, high in filters:
        filter_dict[param].append((float(fraction), float(low), float(high)))

    for param, intervals in filter_dict.items():
        arr = idata.posterior[param].values.flatten()
        # Union of all intervals
        mask = np.zeros_like(arr, dtype=bool)
        required_fraction = None
        for fraction, low, high in intervals:
            mask |= ((arr >= low) & (arr <= high))
            if required_fraction is None:
                required_fraction = fraction
            elif required_fraction != fraction:
                raise ValueError(f"Multiple different fractions specified for {param} in posterior filter.")
        frac_in_range = np.mean(mask)
        if frac_in_range < required_fraction:
            intervals_str = " or ".join([f"[{low}, {high}]" for _, low, high in intervals])
            return False, (
                f"Did not fit; {param}: only {frac_in_range:.2f} in {intervals_str} (required {required_fraction})"
            )
    return True, "Pass"
