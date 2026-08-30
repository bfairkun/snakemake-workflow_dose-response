"""Shim: the implementation lives in src/dose_response/covariates.py.

Kept so notebooks that predate the package can still `import covariate_design`.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(os.path.realpath(__file__)).parents[1] / "src"))

from dose_response.covariates import *  # noqa: F401,F403
from dose_response.covariates import (  # noqa: F401
    DEFAULT_COVARIATE_PRIOR_SD,
    CovariateDesignError,
    CovariateSpec,
    design_for_feature,
    load_covariate_table,
    parse_covariate_priors,
    prepare_covariates,
)
