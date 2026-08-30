"""Model registry and per-model configuration."""
from functools import partial

from .models.expression_absolute import fit_expression_absolute
from .models.expression_logfc import fit_expression_logfc
from .models.splicing import fit_splicing
from .summarize import r2_by_treatment_expression, r2_by_treatment_splicing

__all__ = ["MODEL_CONFIG", "MODEL_REGISTRY", "MODEL_NAMES", "resolve_model",
           "COVARIATE_INDEXED_SUMMARY_VARS", "TREATMENT_INDEXED_PARAMS",
           "COVARIATE_SUPPORTED_MODELS", "validate_covariate_args",
           "validate_treatment_specific_priors"]

_PSI_OUTCOME = lambda t_data: t_data["y"] / t_data["n"]
_LOG2_OUTCOME = lambda t_data: t_data["y"]

MODEL_CONFIG = {
    1: {
        "name": "expression_logfc",
        "fit_func": fit_expression_logfc,
        "spearman_func": _LOG2_OUTCOME,
        "summary_vars_scalar": ["baseline_log2", "span_log2", "plateau_log2", "sigma"],
        "summary_vars_treatment": ["rate", "logEC50", "logEC2x"],
        "r2_func": r2_by_treatment_expression,
    },
    2: {
        "name": "splicing_psi_vertical",
        "description": "logistic on PSI; covariate shifts floor and ceiling together",
        "fit_func": partial(fit_splicing, scale="psi",
                             covariate_target="vertical"),
        "spearman_func": _PSI_OUTCOME,
        "summary_vars_scalar": ["baseline_log2odds", "span_log2odds", "plateau_log2odds",
                                "baseline_PSI", "plateau_PSI", "span_PSI",
                                "hill", "rate", "span_sign_min", "phi"],
        "summary_vars_treatment": ["span_by_arm_log2odds", "logEC50", "logEC50_PSI",
                                   "logEC_dPSI05", "logEC2x_odds", "dPSI_at_maxdose",
                                   "frac_realized"],
        "r2_func": r2_by_treatment_splicing,
    },
    3: {
        "name": "expression_absolute",
        "fit_func": fit_expression_absolute,
        "spearman_func": _LOG2_OUTCOME,
        "summary_vars_scalar": ["baseline_log2", "span_log2", "plateau_log2", "sigma"],
        "summary_vars_treatment": ["rate", "logEC50", "logEC2x"],
        "r2_func": r2_by_treatment_expression,
    },
    4: {
        "name": "splicing_log2odds_sharedceiling",
        "description": "logistic on log2-odds; covariate shifts the floor only",
        "fit_func": partial(fit_splicing, scale="log2odds",
                             covariate_target="sharedceiling"),
        "spearman_func": _PSI_OUTCOME,
        "summary_vars_scalar": ["baseline_log2odds", "span_log2odds", "plateau_log2odds",
                                "baseline_PSI", "plateau_PSI", "span_PSI",
                                "hill", "rate", "span_sign_min", "phi"],
        "summary_vars_treatment": ["span_by_arm_log2odds", "logEC50", "logEC50_PSI",
                                   "logEC_dPSI05", "logEC2x_odds", "dPSI_at_maxdose",
                                   "frac_realized"],
        "r2_func": r2_by_treatment_splicing,
    },
    5: {
        "name": "splicing_psi_sharedceiling",
        "description": "logistic on PSI; covariate shifts the floor only",
        "fit_func": partial(fit_splicing, scale="psi",
                             covariate_target="sharedceiling"),
        "spearman_func": _PSI_OUTCOME,
        "summary_vars_scalar": ["baseline_log2odds", "span_log2odds", "plateau_log2odds",
                                "baseline_PSI", "plateau_PSI", "span_PSI",
                                "hill", "rate", "span_sign_min", "phi"],
        "summary_vars_treatment": ["span_by_arm_log2odds", "logEC50", "logEC50_PSI",
                                   "logEC_dPSI05", "logEC2x_odds", "dPSI_at_maxdose",
                                   "frac_realized"],
        "r2_func": r2_by_treatment_splicing,
    },
    6: {
        "name": "splicing_log2odds_vertical",
        "description": "logistic on log2-odds; covariate shifts floor and ceiling together",
        "fit_func": partial(fit_splicing, scale="log2odds",
                             covariate_target="vertical"),
        "spearman_func": _PSI_OUTCOME,
        "summary_vars_scalar": ["baseline_log2odds", "span_log2odds", "plateau_log2odds",
                                "baseline_PSI", "plateau_PSI", "span_PSI",
                                "hill", "rate", "span_sign_min", "phi"],
        "summary_vars_treatment": ["span_by_arm_log2odds", "logEC50", "logEC50_PSI",
                                   "logEC_dPSI05", "logEC2x_odds", "dPSI_at_maxdose",
                                   "frac_realized"],
        "r2_func": r2_by_treatment_splicing,
    },
}

MODEL_NAMES = {cfg["name"]: num for num, cfg in MODEL_CONFIG.items()}

# Short aliases kept so existing configs keep resolving.
MODEL_ALIASES = {
    "splicing_psi": 2,
    "splicing_log2odds": 4,
}
MODEL_NAMES.update(MODEL_ALIASES)
MODEL_REGISTRY = {**MODEL_CONFIG, **{name: MODEL_CONFIG[num] for name, num in MODEL_NAMES.items()}}


def resolve_model(spec):
    """Accept a model name or its integer alias; return the integer key."""
    if spec in MODEL_NAMES:
        return MODEL_NAMES[spec]
    try:
        num = int(spec)
    except (TypeError, ValueError):
        num = None
    if num in MODEL_CONFIG:
        return num
    raise ValueError(f"Unknown model {spec!r}. Choose from: "
                     + ", ".join(f"{n} ({k})" for k, n in sorted(MODEL_NAMES.items(),
                                                                 key=lambda kv: kv[1])))


_SPLICING = (2, 4, 5, 6)

COVARIATE_INDEXED_SUMMARY_VARS = {1: [], 3: ["beta_log2"],
                                  **{k: ["beta_log2odds"] for k in _SPLICING}}

TREATMENT_INDEXED_PARAMS = {1: {"logEC50", "rate"}, 3: {"logEC50", "rate"},
                            **{k: {"logEC50"} for k in _SPLICING}}

COVARIATE_SUPPORTED_MODELS = {1: False, 3: True, **{k: True for k in _SPLICING}}


def validate_covariate_args(args, model_num):
    if getattr(args, "covariates", None) and not COVARIATE_SUPPORTED_MODELS.get(model_num, False):
        raise ValueError(
            f"--covariates is not supported for model {model_num} "
            f"({MODEL_CONFIG[model_num]['name']}). It pins the untreated mean at exactly 0 and "
            "has no free baseline, so a covariate would have no reference level to shift from "
            "and the distortion would be pushed into span_log2. Use expression_absolute."
        )
    if not getattr(args, "covariates", None):
        for opt in ("covariate_cols", "covariate_prior"):
            if getattr(args, opt, None):
                raise ValueError(f"--{opt} was given without --covariates.")


def validate_treatment_specific_priors(args, model_num):
    allowed = TREATMENT_INDEXED_PARAMS[model_num]
    if hasattr(args, "prior") and args.prior:
        for prior in args.prior:
            param, treatment, *_ = prior
            if treatment != "ALL" and param not in allowed:
                raise ValueError(
                    f"'{param}' is not indexed by treatment in model {model_num} "
                    f"({MODEL_CONFIG[model_num]['name']}); treatment-specific priors are only "
                    f"available for {sorted(allowed)}."
                )
