"""Model registry and per-model configuration."""
from .models.expression_absolute import fit_expression_absolute
from .models.expression_logfc import fit_expression_logfc
from .models.splicing_log2odds import fit_splicing_log2odds
from .models.splicing_psi import fit_splicing_psi
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
        "name": "splicing_psi",
        "fit_func": fit_splicing_psi,
        "spearman_func": _PSI_OUTCOME,
        "summary_vars_scalar": ["baseline_log2odds", "span_log2odds", "plateau_log2odds",
                                "hill", "rate", "phi"],
        "summary_vars_treatment": ["baseline_PSI", "plateau_PSI", "span_PSI", "logEC50",
                                   "logEC50_log2odds", "logEC_dPSI05", "logEC2x_odds",
                                   "dPSI_at_maxdose", "frac_realized"],
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
        "name": "splicing_log2odds",
        "fit_func": fit_splicing_log2odds,
        "spearman_func": _PSI_OUTCOME,
        "summary_vars_scalar": ["baseline_log2odds", "span_log2odds", "plateau_log2odds",
                                "hill", "rate", "plateau_PSI", "span_sign_min", "phi"],
        "summary_vars_treatment": ["baseline_PSI", "span_PSI", "span_by_arm_log2odds",
                                   "logEC50", "logEC50_log2odds", "dPSI_at_maxdose",
                                   "frac_realized"],
        "r2_func": r2_by_treatment_splicing,
    },
}

MODEL_NAMES = {cfg["name"]: num for num, cfg in MODEL_CONFIG.items()}
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


COVARIATE_INDEXED_SUMMARY_VARS = {1: [], 2: ["beta_log2odds"], 3: ["beta_log2"],
                                  4: ["beta_log2odds"]}

TREATMENT_INDEXED_PARAMS = {
    1: {"logEC50", "rate"},
    2: {"logEC50"},
    3: {"logEC50", "rate"},
    4: {"logEC50"},
}

COVARIATE_SUPPORTED_MODELS = {1: False, 2: True, 3: True, 4: True}


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
