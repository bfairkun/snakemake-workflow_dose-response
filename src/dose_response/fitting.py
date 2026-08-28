"""Model registry and per-model configuration.

Moved verbatim from scripts/BayesianDoseResponse_ByBatch.py; behaviour is unchanged.
"""
from .models.expression_absolute import fit_expression_absolute_model
from .models.expression_logfc import fit_gene_expression_model
from .models.splicing_psi import fit_splicing_model
from .summarize import r2_by_treatment_expression, r2_by_treatment_splicing

__all__ = ["validate_treatment_specific_priors", "MODEL_CONFIG", "COVARIATE_INDEXED_SUMMARY_VARS", "TREATMENT_INDEXED_PARAMS",
           "COVARIATE_SUPPORTED_MODELS", "validate_covariate_args"]


MODEL_CONFIG = {
    1: {  # Expression model
        "fit_func": fit_gene_expression_model,
        "spearman_func": lambda t_data: t_data["y"],
        "summary_vars_scalar": ["upper", "sigma"],
        "summary_vars_treatment": ["slope", "logEC50", "ED2x"],
        "ppc_var": "y_treated_mu",
        "obs_var": "y_treated_data",
        "treatment_idx_var": "treatment_idx",
        "r2_func": r2_by_treatment_expression,
    },
    2: {  # Splicing model
        "fit_func": fit_splicing_model,
        "spearman_func": lambda t_data: t_data["y"] / t_data["n"],
        "summary_vars_scalar": ["upper", "lower", "slope", "phi", "MaxDeltaPSI"],
        "summary_vars_treatment": ["logEC50", "ED_5dPSI", "ED2x_odds"],
        "ppc_var": "y_treated_mu",
        "obs_var": "y_treated_data",
        "treatment_idx_var": "treatment_idx",
        "r2_func": r2_by_treatment_splicing,
    },
    3: {  # Expression model on absolute log2 abundance (free intercept)
        "fit_func": fit_expression_absolute_model,
        "spearman_func": lambda t_data: t_data["y"],
        "summary_vars_scalar": ["lower", "Delta", "upper", "sigma"],
        "summary_vars_treatment": ["slope", "logEC50", "ED2x"],
        "ppc_var": "y_treated_mu",
        "obs_var": "y_treated_data",
        "treatment_idx_var": "treatment_idx",
        "r2_func": r2_by_treatment_expression,
    }
}

# Covariate coefficients are reported per covariate when covariates are in use. Vars listed
# here are skipped silently when absent from the posterior (i.e. when no --covariates given).
COVARIATE_INDEXED_SUMMARY_VARS = {1: [], 2: ["beta"], 3: ["beta"]}

# Define which parameters are allowed to have treatment-specific priors for each model
TREATMENT_INDEXED_PARAMS = {
    1: {"logEC50", "slope"},  # Example: model 1 allows both logEC50 and slope to be treatment-specific
    2: {"logEC50"},           # Example: model 2 only allows logEC50 to be treatment-specific
    3: {"logEC50", "slope"},  # same as model 1: gene-level slope legitimately varies by drug
}

# Which models accept a sample x covariate matrix. Model 1 is excluded on purpose: it pins the
# untreated mean at 0 and has no free intercept, so a covariate's reference level would have
# nothing to absorb it and the distortion would be pushed into `upper`. Use model 3 instead.
COVARIATE_SUPPORTED_MODELS = {1: False, 2: True, 3: True}

def validate_covariate_args(args, model_num):
    """Reject covariate options the chosen model cannot support, with the reason."""
    if getattr(args, "covariates", None) and not COVARIATE_SUPPORTED_MODELS.get(model_num, False):
        raise ValueError(
            f"--covariates is not supported for model {model_num}. Model 1 pins the untreated "
            "mean at exactly 0 and has no free intercept, so a covariate would have no "
            "reference level to shift from and the distortion would be pushed into `upper`. "
            "Use --model 3, which models absolute log2 abundance with a free intercept."
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
                    f"Parameter '{param}' is not indexed by treatment in model {model_num}, "
                    f"so you cannot specify a treatment-specific prior for it (got --prior {param} {treatment} ...)."
                )
