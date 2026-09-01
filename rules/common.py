import pandas as pd
from snakemake.utils import validate

# ---------------------------------------------------------------------------
# Load config values at workflow parse time
# ---------------------------------------------------------------------------

APPROACHES = config.get("approaches", {})
N_BATCHES  = int(config.get("n_batches", 200))
APPROACH_SPECIFICITY_PARAMS = {
    approach: config["approaches"][approach].get("specificity_test_params", ["logEC50"])
    for approach in APPROACHES
}

# Infer the list of Series (modeling groups) from the samples.tsv at parse time.
# This is safe because samples.tsv is a static input file, not a workflow output.
_samples = pd.read_csv(config["samples"], sep="\t")
SERIES   = _samples["Series"].dropna().unique().tolist() if "Series" in _samples.columns else []
validate(_samples, "../schemas/samples.schema.yaml")

# An approach may cover only some Series -- the SpliSER quantifications, for instance, were
# run on a subset of the samples, so those approaches have nothing to fit for the ported
# series. Listing them per approach keeps the DAG honest instead of failing at job time.
def SeriesForApproach(approach):
    declared = config["approaches"][approach].get("series")
    if not declared:
        return SERIES
    unknown = [s for s in declared if s not in SERIES]
    if unknown:
        raise ValueError(f"approach {approach!r} lists unknown series {unknown}; "
                         f"known series are {SERIES}")
    return list(declared)


APPROACH_SERIES = {approach: SeriesForApproach(approach) for approach in APPROACHES}


# Series with >1 unique non-control Treatment (i.e. Treatment != control_treatment).
# SpecificityTest is only meaningful for these — single-drug series produce a
# drug-vs-DMSO comparison that is already captured by the dose-response model itself.
def _n_nondmso_treatments(df):
    return df.loc[df["Treatment"] != df["control_treatment"], "Treatment"].nunique()

SERIES_WITH_MULTIPLE_TREATMENTS = [
    s for s in SERIES
    if _n_nondmso_treatments(_samples[_samples["Series"] == s]) > 1
]


# ---------------------------------------------------------------------------
# Memory scaling helper for successive SLURM retry attempts
# ---------------------------------------------------------------------------

def GetMemForSuccessiveAttempts(*args, max_mb=48000):
    """Return a Snakemake resource function that scales memory by attempt.

    Usage examples:
        resources: mem_mb = GetMemForSuccessiveAttempts(8000, 24000)
            # attempt 1 → 8000 MB, attempt 2 → 24000 MB, attempt 3+ → 48000 MB

        resources: mem_mb = GetMemForSuccessiveAttempts(58000)
            # attempt 1 → 58000 MB, attempt 2+ → 48000 MB (default max)
    """
    def ReturnMemMb(wildcards, attempt):
        i = int(attempt) - 1
        try:
            return args[i]
        except IndexError:
            return max_mb
    return ReturnMemMb


def InputArgsForApproach(wildcards, config, n_batches):
    """The input-related CLI arguments for one fit job, for either input path.

    Snakemake's {n} runs 0..n_batches-1 while --chunks counts 1..M (0 being the header-only
    chunk, which the pipeline does not need because the gather step reads the TSVs rather than
    concatenating them), hence the +1.
    """
    approach = config["approaches"][wildcards.Approach]
    matrices = approach.get("matrices")
    if not matrices:
        return f"--inputlong DoseResponseModelling/{wildcards.Approach}/DataBatched/{wildcards.series}/{wildcards.n}.tsv.gz"
    args = " ".join(f"--matrix {outcome} {path}" for outcome, path in matrices.items())
    design = f"DoseResponseModelling/{wildcards.Approach}/Designs/{wildcards.series}.tsv"
    return f"{args} --design {design} --chunks {int(wildcards.n) + 1} {n_batches}"
