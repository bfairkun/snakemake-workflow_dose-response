import pandas as pd
from snakemake.utils import validate

# ---------------------------------------------------------------------------
# Load config values at workflow parse time
# ---------------------------------------------------------------------------

APPROACHES = config.get("approaches", {})
N_BATCHES  = int(config.get("n_batches", 200))

# Infer the list of Series (modeling groups) from the samples.tsv at parse time.
# This is safe because samples.tsv is a static input file, not a workflow output.
_samples = pd.read_csv(config["samples"], sep="\t")
SERIES   = _samples["Series"].dropna().unique().tolist() if "Series" in _samples.columns else []
validate(_samples, "../schemas/samples.schema.yaml")


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
