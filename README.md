# snakemake-workflow_dose-response

A reusable Snakemake module for Bayesian dose-response modeling of RNA-seq data.

Given feature-by-sample count/expression tables and a samples metadata file, this
workflow fits log-logistic dose-response models (via PyMC) to either gene expression
log2 fold-change or splicing junction counts, across any number of user-defined
experimental series and modeling approaches.

## Pipeline overview

```
feature_by_sample_table (BED6+)
          │
          ▼
    CreateTidyData              ← one job per Approach; loops over Series
          │
          ▼
SeparateTidyDataIntoBatches     ← one job per Approach × Series
          │
          ▼
FitBayesianDoseResponse_ByBatch ← N_BATCHES jobs per Approach × Series (cluster-parallel)
          │
          ▼
FitBayesianDoseResponse_GatherBatches
  SpecificityTest               ← one job per Approach × Series
  WriteSQLite                   ← one job (all batches → SQLite)
```

## Quickstart (standalone)

```bash
# 1. Create a samples.tsv with columns: sample, Series, Treatment, dose.nM
# 2. Copy config/config.yaml and edit the approaches section
# 3. Run:
cd /your/project/workdir
snakemake --snakefile path/to/snakemake-workflow_dose-response/Snakefile \
          --configfile your_config.yaml \
          --cores 8
```

## Use as a Snakemake module (recommended)

Add this repo as a git submodule and import it in your project's `Snakefile`:

```python
module dose_response:
    snakefile: "module_workflows/snakemake-workflow_dose-response/Snakefile"
    prefix: "dose-response"
    config: config["dose_response"]

use rule * from dose_response as dose_response_*
CreateSymlinksOfDir1ContentsIntoDir2(
    "module_workflows/snakemake-workflow_dose-response/scripts/",
    "scripts/"
)
```

In your `config/config.yaml`, add a `dose_response:` section overriding any defaults:

```yaml
dose_response:
  samples: "config/samples.tsv"
  n_batches: 200
  pytensor_scratch: "/scratch/midway3/youruser/"
  approaches:
    ExpressionLogFC:
      feature_by_sample_table: "rna-seq/ExpressionMatrices/.../log2Filtered_TMM_CPM.sorted.bed.gz"
      tidy_transform: "expression_log2fc"
      model_params: "--model 1 --PreFilterByNumberReasonableObservedOutcomes y 2 -100 -1 --PreFilterByNumberReasonableObservedOutcomes y 2 1 100"
    SplicingPSI:
      feature_by_sample_table: "rna-seq/SplicingAnalysis/leafcutter/.../JuncCounts.sorted.bed.gz"
      tidy_transform: "junction_counts"
      model_params: "--model 2 --PreFilterByNumberReasonableObservedOutcomes n 5 10 100000 --PreFilterByNumberReasonableObservedOutcomes y 3 3 100000 --PosteriorFilter MaxDeltaPSI 0.95 -1 -0.1 --PosteriorFilter MaxDeltaPSI 0.95 0.1 1"
```

## `samples.tsv` schema

| Column | Description |
|--------|-------------|
| `sample` | Unique sample identifier matching column headers in `feature_by_sample_table` |
| `Series` | Modeling group — all samples that share a dose-response design and DMSO controls. Samples can appear in multiple Series (as multiple rows). |
| `Treatment` | Drug/condition name (e.g. "Branaplam", "DMSO") |
| `dose.nM` | Numeric dose in nM; DMSO controls use `0` |

Additional columns are allowed and ignored.

## Config reference

| Key | Default | Description |
|-----|---------|-------------|
| `samples` | `config/samples.tsv` | Path to samples metadata |
| `n_batches` | `200` | Number of parallel fitting batches per Approach × Series |
| `pytensor_scratch` | `/tmp/` | Scratch dir for PyTensor cache (use fast local storage on HPC) |
| `approaches` | see config.yaml | Dict of named approaches; each needs `feature_by_sample_table`, `tidy_transform`, `model_params` |

## Adding a custom approach

1. Create `scripts/transforms/myapproach.R` following the interface:
   ```
   Rscript myapproach.R <samples_fn> <feature_by_sample_table> <output_dir>/
   ```
   Output one file per Series: `{output_dir}/{series}_TidyDataForModelling.tsv.gz`
   Required columns: `featureID`, `treatment`, `sample`, `dose`, and data columns (`y` for model 1; `y` + `n` for model 2).

2. Add a new entry under `approaches:` in your config:
   ```yaml
   approaches:
     MyApproach:
       feature_by_sample_table: "path/to/my_table.bed.gz"
       tidy_transform: "myapproach"
       model_params: "--model 1 ..."
   ```

## Outputs

| Path | Description |
|------|-------------|
| `DoseResponseModelling/Data/{Approach}/{series}_TidyDataForModelling.tsv.gz` | Tidy per-Series data for modeling |
| `DoseResponseModelling/{Approach}/DataBatched/{series}/` | Batched input files |
| `DoseResponseModelling/{Approach}/ResultsBatched/{series}/{n}.pkl` | Per-batch InferenceData pickles |
| `DoseResponseModelling/{Approach}/Results/{series}.pkl` | Gathered InferenceData (all features) |
| `DoseResponseModelling/{Approach}/Results/{series}.tsv.gz` | Summary table (posterior means, HDIs, R²) |
| `DoseResponseModelling/{Approach}/SpecificityTest/{series}_SpecificityTestResults.tsv.gz` | Pairwise treatment specificity (FSP, q-values) |
| `DoseResponseModelling/InferenceDataResults.sqlite` | All InferenceData objects as NetCDF BLOBs |

## Conda environments

| File | Used for |
|------|----------|
| `envs/pymc.yaml` | Bayesian fitting (PyMC 5.x, ArviZ, SQLite export) |
| `envs/r_deps.yaml` | Tidy data preparation (tidyverse, data.table) |
