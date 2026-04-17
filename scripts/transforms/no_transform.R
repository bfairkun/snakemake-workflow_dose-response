#!/usr/bin/env Rscript
# Tidy a bed6+ feature-by-sample table without transforming values.
# Use this when the feature_by_sample_table already contains the desired
# response variable (e.g. pre-computed fold-changes, PSI, etc.).
#
# Usage: Rscript no_transform.R <samples_fn> <bed_fn> <output_dir>/
#
# Input bed_fn: BED6+ file where:
#   col 1-6 = standard BED6 (chrom, start, end, name, score, strand)
#   col 4 (name) = featureID
#   col 7+ = one column per sample
#
# Output: one TSV per Series in output_dir/{series}_TidyDataForModelling.tsv.gz
#   Columns: featureID, treatment, sample, dose, y (raw value from bed_fn)

if (interactive()) {
    args <- scan(text = paste(
        "config/dose_response_samples.tsv",
        "path/to/feature_by_sample_table.bed.gz",
        "DoseResponseModelling/Data/SomeApproach"
    ), what = "character")
} else {
    args <- commandArgs(trailingOnly = TRUE)
}

samples_fn <- args[1]
bed_fn     <- args[2]
output_dir <- args[3]

library(tidyverse)
library(data.table)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

samples <- read_tsv(samples_fn)

# Read bed6+ file
bed <- fread(bed_fn)
col_names   <- names(bed)
feature_col <- col_names[4]    # BED col 4 = name = featureID
sample_cols <- col_names[7:length(col_names)]

# Pivot to long format once — then split by Series
bed_long <- bed %>%
    dplyr::select(featureID = all_of(feature_col), all_of(sample_cols)) %>%
    pivot_longer(
        cols      = all_of(sample_cols),
        names_to  = "sample",
        values_to = "y"
    )

series_list <- unique(samples$Series)

for (s in series_list) {
    series_samples <- samples %>%
        filter(Series == s) %>%
        dplyr::select(sample, Treatment, dose = dose.nM) %>%
        replace_na(list(dose = 0))

    bed_long %>%
        inner_join(series_samples, by = "sample") %>%
        dplyr::select(featureID, treatment = Treatment, sample, dose, y) %>%
        write_tsv(file.path(output_dir, paste0(s, "_TidyDataForModelling.tsv.gz")))

    message("Wrote Series: ", s)
}
