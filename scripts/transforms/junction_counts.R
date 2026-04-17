#!/usr/bin/env Rscript
# Transform leafcutter junction counts bed into tidy y/n count table per Series.
#
# Usage: Rscript junction_counts.R <samples_fn> <junc_counts_fn> <output_dir>/
#
# Input junc_counts_fn: BED6+ leafcutter junction count file where:
#   col named "junc" = junction ID (featureID)
#   col named "gid"  = gene cluster ID (used to compute denominator n)
#   remaining cols   = one column per sample, values are junction read counts
#
# Output: one TSV per Series in output_dir/{series}_TidyDataForModelling.tsv.gz
#   Columns: featureID, treatment, sample, dose, y (included count), n (total cluster count)

if (interactive()) {
    args <- scan(text = paste(
        "config/samples.tsv",
        "rna-seq/SplicingAnalysis/leafcutter/GRCh38_GencodeRelease44Comprehensive/juncTableBeds/JuncCounts.sorted.bed.gz",
        "DoseResponseModelling/Data/SplicingPSI"
    ), what = "character")
} else {
    args <- commandArgs(trailingOnly = TRUE)
}

samples_fn     <- args[1]
junc_counts_fn <- args[2]
output_dir     <- args[3]

library(tidyverse)
library(data.table)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

samples     <- read_tsv(samples_fn)
junc_counts <- fread(junc_counts_fn)

series_list <- unique(samples$Series)

for (s in series_list) {
    series_samples <- samples %>%
        filter(Series == s) %>%
        dplyr::select(sample, Treatment, dose = dose.nM) %>%
        replace_na(list(dose = 0))

    sample_ids <- series_samples$sample

    junc_counts %>%
        dplyr::select(featureID = junc, gid, all_of(sample_ids)) %>%
        pivot_longer(
            cols      = all_of(sample_ids),
            names_to  = "sample",
            values_to = "y"
        ) %>%
        inner_join(series_samples, by = "sample") %>%
        group_by(sample, gid) %>%
        mutate(n = sum(y)) %>%
        ungroup() %>%
        dplyr::select(featureID, treatment = Treatment, sample, dose, y, n) %>%
        write_tsv(file.path(output_dir, paste0(s, "_TidyDataForModelling.tsv.gz")))

    message("Wrote Series: ", s)
}
