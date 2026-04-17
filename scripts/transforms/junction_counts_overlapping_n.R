#!/usr/bin/env Rscript
# Transform leafcutter junction counts bed into tidy y/n count table per Series.
# Identical to junction_counts.R except n is computed as the sum of read counts
# from junctions in the same cluster whose genomic interval overlaps the junction
# in question (start_other < end_J AND end_other > start_J), rather than the full
# cluster sum. This avoids inflating n with distant constitutive junctions that
# don't compete with the junction of interest.
#
# Usage: Rscript junction_counts_overlapping_n.R <samples_fn> <junc_counts_fn> <output_dir>/
#
# Memory-efficient strategy:
#   1. Build overlap_map (featureID → featureID_other pairs) ONCE from coordinate
#      columns only — this is small regardless of sample count.
#   2. For each series, join overlap_map with the series-specific long data to
#      compute n, avoiding a costly long-format self-join.

if (interactive()) {
    args <- scan(text = paste(
        "config/samples.tsv",
        "rna-seq/SplicingAnalysis/leafcutter/GRCh38_GencodeRelease44Comprehensive/juncTableBeds/JuncCounts.sorted.bed.gz",
        "DoseResponseModelling/Data/SplicingPSI_OverlappingJuncsForN"
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

# Rename the first column (#Chrom → chrom) to avoid backtick issues
setnames(junc_counts, old = names(junc_counts)[1], new = "chrom")

# ---------------------------------------------------------------------------
# Step 1: Build overlap_map once from coordinate data only.
# overlap_map[i]: featureID_other junctions whose interval overlaps featureID_i.
# Using a data.table self-join by gid keeps this entirely in coordinate space —
# no per-sample counts involved, so this stays small.
# ---------------------------------------------------------------------------
junc_coords <- junc_counts[, .(featureID = junc, gid, s = start, e = end)]

overlap_map <- junc_coords[
    junc_coords, on = "gid", allow.cartesian = TRUE, nomatch = NULL
][s < i.e & i.s < e, .(featureID, featureID_other = i.featureID)]

message("Built overlap_map: ", nrow(overlap_map), " pairs across ",
        uniqueN(overlap_map$featureID), " junctions")

# ---------------------------------------------------------------------------
# Step 2: For each series, compute n using overlap_map + series-specific y.
# ---------------------------------------------------------------------------
series_list <- unique(samples$Series)

for (s in series_list) {
    series_samples <- samples %>%
        filter(Series == s) %>%
        dplyr::select(sample, Treatment, dose = dose.nM) %>%
        replace_na(list(dose = 0))

    sample_ids <- series_samples$sample

    # Long-format y values for this series only (no coordinate columns needed)
    long_series <- junc_counts[, c("junc", "gid", sample_ids), with = FALSE] %>%
        pivot_longer(
            cols      = all_of(sample_ids),
            names_to  = "sample",
            values_to = "y"
        ) %>%
        rename(featureID = junc)

    # Compute n: for each (featureID, sample), sum y of overlapping junctions.
    # Join overlap_map with long_series ON featureID_other to get y_other, then
    # aggregate. This join is (n_overlap_pairs) × (n_samples) — much cheaper
    # than a long-format self-join.
    n_df <- overlap_map %>%
        inner_join(
            long_series %>% dplyr::select(featureID_other = featureID, sample, y_other = y),
            by = "featureID_other",
            relationship = "many-to-many"
        ) %>%
        group_by(featureID, sample) %>%
        summarize(n = sum(y_other), .groups = "drop")

    long_series %>%
        inner_join(series_samples, by = "sample") %>%
        left_join(n_df, by = c("featureID", "sample")) %>%
        dplyr::select(featureID, treatment = Treatment, sample, dose, y, n) %>%
        write_tsv(file.path(output_dir, paste0(s, "_TidyDataForModelling.tsv.gz")))

    message("Wrote Series: ", s)
}
