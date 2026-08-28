#!/usr/bin/env Rscript
# Tidy an absolute log2 abundance matrix (e.g. log2TPM) without transforming values.
#
# Usage: Rscript expression_absolute.R <samples_fn> <bed_fn> <output_dir>/
#
# Pair with `--model 3`, which treats the untreated level as a free parameter `lower` rather
# than subtracting a baseline. Nothing is differenced here, so `y` stays on the absolute scale
# and `lower` is directly interpretable (in log2TPM, `lower > 0` means above 1 TPM).
#
# NO GENE FILTERING is done here, deliberately. Unexpressed features are removed by the
# fitting script's pre-filters, which cost well under a millisecond per feature and ask the
# question per series rather than once globally. On log2TPM the expression gate is
#     --PreFilterByNumberReasonableObservedOutcomes y 3 0 100
# i.e. at least 3 observations with y in [0, 100] log2 units == TPM >= 1. Note the bounds are
# in the units of `y`, so 100 is an arbitrarily high log2 ceiling, not TPM 100.
#
# `--AbsSpearmanPreFilter` alone is NOT sufficient to drop unexpressed features: with few
# treated points, |rho| >= 0.4 arises by chance for roughly a third of never-expressed genes.
#
# Input bed_fn: BED6+ file where:
#   col 1-6 = standard BED6 (chrom, start, end, name, score, strand)
#   col 4 (name) = featureID
#   col 7+ = one column per sample, values on an absolute log2 scale
#
# Required samples columns: sample, Series, Treatment, dose.nM
# Optional: exclude_expression
#
# Output: one TSV per Series in output_dir/{series}_TidyDataForModelling.tsv.gz
#   Columns: featureID, treatment, sample, dose, y

if (interactive()) {
    args <- scan(text = paste(
        "config/dose_response_samples.tsv",
        "rna_seq/ExpressionMatrices/GRCh38_GencodeRelease44Comprehensive/log2TPM.sorted.bed.gz",
        "DoseResponseModelling/Data/ExpressionAbsoluteTPM"
    ), what = "character")
} else {
    args <- commandArgs(trailingOnly = TRUE)
}

samples_fn <- args[1]
bed_fn     <- args[2]
output_dir <- args[3]

# Optional 4th argument: minimum feature span in bp. OFF unless supplied. Enable per approach
# with `transform_args: "--min_feature_length 200"` in config.
#
# This targets tiny annotations (GENCODE 44 has features down to 8 bp) whose pseudocount-only
# value can clear an expression threshold, producing a feature whose dose series just reads out
# the per-sample normalization factor. Note it filters on the BED span (end - start), which for
# these single-exon features is effectively the exonic length, but for multi-exon genes is the
# genomic span and therefore larger than featureCounts' summed exonic `Length`.
min_feature_length <- NA_real_
if (length(args) >= 4) {
    extra <- args[-(1:3)]
    i <- which(extra == "--min_feature_length")
    if (length(i) == 1 && length(extra) >= i + 1) {
        min_feature_length <- as.numeric(extra[i + 1])
    } else if (length(extra) == 1 && !is.na(suppressWarnings(as.numeric(extra)))) {
        min_feature_length <- as.numeric(extra)
    }
}

library(tidyverse)
library(data.table)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

samples <- read_tsv(samples_fn)

# Optional per-approach exclusion. A sample can be legitimately usable for one approach and
# not another: libraries with a 3'-biased coverage profile distort gene-level expression
# (long transcripts under-counted) while relative junction counts within a cluster are far
# more robust to it, because numerator and denominator share the same local bias.
#
# This matters more on an absolute scale than for the log2FC transforms: there is no baseline
# subtraction to partially cancel a length-dependent tilt shared with a similarly degraded
# control.
if ("exclude_expression" %in% names(samples)) {
    drop <- samples %>% filter(toupper(as.character(exclude_expression)) == "TRUE")
    if (nrow(drop) > 0) {
        message("Excluding ", nrow(drop), " sample-series row(s): ",
                paste(unique(drop$sample), collapse = ", "))
        samples <- samples %>% filter(!toupper(as.character(exclude_expression)) == "TRUE")
    }
}

bed <- fread(bed_fn)
col_names   <- names(bed)
feature_col <- col_names[4]    # BED col 4 = name = featureID
sample_cols <- col_names[7:length(col_names)]

if (!is.na(min_feature_length)) {
    span <- bed[[col_names[3]]] - bed[[col_names[2]]]
    keep <- span >= min_feature_length
    message("Dropping ", sum(!keep), " of ", length(keep),
            " features with BED span < ", min_feature_length, " bp")
    bed <- bed[keep]
}

series_list <- unique(samples$Series)

# Subset columns per series before pivoting. With no gene filter this table is ~62k features
# wide-by-209, and pivoting the whole thing once then joining per series peaks much higher
# than pivoting only the columns each series needs.
for (s in series_list) {
    series_samples <- samples %>%
        filter(Series == s) %>%
        dplyr::select(sample, Treatment, dose = dose.nM) %>%
        replace_na(list(dose = 0))

    sample_ids <- intersect(series_samples$sample, sample_cols)
    if (length(sample_ids) == 0) {
        warning("No matching sample columns for Series ", s, "; skipping")
        next
    }

    bed[, c(feature_col, sample_ids), with = FALSE] %>%
        dplyr::rename(featureID = all_of(feature_col)) %>%
        pivot_longer(cols = all_of(sample_ids), names_to = "sample", values_to = "y") %>%
        inner_join(series_samples, by = "sample") %>%
        dplyr::select(featureID, treatment = Treatment, sample, dose, y) %>%
        write_tsv(file.path(output_dir, paste0(s, "_TidyDataForModelling.tsv.gz")))

    message("Wrote Series: ", s, " (", length(sample_ids), " samples)")
}
