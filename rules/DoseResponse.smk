# ---------------------------------------------------------------------------
# Dose-Response Modelling Rules
# ---------------------------------------------------------------------------
# All output lives under DoseResponseModelling/.
# Wildcards: {Approach} (from config approaches dict), {series} (from samples.tsv),
#            {n} (batch index 0..N_BATCHES-1)
# ---------------------------------------------------------------------------


rule CreateTidyData:
    """Convert a raw feature-by-sample table into per-Series tidy TSVs.

    The transform script is specified per approach via config["approaches"][Approach]["tidy_transform"].
    Built-in transforms live in scripts/transforms/; custom ones can be added there.
    All transform scripts share the same 3-argument interface:
        Rscript <script> <samples_fn> <feature_by_sample_table> <output_dir>/
    Output: {output_dir}/{series}_TidyDataForModelling.tsv.gz for each series.
    """
    input:
        samples                = config["samples"],
        feature_by_sample_table = lambda wc: config["approaches"][wc.Approach]["feature_by_sample_table"]
    output:
        expand(
            "DoseResponseModelling/Data/{{Approach}}/{series}_TidyDataForModelling.tsv.gz",
            series=SERIES
        )
    params:
        transform  = lambda wc: config["approaches"][wc.Approach]["tidy_transform"],
        output_dir = lambda wc: f"DoseResponseModelling/Data/{wc.Approach}",
        # Optional extra arguments appended after the standard three. Transforms that do not
        # read them ignore them, so this is safe for every approach.
        transform_args = lambda wc: config["approaches"][wc.Approach].get("transform_args", "")
    log:
        "logs/CreateTidyData.{Approach}.log"
    conda:
        "../envs/r_deps.yaml"
    resources:
        # 24000 reliably OOMs on the splicing tables (~23G used) and each failed attempt
        # costs ~27 min before the 48000 retry even starts, so start at 48000.
        mem_mb = GetMemForSuccessiveAttempts(48000, 64000)
    shell:
        """
        Rscript scripts/transforms/{params.transform}.R \
            {input.samples} \
            {input.feature_by_sample_table} \
            {params.output_dir}/ \
            {params.transform_args} \
            &> {log}
        """


rule SeparateTidyDataIntoBatches:
    """Split one Series × Approach tidy TSV into N_BATCHES smaller files for parallel fitting."""
    input:
        "DoseResponseModelling/Data/{Approach}/{series}_TidyDataForModelling.tsv.gz"
    output:
        outdir  = directory("DoseResponseModelling/{Approach}/DataBatched/{series}"),
        batches = expand(
            "DoseResponseModelling/{{Approach}}/DataBatched/{{series}}/{n}.tsv.gz",
            n=range(N_BATCHES)
        )
    params:
        n_batches = N_BATCHES
    log:
        "logs/SeparateTidyDataIntoBatches.{Approach}.{series}.log"
    conda:
        "../envs/r_deps.yaml"
    resources:
        mem_mb = 12000
    shell:
        """
        Rscript scripts/SeparateTidyDataIntoBatches.R \
            {input} \
            DoseResponseModelling/{wildcards.Approach}/DataBatched/{wildcards.series}/ \
            {params.n_batches} \
            &> {log}
        """


rule CreateSeriesCovariateMatrix:
    """Pivot the long-format covariate declarations into one series' sample x covariate matrix.

    The long file (Series, sample, covariate, value) is the thing a human edits: a row's
    presence declares that the covariate is to be estimated in that series. The fitter works
    one series at a time and takes the matrix, so the reshape belongs here. A series with no
    rows yields a header-only file, which the fitter treats as covariate-free.
    """
    input:
        covariates = lambda wc: config["approaches"][wc.Approach]["covariates"]
    output:
        "DoseResponseModelling/{Approach}/CovariateMatrices/{series}.tsv"
    log:
        "logs/CreateSeriesCovariateMatrix.{Approach}.{series}.log"
    run:
        import pandas as pd
        long = pd.read_csv(input.covariates, sep="\t")
        sub = long[long["Series"].astype(str) == wildcards.series]
        if sub.empty:
            pd.DataFrame({"sample": []}).to_csv(output[0], sep="\t", index=False)
        else:
            wide = sub.pivot(index="sample", columns="covariate", values="value")
            wide.columns.name = None
            wide.reset_index().to_csv(output[0], sep="\t", index=False)


rule CreateSeriesDesignFile:
    """Sample-level design for one Series x Approach: sample, treatment, dose.

    This is all the fitter needs besides the matrices -- dose and treatment are properties of a
    sample, so there is no reason to repeat them once per feature. It is per Approach, not just
    per Series, because `exclude_expression` drops degraded libraries from the expression
    approaches but not the splicing ones, so the two see different sample sets.
    """
    input:
        samples = config["samples"]
    output:
        "DoseResponseModelling/{Approach}/Designs/{series}.tsv"
    params:
        exclude_flag = lambda wc: config["approaches"][wc.Approach].get("exclude_flag", "")
    log:
        "logs/CreateSeriesDesignFile.{Approach}.{series}.log"
    run:
        import pandas as pd
        samples = pd.read_csv(input.samples, sep="\t")
        sub = samples[samples["Series"].astype(str) == wildcards.series]
        flag = params.exclude_flag
        if flag and flag in sub.columns:
            sub = sub[sub[flag].astype(str).str.upper() != "TRUE"]
        out = pd.DataFrame({
            "sample": sub["sample"],
            "treatment": sub["Treatment"],
            "dose": sub["dose.nM"],
        }).drop_duplicates()
        out.to_csv(output[0], sep="\t", index=False)


rule FitBayesianDoseResponse_ByBatch:
    """Fit Bayesian dose-response model to one batch of features."""
    input:
        # A matrix approach reads the matrices directly and slices its own chunk, so it needs
        # neither the tidy data nor the pre-split batch files.
        data = lambda wc: (list(config["approaches"][wc.Approach]["matrices"].values())
                            + [f"DoseResponseModelling/{wc.Approach}/Designs/{wc.series}.tsv"]
                           if config["approaches"][wc.Approach].get("matrices")
                           else [f"DoseResponseModelling/{wc.Approach}/DataBatched/{wc.series}/{wc.n}.tsv.gz"]),
        # Declared as a real input (not buried in the model_params string) so that editing the
        # covariate table re-triggers the fits. Empty list when the approach declares none.
        covariates = lambda wc: [f"DoseResponseModelling/{wc.Approach}/CovariateMatrices/{wc.series}.tsv"]
                                if config["approaches"][wc.Approach].get("covariates") else []
    output:
        pkl = "DoseResponseModelling/{Approach}/ResultsBatched/{series}/{n}.pkl",
        tsv = "DoseResponseModelling/{Approach}/ResultsBatched/{series}/{n}.tsv.gz"
    log:
        "logs/FitBayesianDoseResponse_ByBatch.{Approach}.{series}.{n}.log"
    conda:
        "../envs/pymc.yaml"
    params:
        extra           = lambda wc: config["approaches"][wc.Approach]["model_params"],
        covariates      = lambda wc: ("--covariates DoseResponseModelling/"
                                      f"{wc.Approach}/CovariateMatrices/{wc.series}.tsv")
                                     if config["approaches"][wc.Approach].get("covariates") else "",
        pytensor_scratch = config.get("pytensor_scratch", ""),
        input_args = lambda wc: InputArgsForApproach(wc, config, N_BATCHES)
    resources:
        mem_mb = GetMemForSuccessiveAttempts(16000, 48000, max_mb=64000)
    shell:
        """
        # $TMPDIR is Slurm's per-job dir; a shared /tmp is swept by the epilog when any
        # sibling job of the same user ends on the node, deleting a live compile cache.
        CacheBase="{params.pytensor_scratch}" && \
        CacheBase="${{CacheBase:-${{TMPDIR:-/tmp}}}}" && \
        export PYTENSOR_FLAGS="compiledir=$CacheBase/pytensor_cache_${{SLURM_JOBID:-$$}}" && \
        python scripts/BayesianDoseResponse_ByBatch.py \
            {params.input_args} \
            --output_pkl {output.pkl} \
            --output_tsv {output.tsv} \
            {params.covariates} \
            {params.extra} \
            &> {log}
        """


rule FitBayesianDoseResponse_GatherBatches:
    """Combine all batch results for one Series × Approach into a single pkl + tsv."""
    input:
        pkls = expand(
            "DoseResponseModelling/{{Approach}}/ResultsBatched/{{series}}/{n}.pkl",
            n=range(N_BATCHES)
        ),
        tsvs = expand(
            "DoseResponseModelling/{{Approach}}/ResultsBatched/{{series}}/{n}.tsv.gz",
            n=range(N_BATCHES)
        )
    output:
        pkl = "DoseResponseModelling/{Approach}/Results/{series}.pkl",
        tsv = "DoseResponseModelling/{Approach}/Results/{series}.tsv.gz"
    log:
        "logs/FitBayesianDoseResponse_GatherBatches.{Approach}.{series}.log"
    conda:
        "../envs/pymc.yaml"
    resources:
        mem_mb = 58000
    shell:
        """
        python scripts/BayesianDoseResponse_GatherBatches.py \
            --tsvs {input.tsvs} \
            --pkls {input.pkls} \
            --output_tsv {output.tsv} \
            --output_pkl {output.pkl} \
            &> {log}
        """


rule SpecificityTest:
    """Compare posterior dose-response curves across treatments to identify modulator-specific effects."""
    input:
        pkl = "DoseResponseModelling/{Approach}/Results/{series}.pkl"
    output:
        "DoseResponseModelling/{Approach}/SpecificityTest/{series}/{posterior_param}_SpecificityTestResults.tsv.gz"
    log:
        "logs/SpecificityTest.{Approach}.{series}.{posterior_param}.log"
    conda:
        "../envs/pymc.yaml"
    resources:
        mem_mb = 48000
    shell:
        """
        python scripts/DoseResponseSpecificityTest_cli.py \
            --infile {input.pkl} \
            --outfile {output} \
            --posterior_param {wildcards.posterior_param} \
            &> {log}
        """


rule WriteSQLite:
    """Write all batch InferenceData pkl files into a single queryable SQLite database."""
    input:
        pkls = [f"DoseResponseModelling/{approach}/ResultsBatched/{series}/{n}.pkl"
                for approach in APPROACHES
                for series in APPROACH_SERIES[approach]
                for n in range(N_BATCHES)]
    output:
        db = "DoseResponseModelling/InferenceDataResults.sqlite"
    log:
        "logs/WriteSQLite.log"
    conda:
        "../envs/pymc.yaml"
    resources:
        mem_mb = 8000
    shell:
        """
        python scripts/WriteSQLite_AllBatches.py \
            --output_db {output.db} \
            --batch_size 100 \
            {input.pkls} \
            &> {log}
        """


rule GatherAll:
    """Aggregate target: all gathered results, specificity tests, and SQLite database."""
    input:
        [f"DoseResponseModelling/{approach}/Results/{series}.pkl"
         for approach in APPROACHES for series in APPROACH_SERIES[approach]],
        [f"DoseResponseModelling/{approach}/Results/{series}.tsv.gz"
         for approach in APPROACHES for series in APPROACH_SERIES[approach]],
        [
            f"DoseResponseModelling/{approach}/SpecificityTest/{series}/{param}_SpecificityTestResults.tsv.gz"
            for approach in APPROACHES
            for series in SERIES_WITH_MULTIPLE_TREATMENTS
            if series in APPROACH_SERIES[approach]
            for param in APPROACH_SPECIFICITY_PARAMS[approach]
        ],
        "DoseResponseModelling/InferenceDataResults.sqlite"
