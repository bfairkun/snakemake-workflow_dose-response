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
        output_dir = lambda wc: f"DoseResponseModelling/Data/{wc.Approach}"
    log:
        "logs/CreateTidyData.{Approach}.log"
    conda:
        "../envs/r_deps.yaml"
    resources:
        mem_mb = GetMemForSuccessiveAttempts(24000, 48000)
    shell:
        """
        Rscript scripts/transforms/{params.transform}.R \
            {input.samples} \
            {input.feature_by_sample_table} \
            {params.output_dir}/ \
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


rule FitBayesianDoseResponse_ByBatch:
    """Fit Bayesian dose-response model to one batch of features."""
    input:
        "DoseResponseModelling/{Approach}/DataBatched/{series}/{n}.tsv.gz"
    output:
        pkl = "DoseResponseModelling/{Approach}/ResultsBatched/{series}/{n}.pkl",
        tsv = "DoseResponseModelling/{Approach}/ResultsBatched/{series}/{n}.tsv.gz"
    log:
        "logs/FitBayesianDoseResponse_ByBatch.{Approach}.{series}.{n}.log"
    conda:
        "../envs/pymc.yaml"
    params:
        extra           = lambda wc: config["approaches"][wc.Approach]["model_params"],
        pytensor_scratch = config.get("pytensor_scratch", "/tmp/")
    resources:
        mem_mb = GetMemForSuccessiveAttempts(58000)
    shell:
        """
        export PYTENSOR_FLAGS="compiledir={params.pytensor_scratch}/${{SLURM_JOBID:-$$}}/pytensor_cache_${{RANDOM}},force_compile=True" && \
        python scripts/BayesianDoseResponse_ByBatch.py \
            --input {input} \
            --output_pkl {output.pkl} \
            --output_tsv {output.tsv} \
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
        "DoseResponseModelling/{Approach}/SpecificityTest/{series}_SpecificityTestResults.tsv.gz"
    log:
        "logs/SpecificityTest.{Approach}.{series}.log"
    conda:
        "../envs/pymc.yaml"
    resources:
        mem_mb = 48000
    shell:
        """
        python scripts/DoseResponseSpecificityTest_cli.py \
            --infile {input.pkl} \
            --outfile {output} \
            &> {log}
        """


rule WriteSQLite:
    """Write all batch InferenceData pkl files into a single queryable SQLite database."""
    input:
        pkls = expand(
            "DoseResponseModelling/{Approach}/ResultsBatched/{series}/{n}.pkl",
            Approach=list(APPROACHES.keys()),
            series=SERIES,
            n=range(N_BATCHES)
        )
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
        expand(
            "DoseResponseModelling/{Approach}/Results/{series}.pkl",
            Approach=list(APPROACHES.keys()),
            series=SERIES
        ),
        expand(
            "DoseResponseModelling/{Approach}/Results/{series}.tsv.gz",
            Approach=list(APPROACHES.keys()),
            series=SERIES
        ),
        expand(
            "DoseResponseModelling/{Approach}/SpecificityTest/{series}_SpecificityTestResults.tsv.gz",
            Approach=list(APPROACHES.keys()),
            series=SERIES_WITH_MULTIPLE_TREATMENTS
        ),
        "DoseResponseModelling/InferenceDataResults.sqlite"
