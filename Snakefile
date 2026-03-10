configfile: "config/config.yaml"
include: "rules/common.py"

include: "rules/DoseResponse.smk"

rule all:
    input:
        rules.GatherAll.input
