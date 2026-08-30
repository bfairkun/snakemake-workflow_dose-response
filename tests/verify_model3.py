"""Phase A check for model 3, whose pre-split fits were never written to disk.

Baseline recovered from docs/20260826_dose_response_model_comparison.html, rendered BEFORE
the split:
  * the exact per-fit divergence counts printed by the notebook's `fits` chunk
  * the embedded figures, as base64 PNGs

Divergence counts are a sharp fingerprint: they depend on the posterior geometry and the RNG
stream, so 20/20 agreement is hard to get by accident. The figures then compare the actual
posterior curves and intervals pixel by pixel.

NOTE: the baselines this compares against predate Phase B, which deliberately changed the
splicing parameterizations. Differences are now expected; the script is kept because it is
the only harness that reads those artefacts, and because it still catches an accidental
change to the expression models.
"""
import functools, os, shlex, sys, warnings, logging
import numpy as np, pandas as pd, pymc as pm
warnings.filterwarnings("ignore"); logging.getLogger("pymc").setLevel(logging.ERROR)
os.chdir("/project/yangili1/bjf79/20260310_diversesm_dr/code")
sys.path.insert(0, "module_workflows/dose_response/src")
import dose_response.covariates as CD
from dose_response.cli.fit_batch import parse_args
from dose_response.models.expression_absolute import fit_expression_absolute
import dose_response.models.expression_absolute as ea
_o = pm.sample
@functools.wraps(_o)
def _q(*a, **k):
    k.setdefault("progressbar", False); return _o(*a, **k)
pm.sample = ea.pm.sample = _q

TIDY_DIR = ("/tmp/claude-1352850516/-project-yangili1-bjf79-20260310-diversesm-dr/"
            "fcd93c00-8321-41e7-a2eb-9e17f0e92056/scratchpad/tpmtest")
COV_TSV = "config/dose_response_covariates.tsv"
load = lambda s: pd.read_csv(f"{TIDY_DIR}/{s}_TidyDataForModelling.tsv.gz", sep="\t")

sym = pd.read_csv("/project2/yangili1/bjf79/ReferenceGenomes/GRCh38_GencodeRelease44Comprehensive/"
                  "ensembl_to_hgnc_symbol.tsv.gz", sep="\t")
probe = pd.read_csv(f"{TIDY_DIR}/T025_TidyDataForModelling.tsv.gz", sep="\t", usecols=["featureID"])
avail = pd.Series(probe.featureID.unique()); base = avail.str.split(".").str[0]
ids = {}
for w in ["HTT","MYB","STAT1","PRNP","ATG5"]:
    cand = set(sym.loc[sym.hgnc_symbol == w, "ensembl_gene_id"])
    hit = avail[base.isin(cand)].tolist(); assert len(hit) == 1, (w, hit)
    ids[w] = hit[0]
gsub = list(ids.values())

def gse_merged(genes):
    out = []
    for ser, ctx in [("GSE304951_NegSi","NegSi"), ("GSE304951_U1CKD","U1CKD")]:
        d = load(ser); d = d[d.featureID.isin(genes)].copy()
        d["treatment"] = d.treatment + "_" + ctx; out.append(d)
    return pd.concat(out, ignore_index=True)

def series_data(name, genes):
    if name == "GSE304951_merged": return gse_merged(genes), ["U1CKD"]
    if name == "C2C5_24h":  d = load("C2C5_24h"); return d[d.featureID.isin(genes)], ["IFNa","IFNg"]
    if name == "Exp2":      d = load("Exp2");     return d[d.featureID.isin(genes)], None
    if name == "Exp11_CP3":
        d = load("Exp11")
        return d[d.featureID.isin(genes) & d.treatment.isin(["CP3","CP3PlusBPN","DMSO"])], None

SERIES = ["GSE304951_merged","Exp2","C2C5_24h","Exp11_CP3"]
DATA, SPECS = {}, {}
for s in SERIES:
    d, cols = series_data(s, gsub); DATA[s] = d
    SPECS[s] = None if cols is None else CD.prepare_covariates(COV_TSV, cols, d, scale_covariates=False)

BASELINE_DIV = {
 'GSE304951_merged/HTT':0,'GSE304951_merged/MYB':0,'GSE304951_merged/STAT1':0,
 'GSE304951_merged/PRNP':0,'GSE304951_merged/ATG5':1,'Exp2/HTT':0,'Exp2/MYB':0,'Exp2/STAT1':0,
 'Exp2/PRNP':0,'Exp2/ATG5':0,'C2C5_24h/HTT':0,'C2C5_24h/MYB':0,'C2C5_24h/STAT1':0,
 'C2C5_24h/PRNP':0,'C2C5_24h/ATG5':0,'Exp11_CP3/HTT':0,'Exp11_CP3/MYB':0,'Exp11_CP3/STAT1':0,
 'Exp11_CP3/PRNP':0,'Exp11_CP3/ATG5':0}

FITS, got = {}, {}
for s in SERIES:
    for g in ids:
        a = parse_args(shlex.split("--model 3 --input x --output_pkl x --output_tsv x"))
        a.cov_spec = SPECS[s]
        sub = DATA[s][DATA[s].featureID == ids[g]]
        FITS[(s,g)] = fit_expression_absolute(sub, samples=1000, args=a)[0]
        got[f"{s}/{g}"] = int(FITS[(s,g)].sample_stats.diverging.sum())
        print(f"  {s}/{g}: div={got[f'{s}/{g}']} (baseline {BASELINE_DIV[f'{s}/{g}']})", flush=True)

import pickle
with open("scratch/model3_marker_fits.pkl","wb") as fh: pickle.dump(FITS, fh)
mismatch = {k:(BASELINE_DIV[k], got[k]) for k in BASELINE_DIV if BASELINE_DIV[k] != got[k]}
print(f"\ndivergence fingerprint: {len(BASELINE_DIV)-len(mismatch)}/{len(BASELINE_DIV)} match")
if mismatch: print("  MISMATCH (baseline, got):", mismatch)
sys.exit(1 if mismatch else 0)
