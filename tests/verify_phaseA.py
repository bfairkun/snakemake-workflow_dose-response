"""Phase A check: does the packaged code reproduce pre-split fits bit-for-bit?

Baseline = scratch/model2_marker_fits/*.nc, the 18 model-2 fits produced earlier today by
the monolithic scripts/BayesianDoseResponse_ByBatch.py (with covariates). Refit the same
features through the package and require every posterior array to match exactly.

NOTE: the baselines this compares against predate Phase B, which deliberately changed the
splicing parameterizations. Differences are now expected; the script is kept because it is
the only harness that reads those artefacts, and because it still catches an accidental
change to the expression models.
"""
import functools, os, shlex, sys, warnings, logging
import numpy as np, pandas as pd, pymc as pm, arviz as az
warnings.filterwarnings("ignore"); logging.getLogger("pymc").setLevel(logging.ERROR)
os.chdir("/project/yangili1/bjf79/20260310_diversesm_dr/code")
sys.path.insert(0, "module_workflows/dose_response/src")

from dose_response.covariates import prepare_covariates
from dose_response.filters import check_prefilter_by_number
from dose_response.fitting import MODEL_CONFIG
from dose_response.cli.fit_batch import parse_args
import dose_response.models.splicing_psi as sp
_o = pm.sample
@functools.wraps(_o)
def _q(*a, **k):
    k.setdefault("progressbar", False); return _o(*a, **k)
pm.sample = sp.pm.sample = _q

BASE = "scratch/model2_marker_fits"
J = pd.read_csv(f"{BASE}/marker_junction_rows.tsv.gz", sep="\t")
JCOV = {"GSE304951_merged":["U1CKD"], "C2C5_24h":["IFNa","IFNg"], "Exp2":None, "Exp11_CP3":None}
PF = [("n",5,10,1e5), ("y",3,3,1e5)]
ok = bad = 0; arrays = 0
for s in ["GSE304951_merged","Exp2","C2C5_24h","Exp11_CP3"]:
    sub_s = J[J.ser == s]
    spec = None if JCOV[s] is None else prepare_covariates(
        "config/dose_response_covariates.tsv", JCOV[s], sub_s, scale_covariates=False)
    for g in ["HTT","STAT1","MYB","ATG5","PRNP"]:
        nc = f"{BASE}/{s}__{g}.nc"
        if not os.path.exists(nc): continue
        sub = sub_s[sub_s.gene == g]
        assert check_prefilter_by_number(sub, PF)[0]
        a = parse_args(shlex.split("--model 2 --input x --output_pkl x --output_tsv x"))
        a.cov_spec = spec
        new = MODEL_CONFIG[2]["fit_func"](sub, samples=1000, args=a)[0]
        old = az.from_netcdf(nc)
        diffs = []
        for v in old.posterior.data_vars:
            if v not in new.posterior: diffs.append(f"{v}: MISSING"); continue
            A, B = np.asarray(old.posterior[v]), np.asarray(new.posterior[v])
            arrays += 1
            if A.shape != B.shape: diffs.append(f"{v}: shape {A.shape} vs {B.shape}")
            elif not np.array_equal(A, B, equal_nan=True):
                diffs.append(f"{v}: max|diff|={np.nanmax(np.abs(A-B)):.3e}")
        extra = set(new.posterior.data_vars) - set(old.posterior.data_vars)
        if extra: diffs.append(f"extra vars: {sorted(extra)}")
        if diffs:
            bad += 1; print(f"  DIFF {s}/{g}: " + "; ".join(diffs), flush=True)
        else:
            ok += 1; print(f"  ok   {s}/{g}  ({len(old.posterior.data_vars)} posterior arrays identical)", flush=True)
print(f"\n{ok} identical, {bad} differing, {arrays} posterior arrays compared")
sys.exit(1 if bad else 0)
