#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

def parse_args():
    p = argparse.ArgumentParser(description="Dose-response specificity test over posterior differences")
    p.add_argument("--infile", required=True, help="Pickle file with dict: gene -> idata")
    p.add_argument("--outfile", required=True, help="Output TSV path")
    p.add_argument("--posterior_param", default="logEC50", help="Posterior parameter name to compare (default: logEC50)")
    return p.parse_args()

def get_assayed_ranges(idata):
    log10_dose = idata.constant_data['log10_dose'].values
    trt_idx = idata.constant_data['treatment_idx'].values
    trt_names = idata.posterior[posterior_param].coords["treatment"].values
    ranges = {}
    for i, t in enumerate(trt_names):
        d = 10**log10_dose[trt_idx == i]
        ranges[t] = (np.min(d), np.max(d)) if len(d) else (np.nan, np.nan)
    return trt_names, ranges

def get_assayed_doses(idata, posterior_param):
    log10_dose = idata.constant_data['log10_dose'].values
    trt_idx = idata.constant_data['treatment_idx'].values
    trt_names = idata.posterior[posterior_param].coords['treatment'].values
    doses = {}
    for i, t in enumerate(trt_names):
        d = 10**log10_dose[trt_idx == i]
        # sort unique, format compactly
        d_unique = np.unique(d)
        doses[t] = d_unique
    return doses

def posterior_samples(idata):
    arr = idata.posterior[posterior_param]  # dims: chain x draw x treatment
    samples = arr.stack(sample=("chain","draw")).transpose("sample","treatment").values  # [S, T]
    trt_names = arr.coords['treatment'].values
    return trt_names, samples

def posterior_mean(idata):
    arr = idata.posterior[posterior_param].mean(dim=("chain","draw")).values  # [T]
    trt_names = idata.posterior[posterior_param].coords['treatment'].values
    return trt_names, arr

def bh_qvalues(p):
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.zeros(n)
    prev = 1.0
    for i in range(n-1, -1, -1):
        q[order[i]] = prev = min(prev, ranked[i] * n / (i+1))
    return q

def main():
    global posterior_param
    args = parse_args()
    posterior_param = args.posterior_param

    with open(args.infile, "rb") as f:
        all_dict = pickle.load(f)

    # Identify treatments
    Treatments = list(next(iter(all_dict.values())).posterior[posterior_param].coords['treatment'].values)
    pairs = list(combinations(Treatments, 2))

    # Collect dose ranges and posterior means per gene
    gene_info = {}
    for gene, idata in all_dict.items():
        trt_names, ranges = get_assayed_ranges(idata)
        trt_names_m, post_mean = posterior_mean(idata)
        gene_info[gene] = {
            'ranges': {t: ranges[t] for t in trt_names},
            'mean': {t: float(post_mean[np.where(trt_names_m == t)[0][0]]) for t in trt_names}
        }

    # Filter genes per pair by assayed range: BOTH treatments must have EC50 mean within their
    # respective assayed range. Using "at least one" inflates mu_gw because junctions where only
    # the more-potent drug responds (the other is extrapolated far out of range) contribute large
    # differences that bias the genome-wide null away from the true head-to-head potency ratio.
    gw_records = []
    for gene, info in gene_info.items():
        for (t1, t2) in pairs:
            m1 = info['mean'][t1]
            m2 = info['mean'][t2]
            e1 = 10**m1
            e2 = 10**m2
            r1 = info['ranges'][t1]
            r2 = info['ranges'][t2]
            in1 = (np.isfinite(e1) and np.isfinite(r1[0]) and r1[0] <= e1 <= r1[1])
            in2 = (np.isfinite(e2) and np.isfinite(r2[0]) and r2[0] <= e2 <= r2[1])
            if in1 and in2:
                gw_records.append({'gene': gene, 'pair': (t1, t2), 'diff_mean': m1 - m2})

    df_pairs = pd.DataFrame(gw_records)
    # Genome-wide mean difference per pair
    gw_means = df_pairs.groupby('pair')['diff_mean'].mean().to_dict() if not df_pairs.empty else {}

    # Gene-level posterior differences, centered by genome-wide mean; compute FSP and q-values
    def posterior_diff(idata, t1, t2):
        trt_names, samples = posterior_samples(idata)  # [S,T]
        i1 = np.where(trt_names == t1)[0][0]
        i2 = np.where(trt_names == t2)[0][0]
        return samples[:, i1] - samples[:, i2]  # [S]

    out_rows = []
    for gene, idata in all_dict.items():
        # collect full dose lists per treatment for this gene
        dose_lists = get_assayed_doses(idata, posterior_param)
        for (t1, t2) in pairs:
            if (t1, t2) not in gw_means:
                continue
            mu_gw = gw_means[(t1, t2)]
            diffs = posterior_diff(idata, t1, t2)
            centered = diffs - mu_gw
            # Uncentered posterior diff mean
            posterior_diff_mean = np.nanmean(diffs)
            # Centered posterior diff mean
            posterior_diff_mean_centered = np.nanmean(centered)
            if not np.isfinite(posterior_diff_mean_centered):
                continue
            # False sign probability (one-sided in [0,0.5]); convert to two-sided p-like
            if posterior_diff_mean_centered > 0:
                fsp = np.mean(centered <= 0)
            elif posterior_diff_mean_centered < 0:
                fsp = np.mean(centered >= 0)
            else:
                fsp = 0.5
            p_like = min(1.0, 2.0 * fsp)

            # Uncentered per-treatment posterior means
            t1_uncentered_mean = gene_info[gene]['mean'].get(t1, np.nan)
            t2_uncentered_mean = gene_info[gene]['mean'].get(t2, np.nan)

            # Add assayed ranges (per gene, per treatment in the pair)
            r1 = gene_info[gene]['ranges'].get(t1, (np.nan, np.nan))
            r2 = gene_info[gene]['ranges'].get(t2, (np.nan, np.nan))
            assayed_ranges = f"{t1}:{r1[0]:.6g}-{r1[1]:.6g},{t2}:{r2[0]:.6g}-{r2[1]:.6g}"

            # Format full assayed dose lists
            d1 = ";".join(f"{x:.6g}" for x in dose_lists.get(t1, np.array([])))
            d2 = ";".join(f"{x:.6g}" for x in dose_lists.get(t2, np.array([])))

            out_rows.append({
                'gene': gene,
                'pair': f"{t1}|{t2}",
                't1_name': t1,
                't2_name': t2,
                't1_uncentered_posterior_mean': float(t1_uncentered_mean) if np.isfinite(t1_uncentered_mean) else np.nan,
                't2_uncentered_posterior_mean': float(t2_uncentered_mean) if np.isfinite(t2_uncentered_mean) else np.nan,
                'posterior_diff_mean': float(posterior_diff_mean) if np.isfinite(posterior_diff_mean) else np.nan,
                'posterior_diff_mean_centered': float(posterior_diff_mean_centered),
                'gw_mean': mu_gw,
                'FSP': float(fsp),
                'p_like': float(p_like),
                't1_assayed_doses': d1,
                't2_assayed_doses': d2,
            })

    res = pd.DataFrame(out_rows)

    # BH q-values per pair on p_like
    if not res.empty:
        res['qvalue'] = np.nan
        for pair in res['pair'].unique():
            mask = res['pair'] == pair
            res.loc[mask, 'qvalue'] = bh_qvalues(res.loc[mask, 'p_like'].values)

    # Write output
    res.sort_values(['pair','qvalue','FSP'], inplace=True)
    res.to_csv(args.outfile, sep='\t', index=False)

if __name__ == "__main__":
    main()