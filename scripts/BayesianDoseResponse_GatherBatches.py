import argparse
import pandas as pd
import pickle
import sys
import os

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Combine batch output TSVs and pickle files.")
    parser.add_argument('--tsvs', nargs='+', required=True, help="Input .tsv.gz files to concatenate")
    parser.add_argument('--pkls', nargs='+', required=True, help="Input pickle files to merge")
    parser.add_argument('--output_tsv', required=True, help="Output concatenated TSV file (can be .gz)")
    parser.add_argument('--output_pkl', required=True, help="Output merged pickle file")
    return parser.parse_args(args)

def combine_tsvs(tsv_files, output_tsv):
    dfs = []
    for f in tsv_files:
        print(f"Reading {f}")
        dfs.append(pd.read_csv(f, sep="\t"))
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Writing combined TSV to {output_tsv}")
    combined.to_csv(output_tsv, sep="\t", index=False, compression="infer")

def combine_pkls(pkl_files, output_pkl):
    combined = {}
    for f in pkl_files:
        print(f"Reading {f}")
        with open(f, "rb") as pf:
            d = pickle.load(pf)
            overlap = set(combined).intersection(d)
            if overlap:
                print(f"Warning: overlapping featureIDs in {f}: {overlap}")
            combined.update(d)
    print(f"Writing merged pickle to {output_pkl}")
    with open(output_pkl, "wb") as pf:
        pickle.dump(combined, pf)

def main(args=None):
    args = parse_args(args)
    combine_tsvs(args.tsvs, args.output_tsv)
    combine_pkls(args.pkls, args.output_pkl)



if __name__ == "__main__":
    if hasattr(sys, 'ps1'):
        # Example test usage
        # Example: open and print the merged pickle file
        main("--tsvs ExperimentGeneralizedDoseResponseModelling/Expression/ResultsBatched/Exp11/0.tsv.gz ExperimentGeneralizedDoseResponseModelling/Expression/ResultsBatched/Exp11/1.tsv.gz ExperimentGeneralizedDoseResponseModelling/Expression/ResultsBatched/Exp11/2.tsv.gz --pkls ExperimentGeneralizedDoseResponseModelling/Expression/ResultsBatched/Exp11/0.pkl ExperimentGeneralizedDoseResponseModelling/Expression/ResultsBatched/Exp11/1.pkl ExperimentGeneralizedDoseResponseModelling/Expression/ResultsBatched/Exp11/2.pkl --output_tsv scratch/test.combined.tsv.gz --output_pkl scratch/test.combined.pkl".split())
        # Open and print the merged pickle file
        with open("scratch/test.combined.pkl", "rb") as pf:
            merged_data = pickle.load(pf)
            print("Merged pickle keys:", list(merged_data.keys()))
    else:
        main()

