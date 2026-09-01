"""Assemble the fitter's long table directly from feature-by-sample matrices.

The long-format path materialises the same numbers three times (matrix -> tidy -> batches).
Here a job reads only the rows of its chunk and only the columns of its series, so nothing
intermediate is written at all.
"""
import bisect
import gzip
from pathlib import Path

import pandas as pd

__all__ = [
    "MatrixLayoutError", "resolve_columns", "count_features", "chunk_bounds",
    "read_matrix_chunk", "build_feature_index", "read_feature_index", "assemble_long",
]

# Tried in order when the feature column is not named `featureID`: BED col 4 is `name`
# generically and `junc` in the leafcutter junction tables.
FEATURE_COL_CANDIDATES = ("featureID", "name", "junc")

INDEX_SUFFIX = ".fidx"


class MatrixLayoutError(ValueError):
    """The matrix does not have the columns this design needs."""


def _open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_header(path):
    with _open_text(path) as h:
        line = h.readline().rstrip("\n")
    return line.lstrip("#").split("\t")


def resolve_columns(header, design_samples, feature_col=None):
    """Split a header into (feature column, sample columns), for either accepted layout.

    The design's sample list is what distinguishes the two: any header column naming a sample
    is data, and whatever leads is metadata. A BED6+ file and a bare featureID-plus-samples
    matrix therefore need no layout flag and no genome coordinates.
    """
    wanted = list(dict.fromkeys(design_samples))
    present = [c for c in header if c in set(wanted)]
    missing = [s for s in wanted if s not in set(header)]
    if missing:
        raise MatrixLayoutError(
            f"{len(missing)} sample(s) in the design are not columns of the matrix: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )

    meta = [c for c in header if c not in set(present)]
    if feature_col is not None:
        if feature_col not in header:
            raise MatrixLayoutError(f"--feature_col {feature_col!r} is not a column: {header[:8]}")
        return feature_col, present

    for cand in FEATURE_COL_CANDIDATES:
        if cand in meta:
            return cand, present
    if not meta:
        raise MatrixLayoutError(
            "Every column of the matrix is a sample, so there is no feature column. Expected one "
            f"of {FEATURE_COL_CANDIDATES} or --feature_col."
        )
    return meta[0], present


def count_features(path):
    n = 0
    with _open_text(path) as h:
        h.readline()
        for _ in h:
            n += 1
    return n


def chunk_bounds(total, n, m):
    """Row range of chunk `n` when `total` rows are split into `m` contiguous chunks.

    `n` is 1-based; chunk 0 is the header and carries no rows. The remainder is spread one row
    each over the leading chunks, so every feature lands in exactly one chunk and concatenating
    chunks 1..m reproduces the whole file.
    """
    if m < 1:
        raise ValueError(f"number of chunks must be >= 1, got {m}")
    if not 0 <= n <= m:
        raise ValueError(f"chunk index must be in 0..{m}, got {n}")
    if n == 0:
        return 0, 0
    base, rem = divmod(total, m)
    start = (n - 1) * base + min(n - 1, rem)
    count = base + (1 if n <= rem else 0)
    return start, count


def build_feature_index(path, feature_col=None, every=1000):
    """Write the sidecar mapping every `every`-th feature to its BGZF virtual offset."""
    from Bio import bgzf

    header = read_header(path)
    col = feature_col or next((c for c in FEATURE_COL_CANDIDATES if c in header), header[0])
    fi = header.index(col)

    entries = []
    with bgzf.open(str(path), "rt") as h:
        h.readline()
        i = 0
        while True:
            offset = h.tell()
            line = h.readline()
            if not line:
                break
            if i % every == 0:
                entries.append((line.split("\t")[fi], i, offset))
            i += 1

    out = Path(str(path) + INDEX_SUFFIX)
    with open(out, "w") as fh:
        fh.write(f"#featureID\trank\tvirtual_offset\tevery={every}\ttotal={i}\n")
        for fid, rank, offset in entries:
            fh.write(f"{fid}\t{rank}\t{offset}\n")
    return out


def read_feature_index(path):
    idx_path = Path(str(path) + INDEX_SUFFIX)
    if not idx_path.exists():
        raise MatrixLayoutError(
            f"--sorted was given but {idx_path} does not exist. Build it with build_feature_index, "
            "or drop --sorted to read the matrix into memory instead."
        )
    ranks, offsets, ids = [], [], []
    total = None
    with open(idx_path) as fh:
        for line in fh:
            if line.startswith("#"):
                for field in line.rstrip("\n").split("\t"):
                    if field.startswith("total="):
                        total = int(field.split("=", 1)[1])
                continue
            fid, rank, offset = line.rstrip("\n").split("\t")
            ids.append(fid)
            ranks.append(int(rank))
            offsets.append(int(offset))
    return {"ids": ids, "ranks": ranks, "offsets": offsets, "total": total}


def _read_chunk_sorted(path, feature_col, samples, start, count, header):
    from Bio import bgzf

    index = read_feature_index(path)
    pos = bisect.bisect_right(index["ranks"], start) - 1
    if pos < 0:
        raise MatrixLayoutError(f"{path}{INDEX_SUFFIX} has no entry at or before row {start}")

    skip = start - index["ranks"][pos]
    rows = []
    with bgzf.open(str(path), "rt") as h:
        h.seek(index["offsets"][pos])
        for _ in range(skip):
            h.readline()
        for _ in range(count):
            line = h.readline()
            if not line:
                break
            rows.append(line.rstrip("\n").split("\t"))

    df = pd.DataFrame(rows, columns=header)
    if skip == 0 and len(df) and df[feature_col].iloc[0] != index["ids"][pos]:
        raise MatrixLayoutError(
            f"{path}{INDEX_SUFFIX} is stale: row {start} is {df[feature_col].iloc[0]!r} but the "
            f"index says {index['ids'][pos]!r}. Rebuild the index."
        )
    return df[[feature_col] + samples]


def read_matrix_chunk(path, feature_col, samples, start, count, sorted_access=False):
    """Rows [start, start+count) of `path`, restricted to `samples`.

    Without `sorted_access` the matrix is read and sliced -- correct whatever its order. With it,
    the sidecar index is used to seek, which is much faster but assumes the file is sorted by
    feature and that the index matches it (both checked).
    """
    header = read_header(path)
    if sorted_access:
        df = _read_chunk_sorted(path, feature_col, samples, start, count, header)
    else:
        df = pd.read_csv(path, sep="\t", usecols=[feature_col] + samples)
        df = df.iloc[start:start + count]
    for c in samples:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def assemble_long(matrices, design, start, count, sorted_access=False, feature_col=None):
    """Build the fitter's long table from {outcome: matrix path} plus a sample-level design.

    `design` has columns sample, treatment, dose. The result carries exactly the columns the
    long-format path produced, in the same row order, so the fit functions cannot tell the
    difference.
    """
    samples = design["sample"].tolist()
    frames = {}
    resolved_feature_col = None
    for outcome, path in matrices.items():
        col, present = resolve_columns(read_header(path), samples, feature_col)
        if resolved_feature_col is None:
            resolved_feature_col = col
        wide = read_matrix_chunk(path, col, present, start, count, sorted_access)
        long = wide.melt(id_vars=[col], var_name="sample", value_name=outcome)
        long = long.rename(columns={col: "featureID"})
        frames[outcome] = long

    outcomes = list(frames)
    out = frames[outcomes[0]]
    for outcome in outcomes[1:]:
        other = frames[outcome]
        # Row-for-row agreement between the y and n matrices is an assumption worth failing on:
        # a silent misalignment would attach real posteriors to the wrong feature.
        if set(out["featureID"].unique()) != set(other["featureID"].unique()):
            raise MatrixLayoutError(
                f"matrices for {outcomes[0]!r} and {outcome!r} cover different features in rows "
                f"[{start}, {start + count}). They must be sorted identically."
            )
        out = out.merge(other, on=["featureID", "sample"], how="inner")

    out = out.merge(design[["sample", "treatment", "dose"]], on="sample", how="left")
    if out["treatment"].isna().any():
        bad = out.loc[out["treatment"].isna(), "sample"].unique()[:5]
        raise MatrixLayoutError(f"samples missing from the design after the join: {list(bad)}")

    # Matches arrange(featureID, desc(IsControl), treatment, dose) in SeparateTidyDataIntoBatches.R;
    # the fit functions and covariate design rely on this within-feature ordering.
    out["_is_control"] = out["dose"] == 0
    out = out.sort_values(["featureID", "_is_control", "treatment", "dose"],
                          ascending=[True, False, True, True], kind="mergesort")
    return out.drop(columns="_is_control").reset_index(drop=True)
