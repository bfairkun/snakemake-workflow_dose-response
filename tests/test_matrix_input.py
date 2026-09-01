"""Matrix input: layout detection, chunk algebra, and equivalence with the long path."""
import gzip

import numpy as np
import pandas as pd
import pytest

from dose_response.matrix_input import (
    MatrixLayoutError, assemble_long, chunk_bounds, count_features, resolve_columns,
)


@pytest.fixture
def design():
    return pd.DataFrame({
        "sample": ["c1", "c2", "t_lo", "t_hi", "u_lo", "u_hi"],
        "treatment": ["DMSO", "DMSO", "drugA", "drugA", "drugB", "drugB"],
        "dose": [0.0, 0.0, 10.0, 100.0, 10.0, 100.0],
    })


def _write(tmp_path, name, rows, header, gz=True):
    path = tmp_path / name
    text = "\t".join(header) + "\n" + "\n".join("\t".join(map(str, r)) for r in rows) + "\n"
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return path


@pytest.fixture
def bed6(tmp_path, design):
    """BED6+ layout: six leading columns, feature id in column 4."""
    header = ["#chrom", "start", "end", "name", "score", "strand"] + design["sample"].tolist()
    rows = [["chr1", i * 100, i * 100 + 50, f"F{i:03d}", 0, "+"] + [i + j for j in range(6)]
            for i in range(20)]
    return _write(tmp_path, "m.bed.gz", rows, header)


@pytest.fixture
def bare(tmp_path, design):
    """Bare layout: featureID plus one column per sample, no coordinates."""
    header = ["featureID"] + design["sample"].tolist()
    rows = [[f"F{i:03d}"] + [i + j for j in range(6)] for i in range(20)]
    return _write(tmp_path, "m.tsv.gz", rows, header)


# --- layout resolution -------------------------------------------------------------------

def test_bed6_layout_resolves(bed6, design):
    col, samples = resolve_columns(
        ["#chrom", "start", "end", "name", "score", "strand"] + design["sample"].tolist(),
        design["sample"].tolist())
    assert col == "name"
    assert samples == design["sample"].tolist()


def test_bare_layout_resolves(design):
    col, samples = resolve_columns(["featureID"] + design["sample"].tolist(),
                                   design["sample"].tolist())
    assert col == "featureID"
    assert samples == design["sample"].tolist()


def test_missing_sample_is_fatal(design):
    with pytest.raises(MatrixLayoutError, match="not columns of the matrix"):
        resolve_columns(["featureID", "c1"], design["sample"].tolist())


def test_both_layouts_give_identical_output(bed6, bare, design):
    """The whole point of inferring layout: coordinates change nothing."""
    a = assemble_long({"y": bed6}, design, 0, 20)
    b = assemble_long({"y": bare}, design, 0, 20)
    pd.testing.assert_frame_equal(a, b)


# --- chunk algebra -----------------------------------------------------------------------

@pytest.mark.parametrize("total,m", [(20, 4), (530145, 200), (7, 3), (5, 5), (3, 10)])
def test_chunks_partition_exactly(total, m):
    covered = []
    for n in range(1, m + 1):
        start, count = chunk_bounds(total, n, m)
        covered.extend(range(start, start + count))
    assert covered == list(range(total)), "every row exactly once, in order"


def test_chunk_zero_is_empty():
    assert chunk_bounds(100, 0, 10) == (0, 0)


def test_remainder_goes_to_leading_chunks():
    sizes = [chunk_bounds(10, n, 3)[1] for n in range(1, 4)]
    assert sizes == [4, 3, 3]


def test_out_of_range_chunk_rejected():
    with pytest.raises(ValueError):
        chunk_bounds(10, 4, 3)


def test_concatenating_chunks_reproduces_whole(bed6, design):
    whole = assemble_long({"y": bed6}, design, 0, 20)
    parts = []
    for n in range(1, 5):
        start, count = chunk_bounds(20, n, 4)
        parts.append(assemble_long({"y": bed6}, design, start, count))
    got = pd.concat(parts, ignore_index=True)
    pd.testing.assert_frame_equal(got, whole)


def test_count_features(bed6):
    assert count_features(bed6) == 20


# --- assembly ----------------------------------------------------------------------------

def test_controls_come_first_within_a_feature(bed6, design):
    """Row order must match arrange(featureID, desc(IsControl), treatment, dose)."""
    out = assemble_long({"y": bed6}, design, 0, 3)
    for _, g in out.groupby("featureID", sort=False):
        is_ctrl = (g["dose"] == 0).tolist()
        assert is_ctrl == sorted(is_ctrl, reverse=True), "controls must lead"
        treated = g[g["dose"] > 0]
        assert treated["treatment"].tolist() == sorted(treated["treatment"].tolist())


def test_two_matrices_merge_on_feature_and_sample(tmp_path, bed6, design):
    header = ["featureID"] + design["sample"].tolist()
    rows = [[f"F{i:03d}"] + [100] * 6 for i in range(20)]
    denom = _write(tmp_path, "n.tsv.gz", rows, header)
    out = assemble_long({"y": bed6, "n": denom}, design, 0, 20)
    assert {"y", "n"} <= set(out.columns)
    assert (out["n"] == 100).all()


def test_misaligned_matrices_are_fatal(tmp_path, bed6, design):
    """A silent misalignment would attach posteriors to the wrong feature."""
    header = ["featureID"] + design["sample"].tolist()
    rows = [[f"OTHER{i:03d}"] + [100] * 6 for i in range(20)]
    denom = _write(tmp_path, "bad.tsv.gz", rows, header)
    with pytest.raises(MatrixLayoutError, match="different features"):
        assemble_long({"y": bed6, "n": denom}, design, 0, 20)


def test_sample_absent_from_design_is_fatal(bed6, design):
    extra = pd.concat([design, pd.DataFrame({"sample": ["ghost"], "treatment": ["x"],
                                             "dose": [1.0]})], ignore_index=True)
    with pytest.raises(MatrixLayoutError, match="not columns of the matrix"):
        assemble_long({"y": bed6}, extra, 0, 5)
