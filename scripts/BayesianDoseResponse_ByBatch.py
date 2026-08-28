#!/usr/bin/env python3
"""Shim: the implementation lives in src/dose_response/cli/fit_batch.py.

The parent workflow symlinks scripts/ into its own workdir, so __file__ is a symlink and
realpath() is what resolves back into this repo. That makes the package importable with no
install, no conda-env change and no edit to any Snakemake rule.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(os.path.realpath(__file__)).parents[1] / "src"))

from dose_response.cli.fit_batch import main  # noqa: E402

if __name__ == "__main__":
    main()
