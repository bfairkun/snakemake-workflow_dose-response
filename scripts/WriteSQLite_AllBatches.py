#!/usr/bin/env python3
"""
Write all batch PKL files into a single SQLite database.
"""
import argparse
import os
import re
import pickle
import sqlite3
import time
import sys
import tempfile
import arviz as az  # Load before unpickling InferenceData objects


def _derive_approach_series(pkl_path: str) -> tuple[str, str]:
    """Derive (approach, series) from a PKL path.
    Expected: .../DoseResponseModelling/<Approach>/ResultsBatched/<series>/<n>.pkl
    """
    m = re.search(r"DoseResponseModelling/([^/]+)/ResultsBatched/([^/]+)/\d+\.pkl$", pkl_path)
    if m:
        return m.group(1), m.group(2)
    # Fallback to path parts
    parts = os.path.normpath(pkl_path).split(os.sep)
    try:
        i = parts.index("DoseResponseModelling")
        approach = parts[i+1]
        if parts[i+2] != "ResultsBatched":
            raise ValueError
        series = parts[i+3]
        return approach, series
    except Exception:
        raise ValueError(f"Unexpected PKL path: {pkl_path}")


def main():
    parser = argparse.ArgumentParser(description="Write batch PKLs to SQLite")
    parser.add_argument("--output_db", required=True, help="Output SQLite database path")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of rows to insert per batch")
    parser.add_argument("pkl_files", nargs="+", help="Input PKL files")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_db), exist_ok=True)
    conn = sqlite3.connect(args.output_db)

    try:
        # Speed up bulk inserts
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-20000;")  # ~20k pages cache (negative => KB)

        # Create table storing NetCDF blobs
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idata (
                feature_id TEXT NOT NULL,
                approach   TEXT NOT NULL,
                series     TEXT NOT NULL,
                netcdf     BLOB NOT NULL,
                PRIMARY KEY (feature_id, approach, series)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_idata_feature
            ON idata(feature_id)
            """
        )

        insert_sql = "INSERT OR REPLACE INTO idata (feature_id, approach, series, netcdf) VALUES (?, ?, ?, ?)"

        total_rows = 0
        t0 = time.time()

        with conn:  # transaction scope per entire run (we also commit per batch)
            for pkl_path in args.pkl_files:
                approach, series = _derive_approach_series(pkl_path)

                with open(pkl_path, "rb") as f:
                    d = pickle.load(f)  # dict: feature_id -> InferenceData
                sys.stderr.write(f"Processing {pkl_path} with {len(d)} features (approach={approach}, series={series})\n")

                if not d:
                    continue

                # Process in batches to avoid memory buildup and make rows visible sooner
                batch = []
                for feat, idata in d.items():
                    # Write InferenceData to a temp NetCDF file, then read bytes
                    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        idata.to_netcdf(tmp_path)
                        with open(tmp_path, "rb") as fh:
                            nc_bytes = fh.read()
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except FileNotFoundError:
                            pass

                    batch.append((str(feat), approach, series, sqlite3.Binary(nc_bytes)))

                    # Insert when batch is full
                    if len(batch) >= args.batch_size:
                        conn.executemany(insert_sql, batch)
                        conn.commit()  # make visible immediately
                        sys.stderr.write(f"Inserted {len(batch)} rows (running total before commit: {total_rows})\n")
                        total_rows += len(batch)
                        batch = []

                # Insert remaining rows from this file
                if batch:
                    conn.executemany(insert_sql, batch)
                    conn.commit()
                    total_rows += len(batch)

                sys.stderr.write(f"Processed {len(d)} features from {pkl_path} (total so far: {total_rows})\n")

        msg = f"Wrote {total_rows} rows into {args.output_db} in {time.time()-t0:.1f}s"
        print(msg)
        sys.stderr.write(msg + "\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
