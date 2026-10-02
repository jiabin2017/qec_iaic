from __future__ import annotations
import argparse
from pathlib import Path
from validation_utils import open_results_db, write_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="p5_multisyndrome_validation")
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    conn = open_results_db(out / "results.sqlite")
    write_summaries(conn, out)
    conn.close()
    print("wrote", out / "summary.csv")
    print("wrote", out / "consistency_vs_b12.csv")


if __name__ == "__main__":
    main()
