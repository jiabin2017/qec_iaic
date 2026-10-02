from __future__ import annotations
import argparse, csv, json, sqlite3
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Summarize p5 numerical cleanup replay.sqlite")
    ap.add_argument("db", nargs="?", default="p5_numerical_cleanup_replay/replay.sqlite")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    dbp = Path(args.db).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else dbp.parent
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(dbp)
    cols = [r[1] for r in con.execute("PRAGMA table_info(replay)")]
    rows = con.execute("SELECT classification,COUNT(*) FROM replay GROUP BY classification ORDER BY COUNT(*) DESC").fetchall()
    print("total", con.execute("SELECT COUNT(*) FROM replay").fetchone()[0])
    for k,n in rows: print(f"{k}: {n}")
    print("gap >1e-10", con.execute("SELECT COUNT(*) FROM replay WHERE gap_ref_vs_replay_b9_prob>1e-10").fetchone()[0])
    print("reference stressed", con.execute("SELECT COUNT(*) FROM replay WHERE reference_stressed=1").fetchone()[0])
    print("ref prefix not explored", con.execute("SELECT COUNT(*) FROM replay WHERE ref_prefix_explored=0").fetchone()[0])
    q = "SELECT * FROM replay ORDER BY gap_ref_vs_replay_b9_prob DESC, reference_stressed DESC LIMIT 100"
    cur = con.execute(q)
    with (out/"replay_priority_top100.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow([d[0] for d in cur.description]); w.writerows(cur.fetchall())
    summary = {
        "total": con.execute("SELECT COUNT(*) FROM replay").fetchone()[0],
        "by_classification": {k:n for k,n in rows},
        "gap_gt_1e-10": con.execute("SELECT COUNT(*) FROM replay WHERE gap_ref_vs_replay_b9_prob>1e-10").fetchone()[0],
        "reference_stressed": con.execute("SELECT COUNT(*) FROM replay WHERE reference_stressed=1").fetchone()[0],
        "ref_prefix_not_explored": con.execute("SELECT COUNT(*) FROM replay WHERE ref_prefix_explored=0").fetchone()[0],
    }
    (out/"replay_summary_compact.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")

if __name__ == "__main__": main()
