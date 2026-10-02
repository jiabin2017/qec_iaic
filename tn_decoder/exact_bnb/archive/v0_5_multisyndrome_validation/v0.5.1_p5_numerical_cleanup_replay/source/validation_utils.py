from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_P_VALUES = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
DEFAULT_STRESS_WEIGHTS = (2, 3, 4, 5)

VALIDATION6_TAGS = (
    "stageA-b9-identity",
    "stageA-b10-identity",
    "stageA-b6-identity",
    "stageA-b12-identity",
    "stageB-b9-random-1",
    "stageB-b9-reverse",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    p: float
    shots: int
    path: Path


def p_label(p: float) -> str:
    return f"{float(p):.6g}".replace(".", "p")


def parse_float_list(text: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(text, str):
        vals = [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    else:
        vals = [float(x) for x in text]
    if not vals:
        raise ValueError("empty float list")
    return tuple(vals)


def parse_int_list(text: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(text, str):
        vals = [int(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    else:
        vals = [int(x) for x in text]
    if not vals:
        raise ValueError("empty integer list")
    return tuple(vals)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = json.dumps([int(base_seed), *parts], sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF


def _gf2_matmul_rows(rows: np.ndarray, matrix_t: np.ndarray) -> np.ndarray:
    # rows: (N, n), matrix_t: (n, m). uint16 is enough because n=72.
    return ((rows.astype(np.uint16) @ matrix_t.astype(np.uint16)) & 1).astype(np.uint8)


def build_natural_dataset(
    *, H: np.ndarray, LZ: np.ndarray, p: float, shots: int, seed: int
) -> dict[str, np.ndarray]:
    n = int(H.shape[1])
    rng = np.random.default_rng(int(seed))
    errors = (rng.random((int(shots), n)) < float(p)).astype(np.uint8)
    syndromes = _gf2_matmul_rows(errors, np.asarray(H, dtype=np.uint8).T)
    logicals = _gf2_matmul_rows(errors, np.asarray(LZ, dtype=np.uint8).T)
    return {
        "errors": errors,
        "syndromes": syndromes,
        "true_lambdas": logicals,
        "error_weights": errors.sum(axis=1).astype(np.uint16),
        "syndrome_weights": syndromes.sum(axis=1).astype(np.uint16),
    }


def balanced_weight_schedule(shots: int, weights: Sequence[int]) -> np.ndarray:
    weights = tuple(int(w) for w in weights)
    if not weights:
        raise ValueError("stress weights must be non-empty")
    out = np.empty(int(shots), dtype=np.uint16)
    for i in range(int(shots)):
        out[i] = weights[i % len(weights)]
    return out


def build_stress_dataset(
    *, H: np.ndarray, LZ: np.ndarray, shots: int, weights: Sequence[int], seed: int
) -> dict[str, np.ndarray]:
    n = int(H.shape[1])
    rng = np.random.default_rng(int(seed))
    schedule = balanced_weight_schedule(int(shots), weights)
    errors = np.zeros((int(shots), n), dtype=np.uint8)
    for i, w in enumerate(schedule):
        w = int(w)
        if w < 0 or w > n:
            raise ValueError(f"invalid stress weight {w} for n={n}")
        if w:
            pos = rng.choice(n, size=w, replace=False)
            errors[i, pos] = 1
    syndromes = _gf2_matmul_rows(errors, np.asarray(H, dtype=np.uint8).T)
    logicals = _gf2_matmul_rows(errors, np.asarray(LZ, dtype=np.uint8).T)
    return {
        "errors": errors,
        "syndromes": syndromes,
        "true_lambdas": logicals,
        "error_weights": errors.sum(axis=1).astype(np.uint16),
        "syndrome_weights": syndromes.sum(axis=1).astype(np.uint16),
    }


def ensure_datasets(
    *,
    output_dir: Path,
    H: np.ndarray,
    LZ: np.ndarray,
    p_values: Sequence[float],
    natural_shots: int,
    stress_shots: int,
    stress_weights: Sequence[int],
    stress_p: float,
    seed: int,
) -> tuple[list[DatasetSpec], dict]:
    output_dir = Path(output_dir)
    ds_dir = output_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ds_dir / "dataset_manifest.json"
    requested = {
        "format_version": 1,
        "base_seed": int(seed),
        "p_values": [float(p) for p in p_values],
        "natural_shots_per_p": int(natural_shots),
        "stress_shots": int(stress_shots),
        "stress_weights": [int(w) for w in stress_weights],
        "stress_p": float(stress_p),
        "n": int(H.shape[1]),
        "num_checks": int(H.shape[0]),
        "k": int(LZ.shape[0]),
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != requested:
            raise RuntimeError(
                "dataset settings differ from existing output directory; use a new --output-dir\n"
                f"existing={old}\nrequested={requested}"
            )
    else:
        manifest_path.write_text(json.dumps(requested, indent=2, sort_keys=True) + "\n")

    specs: list[DatasetSpec] = []
    for p in p_values:
        name = f"natural_p{p_label(p)}"
        path = ds_dir / f"{name}.npz"
        if not path.exists():
            arrays = build_natural_dataset(
                H=H,
                LZ=LZ,
                p=float(p),
                shots=int(natural_shots),
                seed=stable_seed(seed, "natural", float(p), int(natural_shots)),
            )
            np.savez_compressed(path, **arrays)
        specs.append(DatasetSpec(name=name, kind="natural", p=float(p), shots=int(natural_shots), path=path))

    stress_name = "stress_w" + "-".join(str(int(w)) for w in stress_weights) + f"_p{p_label(stress_p)}"
    stress_path = ds_dir / f"{stress_name}.npz"
    if not stress_path.exists():
        arrays = build_stress_dataset(
            H=H,
            LZ=LZ,
            shots=int(stress_shots),
            weights=stress_weights,
            seed=stable_seed(seed, "stress", [int(w) for w in stress_weights], int(stress_shots)),
        )
        np.savez_compressed(stress_path, **arrays)
    specs.append(DatasetSpec(name=stress_name, kind="stress", p=float(stress_p), shots=int(stress_shots), path=stress_path))
    return specs, requested


def load_scan_results(scan_dir: Path) -> dict:
    scan_dir = Path(scan_dir)
    path = scan_dir / "p5_1_scan_results.json"
    if not path.exists():
        raise FileNotFoundError(f"missing P5.1 scan results: {path}")
    return json.loads(path.read_text())


def resolve_layouts(scan_dir: Path, selection: str, explicit_tags: str | None = None) -> list[dict]:
    scan_dir = Path(scan_dir)
    payload = load_scan_results(scan_dir)
    rows = list(payload.get("results", []))
    by_tag = {str(r["tag"]): r for r in rows}

    if explicit_tags:
        tags = [x.strip() for x in explicit_tags.split(",") if x.strip()]
    else:
        key = str(selection).strip().lower()
        if key == "validation6":
            tags = list(VALIDATION6_TAGS)
        elif key in {"all", "all15", "full"}:
            # Preserve the P5.1 result order, which starts with the best configurations.
            tags = [str(r["tag"]) for r in rows]
        elif key in {"stagea", "identity", "identity8"}:
            tags = [str(r["tag"]) for r in rows if str(r["tag"]).startswith("stageA-")]
        elif key in {"stageb", "b9-orders"}:
            tags = [str(r["tag"]) for r in rows if str(r["tag"]).startswith("stageB-")]
        else:
            raise ValueError("unknown --layout-selection; use validation6, all15, stageA, stageB, or --layout-tags")

    missing = [t for t in tags if t not in by_tag]
    if missing:
        raise KeyError(f"layout tags not found in scan results: {missing}")

    out = []
    for tag in tags:
        row = dict(by_tag[tag])
        raw = Path(str(row["profile_path"]))
        candidates = [raw, scan_dir / raw.name, scan_dir.parent / raw]
        profile = next((p for p in candidates if p.exists()), None)
        if profile is None:
            raise FileNotFoundError(f"cannot resolve profile for {tag}: {raw}")
        row["profile_path_resolved"] = str(profile.resolve())
        out.append(row)
    return out


def open_results_db(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shots (
            layout_tag TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            shot_id INTEGER NOT NULL,
            p REAL NOT NULL,
            error_weight INTEGER NOT NULL,
            syndrome_weight INTEGER NOT NULL,
            wall_seconds REAL NOT NULL,
            oracle_seconds REAL NOT NULL,
            oracle_calls INTEGER NOT NULL,
            expanded_internal_nodes INTEGER NOT NULL,
            leaves_evaluated INTEGER NOT NULL,
            pruned_leaves INTEGER NOT NULL,
            best_probability REAL NOT NULL,
            best_lambda TEXT NOT NULL,
            true_lambda TEXT NOT NULL,
            logical_success INTEGER NOT NULL,
            root_mass REAL NOT NULL,
            clipped_negative_entries INTEGER NOT NULL,
            most_negative_raw_mass REAL NOT NULL,
            max_local_guard_abs REAL NOT NULL,
            topology_call_counts TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (layout_tag, dataset_name, shot_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS layout_meta (
            layout_tag TEXT PRIMARY KEY,
            block_size INTEGER NOT NULL,
            logical_order TEXT NOT NULL,
            profile_path TEXT NOT NULL,
            scan_easy_seconds REAL,
            scan_hard_seconds REAL,
            scan_mean_seconds REAL
        )
        """
    )
    conn.commit()
    return conn


def completed_ids(conn: sqlite3.Connection, layout_tag: str, dataset_name: str) -> set[int]:
    cur = conn.execute(
        "SELECT shot_id FROM shots WHERE layout_tag=? AND dataset_name=?",
        (layout_tag, dataset_name),
    )
    return {int(row[0]) for row in cur.fetchall()}


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def write_summaries(conn: sqlite3.Connection, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    rows = conn.execute(
        """
        SELECT layout_tag, dataset_name,
               COUNT(*), AVG(wall_seconds), AVG(oracle_seconds), AVG(oracle_calls),
               AVG(leaves_evaluated), AVG(logical_success)
        FROM shots
        GROUP BY layout_tag, dataset_name
        ORDER BY layout_tag, dataset_name
        """
    ).fetchall()

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layout_tag", "dataset_name", "completed", "mean_wall_s", "median_wall_s",
            "p90_wall_s", "p95_wall_s", "p99_wall_s", "mean_oracle_s", "mean_oracle_calls",
            "mean_leaves", "logical_success_rate", "logical_error_rate"
        ])
        for row in rows:
            layout, dataset = row[0], row[1]
            vals = np.asarray([
                x[0] for x in conn.execute(
                    "SELECT wall_seconds FROM shots WHERE layout_tag=? AND dataset_name=? ORDER BY shot_id",
                    (layout, dataset),
                ).fetchall()
            ], dtype=float)
            success = float(row[7])
            w.writerow([
                layout, dataset, int(row[2]), float(row[3]), float(np.median(vals)),
                percentile(vals, 0.90), percentile(vals, 0.95), percentile(vals, 0.99),
                float(row[4]), float(row[5]), float(row[6]), success, 1.0 - success,
            ])

    # Cross-layout consistency against the b=12 full-logical-tensor control, when present.
    ref_tag = "stageA-b12-identity"
    consistency_path = output_dir / "consistency_vs_b12.csv"
    candidates = [r[0] for r in conn.execute("SELECT DISTINCT layout_tag FROM shots ORDER BY layout_tag").fetchall()]
    with consistency_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout_tag", "overlap_shots", "same_lambda", "same_lambda_fraction", "prob_close_1e-10", "prob_close_fraction"])
        for tag in candidates:
            if tag == ref_tag:
                continue
            cmp_rows = conn.execute(
                """
                SELECT a.best_lambda, a.best_probability, b.best_lambda, b.best_probability
                FROM shots a JOIN shots b
                  ON a.dataset_name=b.dataset_name AND a.shot_id=b.shot_id
                WHERE a.layout_tag=? AND b.layout_tag=?
                """,
                (tag, ref_tag),
            ).fetchall()
            if not cmp_rows:
                continue
            same = sum(1 for a,_,b,_ in cmp_rows if a == b)
            close = sum(1 for _,ap,_,bp in cmp_rows if abs(float(ap)-float(bp)) <= 1e-10)
            n = len(cmp_rows)
            w.writerow([tag, n, same, same/n, close, close/n])


def human_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"
