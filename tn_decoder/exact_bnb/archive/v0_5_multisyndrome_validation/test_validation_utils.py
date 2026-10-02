import sqlite3
from pathlib import Path
import numpy as np
from validation_utils import (
    balanced_weight_schedule, build_natural_dataset, build_stress_dataset,
    open_results_db, resolve_layouts,
)


def test_balanced_weight_schedule():
    a = balanced_weight_schedule(10, [2,3,4,5])
    assert a.tolist() == [2,3,4,5,2,3,4,5,2,3]


def test_datasets_reproducible():
    H = np.array([[1,0,1,0],[0,1,0,1]], dtype=np.uint8)
    L = np.array([[1,1,0,0]], dtype=np.uint8)
    a = build_natural_dataset(H=H, LZ=L, p=0.2, shots=20, seed=7)
    b = build_natural_dataset(H=H, LZ=L, p=0.2, shots=20, seed=7)
    assert np.array_equal(a['errors'], b['errors'])
    s = build_stress_dataset(H=H, LZ=L, shots=8, weights=[1,2], seed=9)
    assert s['error_weights'].tolist() == [1,2,1,2,1,2,1,2]


def test_sqlite_primary_key(tmp_path):
    conn = open_results_db(tmp_path / 'r.sqlite')
    cols = [r[1] for r in conn.execute('PRAGMA table_info(shots)').fetchall()]
    assert 'layout_tag' in cols and 'shot_id' in cols
    conn.close()
