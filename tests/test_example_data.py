import numpy as np
import pandas as pd

from slimnine.example_data import generate_wafer_dataset

EXPECTED_COLS = [
    "lot_id",
    "wafer_id",
    "wafer_no",
    "x_test",
    "y_test",
    "row",
    "col",
    "die_id",
    "radius",
    "vt",
    "idsat",
    "ileak",
    "freq",
    "pass",
]


def test_columns_and_dtypes() -> None:
    df = generate_wafer_dataset(n_lots=2, wafers_per_lot=2, seed=0)
    assert list(df.columns) == EXPECTED_COLS
    assert df["vt"].dtype == np.float32
    assert df["pass"].dtype == np.bool_
    assert isinstance(df["lot_id"].dtype, pd.CategoricalDtype)
    assert isinstance(df["wafer_id"].dtype, pd.CategoricalDtype)


def test_cardinality_matches_inputs() -> None:
    n_lots, wafers_per_lot = 3, 4
    df = generate_wafer_dataset(n_lots=n_lots, wafers_per_lot=wafers_per_lot, seed=0)
    assert df["lot_id"].nunique() == n_lots
    assert df["wafer_id"].nunique() == n_lots * wafers_per_lot
    counts = df.groupby("wafer_id", observed=True).size().unique()
    assert len(counts) == 1


def test_determinism() -> None:
    a = generate_wafer_dataset(n_lots=1, wafers_per_lot=2, seed=42)
    b = generate_wafer_dataset(n_lots=1, wafers_per_lot=2, seed=42)
    pd.testing.assert_frame_equal(a, b)

    c = generate_wafer_dataset(n_lots=1, wafers_per_lot=2, seed=43)
    assert not a["vt"].equals(c["vt"])
