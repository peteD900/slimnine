import math
import re

import numpy as np
import pandas as pd
import pytest

from slimnine.example_data import generate_wafer_dataset, load_wafer_dataset
from slimnine.wafer_maps import plot_wafermap_spectral

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


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    return generate_wafer_dataset(n_lots=2, wafers_per_lot=2, seed=0)


@pytest.fixture(scope="module")
def loaded_df() -> pd.DataFrame:
    return load_wafer_dataset()


def test_columns_and_dtypes(small_df: pd.DataFrame) -> None:
    assert list(small_df.columns) == EXPECTED_COLS
    assert small_df["x_test"].dtype == np.float32
    assert small_df["y_test"].dtype == np.float32
    assert small_df["radius"].dtype == np.float32
    assert small_df["vt"].dtype == np.float32
    assert small_df["pass"].dtype == np.bool_
    assert isinstance(small_df["lot_id"].dtype, pd.CategoricalDtype)
    assert isinstance(small_df["wafer_id"].dtype, pd.CategoricalDtype)


def test_geometry_within_disk(small_df: pd.DataFrame) -> None:
    max_radius = 53.5
    assert (small_df["radius"] <= max_radius + 1e-4).all()
    assert ((small_df["x_test"] == 0) & (small_df["y_test"] == 0)).any()


def test_die_count_per_wafer(small_df: pd.DataFrame) -> None:
    counts = small_df.groupby("wafer_id", observed=True).size().unique()
    assert len(counts) == 1
    expected = math.pi * 53.5**2
    assert abs(counts[0] - expected) / expected < 0.05


def test_lot_and_wafer_cardinality(small_df: pd.DataFrame) -> None:
    assert small_df["lot_id"].nunique() == 2
    assert small_df["wafer_id"].nunique() == 2 * 2


def test_die_id_format_and_origin(small_df: pd.DataFrame) -> None:
    pattern = re.compile(r"^\d{3}\.\d{3}$")
    assert small_df["die_id"].map(lambda s: bool(pattern.match(s))).all()

    one = small_df[small_df["wafer_id"] == "L01_W1"]
    top = one.loc[one["y_test"].idxmax()]
    left = one.loc[one["x_test"].idxmin()]
    assert top["row"] == 1
    assert left["col"] == 1


def test_kpi_ranges(small_df: pd.DataFrame) -> None:
    assert small_df["vt"].between(300, 500).all()
    assert small_df["idsat"].between(600, 1000).all()
    assert (small_df["ileak"] > 0).all()
    assert small_df["freq"].between(800, 2200).all()
    pass_rate = small_df["pass"].mean()
    assert 0.3 < pass_rate < 0.95


def test_correlation_signs(small_df: pd.DataFrame) -> None:
    # KPIs share latent signals by design, so correlations should be stable.
    assert small_df["vt"].corr(small_df["idsat"]) < -0.3
    assert small_df["idsat"].corr(small_df["freq"]) > 0.3
    assert small_df["ileak"].corr(small_df["freq"]) < 0.0
    assert small_df["vt"].corr(small_df["freq"]) < -0.3


def test_determinism() -> None:
    a = generate_wafer_dataset(n_lots=1, wafers_per_lot=2, seed=42)
    b = generate_wafer_dataset(n_lots=1, wafers_per_lot=2, seed=42)
    pd.testing.assert_frame_equal(a, b)

    c = generate_wafer_dataset(n_lots=1, wafers_per_lot=2, seed=43)
    assert not a["vt"].equals(c["vt"])


def test_load_wafer_dataset_matches_schema(loaded_df: pd.DataFrame) -> None:
    assert list(loaded_df.columns) == EXPECTED_COLS
    assert loaded_df["lot_id"].nunique() == 3
    assert loaded_df["wafer_id"].nunique() == 3 * 6
    assert isinstance(loaded_df["lot_id"].dtype, pd.CategoricalDtype)


def test_plot_smoke(small_df: pd.DataFrame) -> None:
    one = small_df[small_df["wafer_id"] == "L01_W1"]
    plot = plot_wafermap_spectral(one, kpi="vt")
    plot.draw()


def test_diagonal_gradient_produces_spatial_variation() -> None:
    # Force a strong diagonal gradient to confirm it creates within-wafer
    # ileak variation correlated with a known direction.
    from slimnine.example_data import _Params, _gen_wafer_kpis, _build_grid

    x, y, r, _, _, _ = _build_grid(1.0, 1.0, 53.5)
    rng = np.random.default_rng(7)
    p = _Params(
        bowl_strength=0.0,
        vt_grad_angle=0.0,
        idsat_grad_angle=0.0,
        ring_strength=0.0,
        defect_rate=0.0,
        n_clusters=0,
        diag_angle=45.0,   # TL→BR at 45°
        diag_strength=1.0,  # maximum strength
        scratch_prob=0.0,
        n_scratches=0,
    )
    vt, _, _, _, _ = _gen_wafer_kpis(x, y, r, 53.5, p, rng)
    proj = (x * np.cos(np.deg2rad(45)) + y * np.sin(np.deg2rad(45)))
    assert np.corrcoef(vt, proj)[0, 1] > 0.3


def test_scratches_elevate_ileak() -> None:
    # _gen_scratches must return a non-zero bump array for n_scratches > 0.
    from slimnine.example_data import _gen_scratches, _build_grid

    x, y, r, _, _, _ = _build_grid(1.0, 1.0, 53.5)
    bump = _gen_scratches(x, y, 53.5, np.random.default_rng(42), n_scratches=2)
    assert bump.dtype == np.float32
    assert (bump > 0).any(), "expected at least one die inside a scratch region"
    assert bump.max() >= 0.4, "scratch ileak bump should be at least min(U[0.4, 1.5])"
