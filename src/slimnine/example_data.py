"""Synthetic wafer-probe data for examples and tests.

Provides a deterministic generator (`generate_wafer_dataset`) that produces a
multi-lot, multi-wafer DataFrame with realistic spatial signatures, and a
loader (`load_wafer_dataset`) that reads the bundled parquet shipped with the
package.
"""

# ---- Imports ---- #

from __future__ import annotations

import importlib.resources as _res
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "generate_wafer_dataset",
    "load_wafer_dataset",
]


_DATA_FILE = "example_wafers.parquet"


# ---- Geometry ---- #


def _build_grid(pitch_x: float, pitch_y: float, max_radius: float):
    """Build the die grid: square lattice masked to a disk.

    Returns 1-D arrays of x, y, r (mm), row, col (1-based, top-left origin),
    and die_id (string `"RRR.CCC"`).
    """
    n_x = int(np.floor(max_radius / pitch_x))
    n_y = int(np.floor(max_radius / pitch_y))
    xs = np.arange(-n_x, n_x + 1) * pitch_x
    ys = np.arange(-n_y, n_y + 1) * pitch_y

    xx, yy = np.meshgrid(xs, ys)
    rr = np.hypot(xx, yy)
    mask = rr <= max_radius

    n_rows = len(ys)
    i_idx, j_idx = np.indices(xx.shape)
    row_grid = (n_rows - i_idx).astype(np.int16)  # row 1 = top (max y)
    col_grid = (j_idx + 1).astype(np.int16)  # col 1 = left (min x)

    x = xx[mask].astype(np.float32)
    y = yy[mask].astype(np.float32)
    r = rr[mask].astype(np.float32)
    row = row_grid[mask]
    col = col_grid[mask]
    die_id = np.array([f"{rr_:03d}.{cc_:03d}" for rr_, cc_ in zip(row, col)])

    return x, y, r, row, col, die_id


# ---- Basis fields ---- #


def _radial_bowl(r: np.ndarray, r_max: float) -> np.ndarray:
    return (r / r_max) ** 2


def _edge_ring(r: np.ndarray, r_max: float, sigma: float) -> np.ndarray:
    return np.exp(-((r - r_max) ** 2) / (2.0 * sigma * sigma))


def _gradient(
    x: np.ndarray, y: np.ndarray, angle_deg: float, r_max: float
) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    proj = (x * np.cos(a) + y * np.sin(a)) / r_max
    return (proj + 1.0) / 2.0


def _sparse_defects(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    n_def = int(round(rate * n))
    if n_def <= 0:
        return out
    idx = rng.choice(n, size=n_def, replace=False)
    out[idx] = rng.uniform(0.5, 1.0, size=n_def).astype(np.float32)
    return out


def _defect_clusters(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_clusters: int,
    sigma: float,
    max_radius: float,
) -> np.ndarray:
    out = np.zeros_like(x)
    if n_clusters <= 0:
        return out
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_clusters)
    radii = np.sqrt(rng.uniform(0.0, 1.0, size=n_clusters)) * max_radius * 0.9
    cx = radii * np.cos(angles)
    cy = radii * np.sin(angles)
    for k in range(n_clusters):
        d2 = (x - cx[k]) ** 2 + (y - cy[k]) ** 2
        out += np.exp(-d2 / (2.0 * sigma * sigma))
    return out.astype(np.float32)


# ---- Per-lot / per-wafer parameters ---- #


@dataclass
class _Params:
    bowl_strength: float
    vt_grad_angle: float
    idsat_grad_angle: float
    ring_strength: float
    defect_rate: float
    n_clusters: int


def _draw_lot_params(rng: np.random.Generator) -> _Params:
    return _Params(
        bowl_strength=float(rng.uniform(0.7, 1.3)),
        vt_grad_angle=float(rng.uniform(-60.0, -30.0)),  # ~ TL -> BR
        idsat_grad_angle=float(rng.uniform(-15.0, 15.0)),  # ~ L -> R
        ring_strength=float(rng.uniform(0.6, 1.4)),
        defect_rate=float(rng.uniform(0.002, 0.008)),
        n_clusters=int(rng.integers(2, 5)),
    )


def _jitter_for_wafer(lot: _Params, rng: np.random.Generator) -> _Params:
    def j(scale: float = 0.15) -> float:
        return 1.0 + float(rng.uniform(-scale, scale))

    return _Params(
        bowl_strength=lot.bowl_strength * j(),
        vt_grad_angle=lot.vt_grad_angle + float(rng.uniform(-10.0, 10.0)),
        idsat_grad_angle=lot.idsat_grad_angle + float(rng.uniform(-10.0, 10.0)),
        ring_strength=lot.ring_strength * j(),
        defect_rate=max(5e-4, lot.defect_rate * j()),
        n_clusters=max(1, int(round(lot.n_clusters * j()))),
    )


# ---- KPI synthesis ---- #


def _gen_wafer_kpis(
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    max_radius: float,
    p: _Params,
    rng: np.random.Generator,
):
    n = len(x)

    bowl = _radial_bowl(r, max_radius).astype(np.float32)
    vt_grad = _gradient(x, y, p.vt_grad_angle, max_radius).astype(np.float32)
    idsat_grad = _gradient(x, y, p.idsat_grad_angle, max_radius).astype(np.float32)

    vt = (
        400.0
        + 30.0 * p.bowl_strength * (bowl - 0.5)
        + 6.0 * (vt_grad - 0.5)
        + rng.normal(0.0, 4.0, n)
    ).astype(np.float32)

    idsat = (
        800.0
        - 60.0 * p.bowl_strength * (bowl - 0.5)
        + 18.0 * (idsat_grad - 0.5)
        + rng.normal(0.0, 12.0, n)
    ).astype(np.float32)

    ring = _edge_ring(r, max_radius, sigma=max_radius * 0.08).astype(np.float32)
    clusters = _defect_clusters(
        x, y, rng, p.n_clusters, sigma=4.0, max_radius=max_radius
    )
    defects = _sparse_defects(rng, n, p.defect_rate)

    ileak = (
        1.0
        + 0.4 * p.ring_strength * ring
        + 1.5 * clusters
        + 2.0 * defects
        + rng.normal(0.0, 0.05, n)
    ).astype(np.float32)
    ileak = np.maximum(ileak, np.float32(0.05))

    z_vt = (vt - vt.mean()) / (vt.std() + 1e-9)
    z_idsat = (idsat - idsat.mean()) / (idsat.std() + 1e-9)
    z_ileak = (ileak - ileak.mean()) / (ileak.std() + 1e-9)
    freq = (
        1500.0
        + 60.0 * z_idsat
        - 40.0 * z_vt
        - 30.0 * z_ileak
        + rng.normal(0.0, 15.0, n)
    ).astype(np.float32)

    pass_flag = (ileak < 1.5) & (vt > 370.0) & (vt < 430.0) & (freq > 1400.0)
    return vt, idsat, ileak, freq, pass_flag


# ---- Public API ---- #


def generate_wafer_dataset(
    n_lots: int = 3,
    wafers_per_lot: int = 6,
    pitch_x: float = 1.0,
    pitch_y: float = 1.0,
    max_radius: float = 53.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic multi-lot wafer dataset.

    Parameters
    ----------
    n_lots, wafers_per_lot
        Number of lots and wafers per lot.
    pitch_x, pitch_y
        Die pitch in mm.
    max_radius
        Wafer radius in mm; die outside the disk are dropped.
    seed
        Seed for `numpy.random.default_rng`. Same seed -> identical output.

    Returns
    -------
    pandas.DataFrame
        One row per die. Columns: lot_id, wafer_id, wafer_no, x_test, y_test,
        row, col, die_id, radius, vt, idsat, ileak, freq, pass.
    """
    rng = np.random.default_rng(seed)

    x, y, r, row, col, die_id = _build_grid(pitch_x, pitch_y, max_radius)

    frames: list[pd.DataFrame] = []
    for li in range(1, n_lots + 1):
        lot_id = f"L{li:02d}"
        lot_params = _draw_lot_params(rng)
        for wi in range(1, wafers_per_lot + 1):
            wafer_id = f"{lot_id}_W{wi}"
            wp = _jitter_for_wafer(lot_params, rng)
            vt, idsat, ileak, freq, pf = _gen_wafer_kpis(x, y, r, max_radius, wp, rng)

            frames.append(
                pd.DataFrame(
                    {
                        "lot_id": lot_id,
                        "wafer_id": wafer_id,
                        "wafer_no": np.int8(wi),
                        "x_test": x,
                        "y_test": y,
                        "row": row,
                        "col": col,
                        "die_id": die_id,
                        "radius": r,
                        "vt": vt,
                        "idsat": idsat,
                        "ileak": ileak,
                        "freq": freq,
                        "pass": pf,
                    }
                )
            )

    out = pd.concat(frames, ignore_index=True)
    out["lot_id"] = out["lot_id"].astype("category")
    out["wafer_id"] = out["wafer_id"].astype("category")
    return out


def load_wafer_dataset() -> pd.DataFrame:
    """Load the bundled example wafer dataset.

    Reads the parquet committed at `slimnine/data/example_wafers.parquet`
    via polars and assembles a pandas DataFrame column-by-column to avoid
    requiring pyarrow at runtime. `lot_id` / `wafer_id` are restored to
    categorical dtype.
    """
    import polars as pl

    resource = _res.files("slimnine").joinpath("data", _DATA_FILE)
    with _res.as_file(resource) as path:
        plf = pl.read_parquet(path)

    df = pd.DataFrame({c: plf[c].to_numpy() for c in plf.columns})
    for c in ("lot_id", "wafer_id"):
        df[c] = df[c].astype("category")
    return df
