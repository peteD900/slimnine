"""Synthetic wafer-probe data for examples and tests.

Provides a deterministic generator (`generate_wafer_dataset`) that produces a
multi-lot, multi-wafer DataFrame with realistic spatial signatures.
"""

# ---- Imports ---- #

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "VariationConfig",
    "generate_wafer_dataset",
]


# ---- User-facing config ---- #


@dataclass(frozen=True)
class VariationConfig:
    """High-level dials for tuning synthetic wafer dataset variation.

    Each field is a multiplier on the corresponding internal baseline.
    ``1.0`` reproduces the default dataset; raise above ``1.0`` to amplify
    that effect, drop below to suppress it.

    Variation between groups
    ------------------------
    lot_to_lot
        Widens (or narrows) the uniform ranges from which per-lot parameters
        are drawn. Larger values -> bigger differences between lots.
    wafer_to_wafer
        Scales the per-wafer jitter applied to lot parameters. Larger values
        -> wafers within the same lot diverge more from each other.
    within_wafer_noise
        Scales the per-die Gaussian noise added to every KPI.

    Spatial signature amplitudes
    ----------------------------
    gradient_strength
        Multiplies the across-wafer linear gradient amplitude on vt, idsat
        and the ileak diagonal contribution.
    bowl_strength
        Multiplies the radial-bowl amplitude on vt and idsat.
    edge_ring_strength
        Multiplies the edge-ring contribution to ileak.
    defect_strength
        Multiplies both the cluster and sparse-defect contributions to ileak.

    Scratch effects
    ---------------
    scratch_kpi_impact
        Multiplies the KPI hit (ileak bump and vt drop) on dies hit by a
        scratch. Does NOT change how often scratches occur.
    scratch_incidence
        Multiplies the per-lot scratch probability (clipped to [0, 1]),
        i.e. controls how many wafers actually get scratches.
    """

    # Variation between groups.
    lot_to_lot: float = 1.0
    wafer_to_wafer: float = 1.0
    within_wafer_noise: float = 1.0

    # Spatial signature amplitudes.
    gradient_strength: float = 1.0
    bowl_strength: float = 1.0
    edge_ring_strength: float = 1.0
    defect_strength: float = 1.0

    # Scratch effects.
    scratch_kpi_impact: float = 1.0
    scratch_incidence: float = 1.0


def _widen(lo: float, hi: float, k: float) -> tuple[float, float]:
    """Widen a uniform-range [lo, hi] around its midpoint by factor k."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * k
    return mid - half, mid + half


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


# ---- Scratch patterns ---- #


def _scratch_arc(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    r_arc: float,
    theta_start: float,
    theta_span: float,
    half_width: float,
) -> np.ndarray:
    """Boolean mask: dies within a curved arc band."""
    dist = np.hypot(x - cx, y - cy)
    radial_ok = np.abs(dist - r_arc) <= half_width
    theta = np.arctan2(y - cy, x - cx)
    delta = (theta - theta_start) % (2.0 * np.pi)
    angle_ok = delta <= theta_span
    return radial_ok & angle_ok


def _min_dist_to_polyline(x: np.ndarray, y: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Minimum distance from each die to any segment of a polyline."""
    dists = np.full(len(x), np.inf)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        abx, aby = bx - ax, by - ay
        ab2 = abx**2 + aby**2
        if ab2 < 1e-12:
            continue
        t = np.clip(((x - ax) * abx + (y - ay) * aby) / ab2, 0.0, 1.0)
        px = ax + t * abx - x
        py = ay + t * aby - y
        dists = np.minimum(dists, np.hypot(px, py))
    return dists


def _scratch_walk_mask(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    angle0: float,
    length: float,
    half_width: float,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boolean mask: dies within a curving walk-path scratch."""
    pts = [(x0, y0)]
    step = length / n_steps
    angle = angle0
    cx, cy = x0, y0
    for _ in range(n_steps):
        angle += rng.uniform(-0.25, 0.25)  # ~±15° random walk per step
        cx += step * np.cos(angle)
        cy += step * np.sin(angle)
        pts.append((cx, cy))
    pts_arr = np.array(pts)
    return _min_dist_to_polyline(x, y, pts_arr) <= half_width


def _gen_scratches(
    x: np.ndarray,
    y: np.ndarray,
    max_radius: float,
    rng: np.random.Generator,
    n_scratches: int,
) -> np.ndarray:
    """Return a float32 ileak-bump array from 0–n_scratches damage marks."""
    out = np.zeros(len(x), dtype=np.float32)
    for _ in range(n_scratches):
        scratch_type = rng.choice(["arc", "walk"])
        ileak_bump = float(rng.uniform(0.4, 1.5))
        if scratch_type == "arc":
            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            rad = float(np.sqrt(rng.uniform(0.0, 1.0)) * max_radius * 0.7)
            cx = rad * np.cos(ang)
            cy = rad * np.sin(ang)
            r_arc = float(rng.uniform(8.0, 25.0))
            theta_start = float(rng.uniform(0.0, 2.0 * np.pi))
            theta_span = float(rng.uniform(np.pi / 6.0, 2.0 * np.pi / 3.0))
            half_width = float(rng.uniform(0.5, 1.5))
            mask = _scratch_arc(
                x, y, cx, cy, r_arc, theta_start, theta_span, half_width
            )
        else:
            edge_angle = float(rng.uniform(0.0, 2.0 * np.pi))
            r0 = max_radius - float(rng.uniform(0.0, 20.0))
            x0 = r0 * np.cos(edge_angle)
            y0 = r0 * np.sin(edge_angle)
            toward_center = np.pi + edge_angle
            angle0 = toward_center + float(rng.uniform(-np.pi / 2.0, np.pi / 2.0))
            length = float(rng.uniform(15.0, 40.0))
            half_width = float(rng.uniform(0.3, 1.0))
            mask = _scratch_walk_mask(x, y, x0, y0, angle0, length, half_width, 20, rng)
        out[mask] += ileak_bump
    return out


# ---- Per-lot / per-wafer parameters ---- #


@dataclass
class _Params:
    bowl_strength: float
    vt_grad_angle: float
    idsat_grad_angle: float
    ring_strength: float
    defect_rate: float
    n_clusters: int
    diag_angle: float
    diag_strength: float
    scratch_prob: float
    n_scratches: int


def _draw_lot_params(
    rng: np.random.Generator, cfg: VariationConfig | None = None
) -> _Params:
    if cfg is None:
        cfg = VariationConfig()
    k = cfg.lot_to_lot

    bowl_lo, bowl_hi = _widen(0.7, 1.3, k)
    vt_a_lo, vt_a_hi = _widen(-60.0, -30.0, k)  # ~ TL -> BR
    id_a_lo, id_a_hi = _widen(-15.0, 15.0, k)  # ~ L -> R
    ring_lo, ring_hi = _widen(0.6, 1.4, k)
    def_lo, def_hi = _widen(0.002, 0.008, k)
    diag_a_lo, diag_a_hi = _widen(-90.0, 90.0, k)
    diag_s_lo, diag_s_hi = _widen(0.0, 0.6, k)
    scr_lo, scr_hi = _widen(0.0, 0.5, k)

    raw_scratch_prob = float(rng.uniform(scr_lo, scr_hi))
    scratch_prob = float(np.clip(raw_scratch_prob * cfg.scratch_incidence, 0.0, 1.0))

    return _Params(
        bowl_strength=float(rng.uniform(bowl_lo, bowl_hi)),
        vt_grad_angle=float(rng.uniform(vt_a_lo, vt_a_hi)),
        idsat_grad_angle=float(rng.uniform(id_a_lo, id_a_hi)),
        ring_strength=float(rng.uniform(ring_lo, ring_hi)),
        defect_rate=max(5e-4, float(rng.uniform(def_lo, def_hi))),
        n_clusters=int(rng.integers(2, 5)),
        diag_angle=float(rng.uniform(diag_a_lo, diag_a_hi)),
        diag_strength=max(0.0, float(rng.uniform(diag_s_lo, diag_s_hi))),
        scratch_prob=scratch_prob,
        n_scratches=0,  # set per-wafer in _jitter_for_wafer
    )


def _jitter_for_wafer(
    lot: _Params, rng: np.random.Generator, cfg: VariationConfig | None = None
) -> _Params:
    if cfg is None:
        cfg = VariationConfig()
    w = cfg.wafer_to_wafer

    def j(scale: float = 0.15) -> float:
        s = scale * w
        return 1.0 + float(rng.uniform(-s, s))

    return _Params(
        bowl_strength=lot.bowl_strength * j(),
        vt_grad_angle=lot.vt_grad_angle + float(rng.uniform(-10.0 * w, 10.0 * w)),
        idsat_grad_angle=lot.idsat_grad_angle + float(rng.uniform(-10.0 * w, 10.0 * w)),
        ring_strength=lot.ring_strength * j(),
        defect_rate=max(5e-4, lot.defect_rate * j()),
        n_clusters=max(1, int(round(lot.n_clusters * j()))),
        diag_angle=lot.diag_angle + float(rng.uniform(-20.0 * w, 20.0 * w)),
        diag_strength=max(0.0, lot.diag_strength * j(0.2)),
        scratch_prob=lot.scratch_prob,
        n_scratches=int(rng.binomial(2, lot.scratch_prob)),
    )


# ---- KPI synthesis ---- #


def _gen_wafer_kpis(
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    max_radius: float,
    p: _Params,
    rng: np.random.Generator,
    cfg: VariationConfig | None = None,
):
    if cfg is None:
        cfg = VariationConfig()
    n = len(x)

    bowl_amp = cfg.bowl_strength
    grad_amp = cfg.gradient_strength
    ring_amp = cfg.edge_ring_strength
    def_amp = cfg.defect_strength
    noise_amp = cfg.within_wafer_noise
    scratch_amp = cfg.scratch_kpi_impact

    bowl = _radial_bowl(r, max_radius).astype(np.float32)
    vt_grad = _gradient(x, y, p.vt_grad_angle, max_radius).astype(np.float32)
    idsat_grad = _gradient(x, y, p.idsat_grad_angle, max_radius).astype(np.float32)
    diag = _gradient(x, y, p.diag_angle, max_radius).astype(np.float32)

    vt = (
        400.0
        + 30.0 * bowl_amp * p.bowl_strength * (bowl - 0.5)
        + 6.0 * grad_amp * (vt_grad - 0.5)
        + 10.0 * grad_amp * p.diag_strength * (diag - 0.5)
        + rng.normal(0.0, 4.0 * noise_amp, n)
    ).astype(np.float32)

    idsat = (
        800.0
        - 60.0 * bowl_amp * p.bowl_strength * (bowl - 0.5)
        + 18.0 * grad_amp * (idsat_grad - 0.5)
        - 20.0 * grad_amp * p.diag_strength * (diag - 0.5)
        + rng.normal(0.0, 12.0 * noise_amp, n)
    ).astype(np.float32)

    ring = _edge_ring(r, max_radius, sigma=max_radius * 0.08).astype(np.float32)
    clusters = _defect_clusters(
        x, y, rng, p.n_clusters, sigma=4.0, max_radius=max_radius
    )
    defects = _sparse_defects(rng, n, p.defect_rate)
    scratches = _gen_scratches(x, y, max_radius, rng, p.n_scratches)

    ileak = (
        1.0
        + 0.4 * ring_amp * p.ring_strength * ring
        + 1.5 * def_amp * clusters
        + 2.0 * def_amp * defects
        + 0.1 * grad_amp * p.diag_strength * (diag - 0.5)
        + scratch_amp * scratches
        + rng.normal(0.0, 0.05 * noise_amp, n)
    ).astype(np.float32)
    ileak = np.maximum(ileak, np.float32(0.05))

    scratch_mask = scratches > 0
    vt = np.where(scratch_mask, vt - np.float32(3.0 * scratch_amp), vt).astype(
        np.float32
    )

    z_vt = (vt - vt.mean()) / (vt.std() + 1e-9)
    z_idsat = (idsat - idsat.mean()) / (idsat.std() + 1e-9)
    z_ileak = (ileak - ileak.mean()) / (ileak.std() + 1e-9)
    freq = (
        1500.0
        + 60.0 * z_idsat
        - 40.0 * z_vt
        - 30.0 * z_ileak
        + rng.normal(0.0, 15.0 * noise_amp, n)
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
    config: VariationConfig | None = None,
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
    config
        `VariationConfig` controlling lot-to-lot, wafer-to-wafer and
        within-wafer variation magnitudes plus per-effect amplitudes
        (gradient, bowl, edge ring, defects, scratches). Defaults to
        `VariationConfig()`, which reproduces the bundled fixture.

    Returns
    -------
    pandas.DataFrame
        One row per die. Columns: lot_id, wafer_id, wafer_no, x_test, y_test,
        row, col, die_id, radius, vt, idsat, ileak, freq, pass.
    """
    if config is None:
        config = VariationConfig()
    rng = np.random.default_rng(seed)

    x, y, r, row, col, die_id = _build_grid(pitch_x, pitch_y, max_radius)

    frames: list[pd.DataFrame] = []
    for li in range(1, n_lots + 1):
        lot_id = f"L{li:02d}"
        lot_params = _draw_lot_params(rng, config)
        for wi in range(1, wafers_per_lot + 1):
            wafer_id = f"{lot_id}_W{wi}"
            wp = _jitter_for_wafer(lot_params, rng, config)
            vt, idsat, ileak, freq, pf = _gen_wafer_kpis(
                x, y, r, max_radius, wp, rng, config
            )

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
