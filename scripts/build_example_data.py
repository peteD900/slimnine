"""Regenerate the bundled example wafer parquet.

Run from the repo root:

    uv run python scripts/build_example_data.py

Tweak the constants below to change the size or geometry of the committed
fixture; tests assume the defaults.
"""

from pathlib import Path

import polars as pl

from slimnine.example_data import generate_wafer_dataset

N_LOTS = 3
WAFERS_PER_LOT = 6
PITCH_X = 1.0
PITCH_Y = 1.0
MAX_RADIUS = 53.5
SEED = 0

OUT = Path("src/slimnine/data/example_wafers.parquet")


def main() -> None:
    df = generate_wafer_dataset(
        n_lots=N_LOTS,
        wafers_per_lot=WAFERS_PER_LOT,
        pitch_x=PITCH_X,
        pitch_y=PITCH_Y,
        max_radius=MAX_RADIUS,
        seed=SEED,
    )

    # Build a polars frame column-by-column from numpy arrays — pl.from_pandas
    # would require pyarrow to handle category / object columns.
    pl_df = pl.DataFrame(
        {
            "lot_id": df["lot_id"].astype(str).to_numpy(),
            "wafer_id": df["wafer_id"].astype(str).to_numpy(),
            "wafer_no": df["wafer_no"].to_numpy(),
            "x_test": df["x_test"].to_numpy(),
            "y_test": df["y_test"].to_numpy(),
            "row": df["row"].to_numpy(),
            "col": df["col"].to_numpy(),
            "die_id": df["die_id"].astype(str).to_numpy(),
            "radius": df["radius"].to_numpy(),
            "vt": df["vt"].to_numpy(),
            "idsat": df["idsat"].to_numpy(),
            "ileak": df["ileak"].to_numpy(),
            "freq": df["freq"].to_numpy(),
            "pass": df["pass"].to_numpy(),
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pl_df.write_parquet(OUT, compression="zstd")
    print(
        f"wrote {len(df):,} rows ({df['wafer_id'].nunique()} wafers"
        f" across {df['lot_id'].nunique()} lots) to {OUT}"
        f" — {OUT.stat().st_size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
