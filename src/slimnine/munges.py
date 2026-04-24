"""Pandas / Polars data manipulation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


# ---------------------------------------------------------------------------
# Pandas helpers
# ---------------------------------------------------------------------------


def reorder_cols(df: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    """Return df with `first` columns leading, remaining columns appended."""
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce numeric column dtypes to smallest fitting type."""
    import pandas as pd

    for col in df.select_dtypes("integer").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes("float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return dtype / null-count / unique-count summary for each column."""
    import pandas as pd

    return pd.DataFrame(
        {
            "dtype": df.dtypes,
            "nulls": df.isna().sum(),
            "unique": df.nunique(),
        }
    )


# ---------------------------------------------------------------------------
# Polars helpers
# ---------------------------------------------------------------------------


def pl_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Return dtype / null-count / unique-count summary for each column."""
    import polars as pl

    return pl.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(d) for d in df.dtypes],
            "nulls": [df[c].null_count() for c in df.columns],
            "unique": [df[c].n_unique() for c in df.columns],
        }
    )
