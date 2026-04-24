# slimnine — Claude context

Personal data analytics module. Thin wrappers for plotnine plots and pandas/polars data manipulation. Possibly linear-model-per-group and generic ML boilerplate later.

## Package layout

```
src/slimnine/
    __init__.py   — re-exports plots and munges
    plots.py      — plotnine chart wrappers
    munges.py     — pandas / polars helpers
tests/
```

## Key conventions

- **Plots**: each function accepts a DataFrame, column name strings, and optional aesthetic overrides. Returns a `ggplot` object — callers call `.draw()` or save themselves.
- **Munges**: pandas helpers operate on `pd.DataFrame`, polars helpers on `pl.DataFrame`. Prefix polars functions with `pl_` to avoid name collisions.
- **Type hints**: use `from __future__ import annotations` and `TYPE_CHECKING` guards for heavy imports (plotnine, polars) so the module is importable without all optional deps installed.
- **No side effects**: functions return values; nothing prints or draws automatically.
- **Style**: ruff, line length 88, imports sorted (I rule).

## Dev commands

```bash
uv sync --group dev          # install dev deps
uv run pytest                # run tests
uv run ruff check src        # lint
uv run ruff format src       # format
```

## Dependencies

- **Core**: plotnine, pandas, polars
- **Optional `stats`**: scipy, statsmodels
- **Optional `ml`**: scikit-learn
- **Dev**: pytest, ruff, ipykernel

## What to add next

- `slimnine/stats.py` — grouped linear models (statsmodels OLS per group), summary tables
- `slimnine/ml.py` — generic sklearn pipeline boilerplate
- More plot types: boxplot, violin, faceted variants
