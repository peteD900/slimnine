# slimnine — Claude context

Personal data analytics module. Thin wrappers for plotnine plots and pandas/polars data manipulation. Possibly linear-model-per-group and generic ML boilerplate later.

## Package layout

Plot modules are grouped **by domain**, not by chart geometry. Each domain
module owns its own `*Config` dataclass and plotting functions.

```
src/slimnine/
    __init__.py    — re-exports domain modules
    wafer_maps.py  — wafer map plots + WaferMapConfig
    # future:
    # distributions.py — hist / density / boxplot / violin
    # timeseries.py    — line / scatter with time axis helpers
    # palettes.py      — shared colour helpers and fill scales
    # munges.py        — pandas helpers
    # munges_pl.py     — polars helpers
tests/
```

## Key conventions

- **Plots**: each function accepts a DataFrame, column name strings, and a domain `*Config`. Returns a `ggplot` object — callers call `.draw()` or save themselves.
- **Domain grouping**: new chart types live in a module named after the analysis domain (e.g. `wafer_maps.py`), not the geometry. Each domain has one base plot function plus any domain-specific variants.
- **Configs**: per-domain frozen dataclass (e.g. `WaferMapConfig`) holds defaults for column names, figure size, facets. Keep defaults domain-appropriate.
- **Munges** (planned): pandas helpers operate on `pd.DataFrame`, polars helpers on `pl.DataFrame`. Prefix polars functions with `pl_` or keep them in a `munges_pl.py` module.
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

- More domain plot modules: `distributions.py`, `timeseries.py`
- `palettes.py` — extract shared colour helpers / fill scales so each new domain doesn't re-implement variants
- `munges.py` / `munges_pl.py` — pandas and polars data helpers
- `stats.py` — grouped linear models (statsmodels OLS per group), summary tables
- `ml.py` — generic sklearn pipeline boilerplate
