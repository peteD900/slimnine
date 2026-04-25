# slimnine

Personal data analytics module. Thin wrappers around plotnine, pandas, and polars for quick exploratory analysis.

Plot modules are grouped **by domain** (wafer maps, distributions, time series, ...) rather than by chart geometry, so each module owns its own config and any variants specific to that analysis type.

## What's in here

| Module | Contents |
|---|---|
| `slimnine.wafer_maps` | wafer map plots (tile geometry, spectral / discrete / diverging / pass-fail fills) + `WaferMapConfig` |

Planned:

| Module | Contents |
|---|---|
| `slimnine.distributions` | hist, density, boxplot, violin |
| `slimnine.timeseries` | line / scatter with time-axis helpers |
| `slimnine.palettes` | shared colour helpers and fill scales |
| `slimnine.munges` / `slimnine.munges_pl` | pandas and polars data helpers |
| `slimnine.stats` | grouped linear models, summary tables |
| `slimnine.ml` | generic sklearn pipeline boilerplate |

## Install

Requires [UV](https://docs.astral.sh/uv/).

```bash
# clone and install in editable mode
git clone https://github.com/peted900/slimnine.git
cd slimnine
uv sync
```

For optional stats / ML extras:

```bash
uv sync --extra stats
uv sync --extra ml
```

## Dev setup

```bash
uv sync --group dev
uv run pytest
uv run ruff check src
```

## Quick example

A synthetic 3-lot × 6-wafer dataset (~9k die per wafer, ~162k rows) ships
with the package — useful for testing plots and exploring the API:

```python
import plotnine as pn

from slimnine.example_data import load_wafer_dataset
from slimnine.wafer_maps import WaferMapConfig, plot_wafermap_spectral

df = load_wafer_dataset()

# single-wafer map
one = df[df["wafer_id"] == "L01_W1"]
plot_wafermap_spectral(one, kpi="vt").draw()

# faceted across one lot
cfg = WaferMapConfig(facet=pn.facet_wrap("~wafer_id"))
plot_wafermap_spectral(df[df["lot_id"] == "L01"], kpi="idsat", cfg).draw()
```

The bundled dataset has KPIs `vt`, `idsat`, `ileak`, `freq` (with realistic
spatial signatures and cross-KPI correlations) plus a derived `pass` flag.
To regenerate it (e.g. at a different size), edit and run
`scripts/build_example_data.py`. To synthesise data ad-hoc:

```python
from slimnine.example_data import generate_wafer_dataset

df = generate_wafer_dataset(n_lots=2, wafers_per_lot=3, seed=0)
```
