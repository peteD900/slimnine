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

```python
import pandas as pd
from slimnine.wafer_maps import WaferMapConfig, plot_wafermap_spectral

df = pd.DataFrame({
    "x_test": [...],
    "y_test": [...],
    "kpi": [...],
})

cfg = WaferMapConfig(twidth=3, theight=3)
plot_wafermap_spectral(df, kpi="kpi", cfg=cfg).draw()
```
