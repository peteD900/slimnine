# slimnine

Personal data analytics module. Thin wrappers around plotnine, pandas, and polars for quick exploratory analysis.

## What's in here

| Module | Contents |
|---|---|
| `slimnine.plots` | plotnine chart wrappers (line, scatter, bar, hist) |
| `slimnine.munges` | pandas / polars data manipulation helpers |

More to come: grouped linear models, generic ML boilerplate.

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
from slimnine import plots, munges

df = pd.DataFrame({"x": range(10), "y": range(10)})
plots.line(df, x="x", y="y", title="Example").draw()

munges.summary(df)
```
