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

`generate_wafer_dataset` synthesises a multi-lot, multi-wafer DataFrame on
demand — useful for testing plots and exploring the API:

```python
import plotnine as pn

from slimnine.example_data import generate_wafer_dataset
from slimnine.wafer_maps import WaferMapConfig, plot_wafermap_spectral

df = generate_wafer_dataset(seed=0)

# single-wafer map
one = df[df["wafer_id"] == "L01_W1"]
plot_wafermap_spectral(one, kpi="vt").draw()

# faceted across one lot
cfg = WaferMapConfig(facet=pn.facet_wrap("~wafer_id"))
plot_wafermap_spectral(df[df["lot_id"] == "L01"], kpi="idsat", cfg).draw()
```

The default dataset is 3 lots × 6 wafers × ~9k die. KPIs are `vt`, `idsat`,
`ileak`, `freq` (with realistic spatial signatures and cross-KPI
correlations) plus a derived `pass` flag. Pass a `VariationConfig` to dial
lot-to-lot, wafer-to-wafer or signature amplitudes up or down:

```python
from slimnine import VariationConfig, generate_wafer_dataset

cfg = VariationConfig(gradient_strength=4.0, scratch_kpi_impact=5.0)
df = generate_wafer_dataset(n_lots=2, wafers_per_lot=3, seed=0, config=cfg)
```

## Example documents

Worked examples live under `docs/` as a [Quarto](https://quarto.org)
website. Drop a new `.qmd` in `docs/examples/` and it will be picked up
automatically by the sidebar and home-page listing.

```bash
uv run quarto preview docs                          # live preview
uv run quarto render docs                           # render the whole site to docs/_site
uv run quarto render docs/examples/wafer_maps.qmd   # render one document
```

Project-wide settings live in `docs/_quarto.yml`; per-folder defaults for
the `examples/` collection live in `docs/examples/_metadata.yml`.

### Publishing the site to GitHub Pages

Quarto can build the site and push it to a `gh-pages` branch in one
command. **First-time setup**, on GitHub: repo → *Settings* → *Pages* →
*Source: Deploy from a branch* → branch `gh-pages` / folder `/ (root)` →
*Save*.

Then, from the repo root with the dev venv active:

```bash
uv run quarto publish gh-pages docs
```

The first run asks to confirm the target and writes `docs/_publish.yml`
recording it (commit that file). Subsequent runs render and push in one
step. The site will be live at `https://<your-username>.github.io/slimnine/`
within a minute or two.

To re-publish after edits, just re-run the same command. CI-driven
auto-publishing on push to `main` can be added later as a GitHub Actions
workflow.
