# ---- Imports ---- #

import pandas as pd
import plotnine as pn
from dataclasses import dataclass
from typing import Tuple


__all__ = [
    "WaferMapConfig",
    "plot_wafermap",
    "plot_wafermap_spectral",
    "plot_wafermap_discrete",
    "plot_wafermap_diverge",
    "plot_wafermap_passfail",
    "colours_discrete",
]


# ---- WAFER MAPS ---- #


@dataclass(frozen=True)
class WaferMapConfig:
    """

    Configuration options for wafer map plotting.
    Defines column names, facet settings, figure size, tile geometry,
    theming, and layout options used by wafer-map plotting functions.

    """

    x: str = "x_test"
    y: str = "y_test"
    facet: str | None = "scribe_id"
    nrow: int = 1
    fig_size: Tuple[int, int] = (10, 6)
    twidth: float = 3
    theight: float = 3
    scales: str = "fixed"
    facet_text_size: int = 9
    panel_spacing: int = 0
    expand_lims: bool = False
    grid_row: str | None = None
    grid_col: str | None = None
    plot_theme: callable = pn.theme_bw


def plot_wafermap(
    df: pd.DataFrame, kpi: str, cfg: WaferMapConfig | None = None
) -> pn.ggplot:
    """

    Create a wafer map plot using tile geometry.
    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing x/y coordinates and the KPI to plot.
    kpi : str
        Column name of the value used to fill the tiles.
    cfg : WaferMapConfig
        Configuration controlling layout, facets, theme, and geometry.
    Returns

    -------
    plotnine.ggplot
        A plotnine wafer map object.
    """

    if cfg is None:
        cfg = WaferMapConfig()

    plt = (
        pn.ggplot(df, pn.aes(cfg.x, cfg.y, fill=kpi))
        + pn.geom_tile(width=cfg.twidth, height=cfg.theight)
        + cfg.plot_theme()
        + pn.coord_fixed(expand=cfg.expand_lims)
        + pn.theme(
            figure_size=cfg.fig_size,
            panel_spacing=cfg.panel_spacing,
            strip_text=pn.element_text(size=cfg.facet_text_size),
            panel_grid=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            axis_title=pn.element_blank(),
            axis_text=pn.element_blank(),
        )
    )

    if cfg.grid_row or cfg.grid_col:
        plt += pn.facet_grid(cfg.grid_row, cfg.grid_col)

    elif cfg.facet:
        plt += pn.facet_wrap(cfg.facet, nrow=cfg.nrow, scales=cfg.scales)

    return plt


def plot_wafermap_spectral(
    df: pd.DataFrame, kpi: str, cfg: WaferMapConfig | None = None
) -> pn.ggplot:

    plot = plot_wafermap(df, kpi, cfg)

    return plot + pn.scale_fill_cmap("Spectral")


def plot_wafermap_discrete(
    df: pd.DataFrame, kpi: str, cfg: WaferMapConfig | None = None
) -> pn.ggplot:

    plot = plot_wafermap(df, kpi, cfg)

    return plot + pn.scale_fill_manual(values=colours_discrete())


def plot_wafermap_discrete_vir(
    df: pd.DataFrame, kpi: str, cfg: WaferMapConfig | None = None
) -> pn.ggplot:

    plot = plot_wafermap(df, kpi, cfg)

    return plot + pn.scale_fill_cmap_d()


def plot_wafermap_diverge(
    df: pd.DataFrame, kpi: str, midp=0, cfg: WaferMapConfig | None = None
) -> pn.ggplot:

    low = "#4575b4"
    mid = "#f7f7f7"
    high = "#d73027"

    plot = plot_wafermap(df, kpi, cfg) + pn.scale_fill_gradient2(
        low=low, mid=mid, high=high, midpoint=midp
    )

    return plot


def plot_wafermap_passfail(
    df: pd.DataFrame, kpi: str, pal: int = 1, cfg: WaferMapConfig | None = None
) -> pn.ggplot:
    """

    Pass/fail maps.

    Pal options:

        1: # yellow-blue
        2: # muted pink, bright green
        3: # bright yellow, muted blue

    """

    # fail - pass

    pals = {
        1: ["#FFFF00", "#0000FF"],  # yellow-blue
        2: ["#E6C1CC", "#00A651"],  # muted pink (False), bright green (True)
        3: ["#FFD400", "#6F8FCF"],  # bright yellow, muted blue
    }

    pf_cols = pals[pal]

    plot = plot_wafermap(df, kpi, cfg)

    return plot + pn.scale_fill_manual(values=pf_cols)


# ---- COLOURS ---- #


def colours_discrete(n=None) -> list:
    """Shortcut for nice discete pal"""

    colours = [
        "red",
        "cyan",
        "orange",
        "navy",
        "magenta",
        "green",
        "gold",
        "purple",
        "deepskyblue",
        "brown",
        "pink",
        "teal",
        "coral",
        "darkgreen",
        "blue",
        "lightpink",
        "olive",
        "slategray",
        "darkorange",
        "lightseagreen",
    ]

    if n is not None:
        colours = colours[0:n]

    return colours


if __name__ == "__main__":
    pass
