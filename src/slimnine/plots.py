"""Thin plotnine wrappers for common chart types."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from plotnine import ggplot


def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    colour: str | None = None,
    title: str | None = None,
) -> ggplot:
    from plotnine import aes, geom_line, ggplot, labs

    mapping = aes(x=x, y=y, colour=colour) if colour else aes(x=x, y=y)
    p = ggplot(df, mapping) + geom_line()
    if title:
        p = p + labs(title=title)
    return p


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    colour: str | None = None,
    title: str | None = None,
) -> ggplot:
    from plotnine import aes, geom_point, ggplot, labs

    mapping = aes(x=x, y=y, colour=colour) if colour else aes(x=x, y=y)
    p = ggplot(df, mapping) + geom_point()
    if title:
        p = p + labs(title=title)
    return p


def bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    fill: str | None = None,
    title: str | None = None,
) -> ggplot:
    from plotnine import aes, geom_col, ggplot, labs

    mapping = aes(x=x, y=y, fill=fill) if fill else aes(x=x, y=y)
    p = ggplot(df, mapping) + geom_col()
    if title:
        p = p + labs(title=title)
    return p


def hist(
    df: pd.DataFrame,
    x: str,
    bins: int = 30,
    fill: str | None = None,
    title: str | None = None,
) -> ggplot:
    from plotnine import aes, geom_histogram, ggplot, labs

    mapping = aes(x=x, fill=fill) if fill else aes(x=x)
    p = ggplot(df, mapping) + geom_histogram(bins=bins)
    if title:
        p = p + labs(title=title)
    return p
