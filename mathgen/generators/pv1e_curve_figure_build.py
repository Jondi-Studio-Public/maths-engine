"""PV1E shared figure helpers: a 12 x 12 window and curve sampling.

Every PV1E row draws a curve on a plane that is twelve grid steps wide and twelve tall (the
window may be scaled, or shifted off the origin, by a tier). A curve is sampled at 193 points
(step 1/16 across a 12-wide window), so every integer x is an exact sample: a function that is
undefined at an integer x (``1/(x - h)``) breaks its polyline exactly there and nowhere else.
No asymptote is drawn: the reader finds it from where the sampled pieces break or flatten.

This module registers no topic. It exists so the four PV1E generators share one plane builder
and one multi-zone sampler instead of four copies.
"""

from __future__ import annotations

from typing import Callable

from .. import figures

N_SAMPLES = 193  # 12 / 192 = 1/16 per step: integer x are exact samples
SPAN = 12  # grid steps across and up in every PV1E window


def plane(
    *, x_range, y_range, alt: str, curves=(), points=(), lines=(), x_grid=1, y_grid=1, equal_scale: bool = True
) -> dict:
    """A cartesian figure with the PV1E defaults (axis names x and y, tick numbers shown)."""
    return figures.cartesian(
        x_range=x_range,
        y_range=y_range,
        alt=alt,
        x_grid=x_grid,
        y_grid=y_grid,
        equal_scale=equal_scale,
        curves=curves,
        points=points,
        lines=lines,
    )


def pieces(f: Callable[[float], float], x_range, y_range, n: int = N_SAMPLES):
    """``f`` sampled across ``x_range`` into polyline pieces clipped to ``y_range``."""
    return figures.sample_curve(f, x_range[0], x_range[1], tuple(y_range), n=n)


def zoned_pieces(f: Callable[[float], float], zones, y_range, n_zone: int = 40):
    """Sample ``f`` on consecutive zones ``[(a, b), ...]`` (each end meets the next start).

    Used for a curve that runs along a vertical gridline (a logarithm near its asymptote):
    the zones get geometrically finer toward the asymptote, so the curve is drawn close to the
    line instead of stopping a sample-step short of it. Pieces that meet at a zone boundary
    are merged back into one polyline.
    """
    out: list[list[list[float]]] = []
    for a, b in zones:
        for piece in figures.sample_curve(f, a, b, tuple(y_range), n=n_zone):
            if out and out[-1][-1] == piece[0]:
                out[-1].extend(piece[1:])
            else:
                out.append(list(piece))
    return out


def curve_pieces(fig: dict) -> list[list[list[float]]]:
    """Every sampled piece of a figure's curves, ordered by where each starts."""
    found = [p for c in fig["curves"] for p in c["pieces"]]
    return sorted(found, key=lambda p: (p[0][0], p[0][1]))
