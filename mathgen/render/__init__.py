"""Draw a figure spec (`mathgen.figures`) as a standalone SVG string.

    from mathgen.render import figure_svg

    svg = figure_svg(question.figure)          # "<svg ...>...</svg>"

The SVG needs no JavaScript, no fonts beyond the page's own and no network: drop it into HTML as is, write it
to a ``.svg`` file, or convert it to PNG or PDF with any SVG tool. Every number drawn comes from the spec; the
renderer evaluates no maths beyond mapping the spec's coordinates onto the drawing.

Accessibility: the root carries ``role="img"`` and a ``<title>`` holding the spec's ``alt``, so a screen reader
announces the figure as one image with that description. ``describe=True`` adds the full description from
`mathgen.describe` as a ``<desc>``, for an accessibility mode.

Theming: every colour is a CSS custom property with a light default. A page sets any of them on the SVG or on
an ancestor to restyle every figure at once (see :data:`THEME_VARIABLES`):

    .question { --mgf-text: #0b1220; --mgf-accent: #7c3aed; }

``theme="dark"`` bakes the dark defaults in instead, for an SVG shown outside a page (a dark slide, a PNG).

The ``<style>`` inside the SVG is scoped to the ``mgf`` class, so it styles nothing else on the page when the
SVG is inlined. Ids (clip paths, arrowheads) are prefixed per figure; pass ``id_prefix`` to choose the prefix
when the same spec is drawn twice on one page.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable

from mathgen import figures
from mathgen.render._svg import STYLE, Ctx, esc, fmt

# kind -> renderer(spec, ctx) -> (width, height, body). `body` is SVG markup in viewBox units.
_RENDERERS: dict[str, Callable[[dict, Ctx], tuple[float, float, str]]] = {}

# The CSS custom properties a page may set, with their light and dark defaults.
THEME_VARIABLES: dict[str, tuple[str, str]] = {
    "--mgf-text": ("#172033", "#e6eaf2"),  # labels, segments, dots
    "--mgf-text-muted": ("#687387", "#9aa4b8"),
    "--mgf-surface": ("#ffffff", "#161c2b"),  # the page behind the figure: label halos, hollow dots
    "--mgf-surface-2": ("#f7f8fa", "#1b2233"),
    "--mgf-surface-3": ("#eef2f6", "#232b3e"),
    "--mgf-border": ("#e2e7ee", "#2b3448"),  # grid lines
    "--mgf-border-strong": ("#cdd5df", "#3c475f"),  # axes, ticks, outlines
    "--mgf-accent": ("#2563eb", "#6d9bff"),  # the first data colour
    "--mgf-accent-soft": ("#eaf1ff", "#1e2a44"),
    "--mgf-tick": ("#475268", "#c9d1e2"),  # tick and axis labels
    "--mgf-c1": ("#c2410c", "#fbbf24"),  # further data colours, in order
    "--mgf-c2": ("#047857", "#4ade80"),
    "--mgf-c3": ("#7c3aed", "#c4b5fd"),
    "--mgf-c4": ("#be185d", "#f9a8d4"),
    "--mgf-c5": ("#0e7490", "#67e8f9"),
}


def register_renderer(kind: str, renderer: Callable[[dict, Ctx], tuple[float, float, str]]) -> None:
    """Register the renderer for a figure kind. A kind has one renderer."""
    if kind in _RENDERERS:
        raise ValueError(f"figure kind {kind!r} already has a renderer")
    _RENDERERS[kind] = renderer


def renderable_kinds() -> tuple[str, ...]:
    return tuple(sorted(_RENDERERS))


def _defaults(theme: str) -> str:
    col = {"light": 0, "dark": 1}[theme]
    # Each private variable reads the page's public one first, so a page can restyle without touching the SVG.
    return ";".join(f"--_{name[6:]}:var({name},{pair[col]})" for name, pair in THEME_VARIABLES.items())


def figure_svg(spec: dict, *, theme: str = "light", id_prefix: str | None = None, describe: bool = False) -> str:
    """The SVG for figure ``spec``. Raises ValueError for an invalid spec or a kind with no renderer.

    ``describe=True`` adds the figure's full description (`mathgen.describe`) as a ``<desc>`` that the root's
    ``aria-describedby`` points at, so a screen reader reads it after the alt text. Use it only in an
    accessibility mode: the description can state the value a question asks the learner to read off.
    """
    if theme not in ("light", "dark"):
        raise ValueError("theme must be 'light' or 'dark'")
    figures.check(spec)
    kind = spec["kind"]
    renderer = _RENDERERS.get(kind)
    if renderer is None:
        raise ValueError(f"figure kind {kind!r} has no SVG renderer yet")
    if id_prefix is None:
        digest = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:8]
        id_prefix = f"mgf{digest}"
    ctx = Ctx(id_prefix)
    w, h, body = renderer(spec, ctx)
    title_id = ctx.uid("title")
    desc, described_by = "", ""
    if describe:
        from mathgen.describe import describe as _describe

        desc_id = ctx.uid("desc")
        desc = f'<desc id="{desc_id}">{esc(_describe(spec))}</desc>'
        described_by = f' aria-describedby="{desc_id}"'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="mgf mgf-{esc(kind)}" viewBox="0 0 {fmt(w)} {fmt(h)}" '
        f'width="{fmt(w)}" height="{fmt(h)}" role="img" aria-labelledby="{title_id}"{described_by} '
        f'style="{_defaults(theme)}">'
        f'<title id="{title_id}">{esc(spec["alt"])}</title>{desc}'
        f"<style>{_style()}</style>"
        f"{body}</svg>"
    )


# Each module registers its kinds on import, and may add rules of its own to the stylesheet as `CSS`.
from mathgen.render import cartesian, data, early, grid, measure, space  # noqa: E402

_STYLE = STYLE + "".join(getattr(m, "CSS", "") for m in (cartesian, data, early, grid, measure, space))


def _style() -> str:
    return _STYLE
