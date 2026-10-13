"""Shared pieces for the SVG renderers: markup building, number formatting, the scoped stylesheet.

Colours: renderers never write a hex colour for anything themeable. They use the private variables
``var(--_text)``, ``var(--_surface)``, ``var(--_accent)``, ``var(--_tick)``, ``var(--_c1)`` ... which
`mathgen.render.figure_svg` defines on the root from the public ``--mgf-*`` ones (see ``THEME_VARIABLES``).
A colour that is part of the object itself (a gold coin, a banknote's colour) may be a literal.

Classes: everything drawn uses the ``mgf-`` classes styled in :data:`STYLE`; prefer a class over a ``style``
attribute so a page can restyle it.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape as _escape

# Drawing width in viewBox units shared by most kinds; the SVG scales to its container.
W = 400

# Text width estimate: digits and most letters are about 0.62 em wide in a sans-serif face.
CHAR_W = 0.62


def esc(s: object) -> str:
    """Text or an attribute value, escaped for XML."""
    return _escape(str(s), {'"': "&quot;"})


def fmt(v: float) -> str:
    """A coordinate as short, stable text: at most two decimals, no trailing zeros, no "-0"."""
    if isinstance(v, bool):
        raise TypeError("fmt takes a number")
    if not math.isfinite(v):
        raise ValueError(f"cannot draw a non-finite coordinate {v!r}")
    r = round(float(v), 2)
    if r == 0:
        return "0"
    s = f"{r:.2f}".rstrip("0").rstrip(".")
    return s


def text_width(text: str, font_size: float) -> float:
    return len(text) * font_size * CHAR_W


def _attrs(attrs: dict) -> str:
    out = []
    for k, v in attrs.items():
        if v is None or v is False:
            continue
        name = k.rstrip("_").replace("_", "-")
        if v is True:
            v = name
        elif isinstance(v, (int, float)):
            v = fmt(v)
        out.append(f' {name}="{esc(v)}"')
    return "".join(out)


def el(tag: str, attrs: dict | None = None, *children: str) -> str:
    """One element. Attribute names use ``_`` for ``-`` (``stroke_width``) and a trailing ``_`` for a Python
    keyword (``class_``); ``None``/``False`` drop the attribute; numbers go through :func:`fmt`. Children are
    markup already, so text content must be passed through :func:`esc` (or use :func:`text`)."""
    a = _attrs(attrs or {})
    if not children:
        return f"<{tag}{a}/>"
    return f"<{tag}{a}>{''.join(children)}</{tag}>"


def g(attrs: dict | None, *children: str) -> str:
    return el("g", attrs, *children) if any(children) else ""


def text(x: float, y: float, s: str, *, anchor: str = "middle", class_: str | None = None, **attrs) -> str:
    """A ``<text>`` element; ``s`` is escaped here."""
    return el("text", {"x": x, "y": y, "text_anchor": anchor, "class_": class_, **attrs}, esc(s))


def points_attr(pts) -> str:
    return " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)


class Ctx:
    """Per-figure state: unique ids for defs (clip paths, markers) so two figures on one page never clash."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self._n = 0

    def uid(self, name: str) -> str:
        self._n += 1
        return f"{self.prefix}-{name}-{self._n}"


# Scoped to `.mgf` so an inlined SVG styles nothing else on the page. Mirrors the reference web renderer.
STYLE = (
    ".mgf{color:var(--_text);font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;max-width:100%;height:auto}"
    ".mgf text{paint-order:stroke;stroke:var(--_surface);stroke-width:5px;stroke-linejoin:round;fill:var(--_text)}"
    ".mgf .mgf-grid line{stroke:var(--_border);stroke-width:1}"
    ".mgf .mgf-grid .mgf-grid-minor{stroke-opacity:.45}"
    ".mgf .mgf-axis line,.mgf .mgf-number-line-rule{stroke:var(--_border-strong);stroke-width:1.6}"
    ".mgf .mgf-tick line{stroke:var(--_border-strong);stroke-width:1.2}"
    ".mgf .mgf-tick-label text{fill:var(--_tick);font-size:14px;font-weight:500}"
    ".mgf .mgf-axis-label text{fill:var(--_tick);font-size:15px;font-weight:700}"
    ".mgf .mgf-line line,.mgf .mgf-curve{stroke:var(--_accent);stroke-width:2;fill:none}"
    ".mgf .mgf-segment line,.mgf .mgf-interval line,.mgf .mgf-ray line{stroke:currentColor;stroke-width:3}"
    ".mgf .mgf-dot{stroke:currentColor;stroke-width:2}"
    ".mgf .mgf-dot-closed{fill:currentColor}"
    ".mgf .mgf-dot-open{fill:var(--_surface)}"
    ".mgf .mgf-arrowhead{fill:var(--_border-strong)}"
    ".mgf .mgf-point text,.mgf .mgf-segment text,.mgf .mgf-line text,.mgf .mgf-mark text,.mgf .mgf-data-label text"
    "{fill:var(--_text);font-size:15px;font-weight:700}"
    ".mgf .mgf-curve-1{stroke:var(--_c1);stroke-dasharray:8 4}"
    ".mgf .mgf-curve-2{stroke:var(--_c2);stroke-dasharray:2 3.5;stroke-width:2.6}"
    ".mgf .mgf-data-label .mgf-curve-text-0{fill:var(--_accent)}"
    ".mgf .mgf-data-label .mgf-curve-text-1{fill:var(--_c1)}"
    ".mgf .mgf-data-label .mgf-curve-text-2{fill:var(--_c2)}"
    ".mgf .mgf-fill-0{fill:var(--_accent)}.mgf .mgf-fill-1{fill:var(--_c1)}.mgf .mgf-fill-2{fill:var(--_c2)}"
    ".mgf .mgf-fill-3{fill:var(--_c3)}.mgf .mgf-fill-4{fill:var(--_c4)}.mgf .mgf-fill-5{fill:var(--_c5)}"
    ".mgf .mgf-title text{font-size:16px}"
    ".mgf .mgf-plain{stroke:none;font-family:inherit}"
    ".mgf .mgf-money-text{fill:#1f2937}"
)
