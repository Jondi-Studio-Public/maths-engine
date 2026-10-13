"""A full text description of a figure spec (`mathgen.figures`), for learners who cannot see it.

    from mathgen.describe import describe

    text = describe(question.figure)   # "An analogue clock. The minute hand points at 3. ..."

A spec's ``alt`` is deliberately vague ("An analogue clock.") so it never gives an answer away, which leaves a
screen-reader user unable to do the question. The description states what a sighted learner sees: every
value, label, mark and position the figure draws, in reading order. It never works anything out that the
figure does not show (a clock is described by where its hands point, not by the time it reads), so it gives a
blind learner the same information as the picture and no more.

That still means it can hold the answer when reading the figure is the whole question: a protractor's arm at
140, a bar of height 7. The picture shows that too. `mathgen.present` flags that case
(``figure.description.reveals_answer``). A platform shows the description in an accessibility mode; the alt text
stays the default.

Every figure kind has a describer; a new kind registers one here, next to its SVG renderer.
"""

from __future__ import annotations

from typing import Callable

from mathgen import figures

# kind -> describer(spec) -> description (one or more full sentences, plain Unicode)
_DESCRIBERS: dict[str, Callable[[dict], str]] = {}


def register_describer(kind: str, describer: Callable[[dict], str]) -> None:
    """Register the describer for a figure kind. A kind has one describer."""
    if kind in _DESCRIBERS:
        raise ValueError(f"figure kind {kind!r} already has a describer")
    _DESCRIBERS[kind] = describer


def describable_kinds() -> tuple[str, ...]:
    return tuple(sorted(_DESCRIBERS))


def describe(spec: dict) -> str:
    """The full description of figure ``spec``. Raises ValueError for an invalid spec or a kind with no
    describer."""
    figures.check(spec)
    describer = _DESCRIBERS.get(spec["kind"])
    if describer is None:
        raise ValueError(f"figure kind {spec['kind']!r} has no describer yet")
    text = " ".join(describer(spec).split())
    title = (spec.get("title") or "").strip()
    if title and title not in text:
        text = f"{sentence(title)} {text}"
    return text


# ---------------------------------------------------------------- helpers for describers


def num(v) -> str:
    """A number as the figure draws it: ``3`` not ``3.0``, ``-2.5``, ``0.1`` not ``0.1000000001``."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.10g}"
    return str(v)


def point(p) -> str:
    """``(x, y)``."""
    return f"({num(p[0])}, {num(p[1])})"


def join(items, final: str = "and") -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {final} {items[-1]}"


def count(n, noun: str, plural: str | None = None) -> str:
    """``1 dot``, ``3 dots``, ``0 dots``."""
    return f"{num(n)} {noun if n == 1 else (plural or noun + 's')}"


def sentence(text: str) -> str:
    """``text`` with a capital first letter and a closing full stop."""
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!" else text + "."


# Each module registers its kinds on import.
from mathgen.describe import cartesian, data, early, grid, measure, space  # noqa: E402, F401
