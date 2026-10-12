"""Shared helpers for the generators that draw ``diagram``, ``tree_diagram`` and ``venn`` figures.

Not a generator module (the leading underscore keeps the package loader from importing it as one). Holds the
two things every such generator needs: a way to DRAW an angle picture that is honest without being measurable,
and the small readers that re-derive an answer from a figure spec alone.
"""

from __future__ import annotations

import re
from typing import Sequence

from mathgen import figures
from mathgen.core import attempts, check
from mathgen.figure_kinds import space

#: Drawn angles stay at least this wide (a label needs room) and this far from 90 and 180 degrees, so a picture
#: never turns an obtuse angle into a right angle or an acute one into a reflex one.
_ZONES = {"acute": (22.0, 84.0), "obtuse": (96.0, 174.0), "reflex": (186.0, 338.0)}


def _zone(value: float) -> tuple[float, float]:
    if value in (90, 180) or value <= 0 or value >= 360:
        raise ValueError(f"cannot draw an angle of {value} degrees in a not-to-scale picture")
    return _ZONES["acute" if value < 90 else "obtuse" if value < 180 else "reflex"]


def drawn_sweeps(rng, true: Sequence[float], hide: Sequence[int] = (), tries: int = 200) -> list[float]:
    """Drawn sizes for a not-to-scale angle picture.

    ``true`` are the real sizes (they total 360 or 180). The result totals the same, keeps every angle the same
    KIND (acute, obtuse or reflex) as its real size, never narrower than 22 degrees, differs from every real
    size by at least 3 degrees and, for each index in ``hide`` (the angle the question asks for), by at least 10
    degrees, so measuring the picture does not give the answer. A few real sets leave no such drawing (an
    angle of 15 degrees beside one of 195 and 150, with the 195 hidden); that raises ``VerificationError``
    and the caller draws another set.
    """
    total = sum(true)
    for _ in attempts(tries, "a not-to-scale set of drawn angles"):
        out = []
        for t in true[:-1]:
            lo, hi = _zone(t)
            out.append(round(rng.uniform(max(lo, t - 28), min(hi, t + 28)), 2))
        out.append(round(total - sum(out), 2))
        lo, hi = _zone(true[-1])
        if not lo <= out[-1] <= hi:
            continue
        if any(abs(o - t) < 3 for o, t in zip(out, true)) or any(abs(out[i] - true[i]) < 10 for i in hide):
            continue
        return out
    raise AssertionError("unreachable")


def leaks(fig: dict, answer_text: str) -> bool:
    """Does the answer appear in the figure's visible text? (Generators redraw when it does.)"""
    try:
        figures.assert_no_leak(fig, [answer_text])
    except AssertionError:
        return True
    return False


def degrees(value: float | int) -> str:
    """A numeric angle label: ``65°``."""
    return f"{value}°"


# ---------------------------------------------------------------- readers (figure alone)

_NUMERIC = re.compile(r"^(\d+(?:\.\d+)?)°$")


def _positions(fig: dict) -> dict:
    return {p["id"]: tuple(p["at"]) for p in fig["points"]}


def arcs_of(fig: dict) -> list[dict]:
    pos = _positions(fig)
    out = []
    for a in fig["angles"]:
        out.append(
            {
                **a,
                "sweep": space.sweep_degrees(pos[a["at"]], pos[a["from"]], pos[a["to"]]),
                "start": space._bearing(pos[a["at"]], pos[a["from"]]),
            }
        )
    return out


def read_missing_angle(fig: dict) -> int:
    """The size of the angle labelled with a letter among angles that fill a straight line or a full turn.

    Works from the picture alone: the marked angles at the vertex tile the line (their drawn sizes total 180) or
    the turn (360), whatever the labels say, so the real total is the drawn total.
    """
    arcs = arcs_of(fig)
    check(len({a["at"] for a in arcs}) == 1, "the angles do not share a vertex")
    drawn_total = sum(a["sweep"] for a in arcs)
    total = 180 if abs(drawn_total - 180) < 1 else 360
    check(abs(drawn_total - total) < 1, "the marked angles do not fill a line or a turn")
    unknown = [a for a in arcs if not _NUMERIC.match(a.get("label", ""))]
    check(len(unknown) == 1, "exactly one angle should be marked with a letter")
    known = 0.0
    for a in arcs:
        if a is not unknown[0]:
            m = _NUMERIC.match(a.get("label", ""))
            check(m is not None, "an angle other than x has no number")
            known += float(m.group(1))
    value = total - known
    check(value == int(value) and value > 0, "the missing angle is not a positive whole number")
    return int(value)


_LINEAR = re.compile(r"^\(?\s*(-?\d*)[a-z]\s*(?:([+-])\s*(\d+))?\s*\)?°$")


def is_expression(label: str) -> bool:
    """Is an angle label an expression such as ``(3x + 10)°`` (rather than ``65°`` or a bare letter)?"""
    return bool(label) and not _NUMERIC.match(label) and not re.fullmatch(r"[a-z]", label)


#: Letters a question may use for the unknown, so the same figure is not always "x".
UNKNOWNS = ("x", "y", "a", "b", "p", "q")


def pick_unknown(rng) -> str:
    return rng.choice(UNKNOWNS)


#: Sentences for the angle questions, so the same picture is not always introduced and asked the same way.
POINT_OPENERS = (
    "The angles in the diagram meet at a point and together complete a full turn.",
    "Several angles meet at a point and together complete a full turn, as shown.",
    "The diagram shows angles meeting at a point and filling a full turn.",
    "A number of angles surround a point and make a full turn, as shown.",
    "The angles shown fit together around a point to make a full turn.",
)
LINE_OPENERS = (
    "Two angles sit side by side on a straight line, as shown.",
    "The diagram shows two angles next to each other on a straight line.",
    "Two adjacent angles lie on a straight line, as shown.",
    "Look at the two angles side by side on the straight line.",
    "A straight line carries two angles, as shown.",
)
CROSSING_OPENERS = (
    "Two straight lines cross, as shown. The two marked angles are vertically opposite, and vertically opposite "
    "angles are always equal.",
    "The diagram shows two straight lines crossing. The marked angles are vertically opposite, and vertically "
    "opposite angles are always equal.",
    "Two lines cross at a point. Vertically opposite angles are always equal, and the two marked angles are "
    "vertically opposite.",
    "Two straight lines intersect, as shown. Vertically opposite angles are always equal, and the marked angles "
    "are a vertically opposite pair.",
    "Look at the two crossing lines. The two marked angles are vertically opposite, and vertically opposite "
    "angles are always equal.",
    "Two lines meet at a point and cross over. The marked angles are vertically opposite, which makes them equal.",
)
SIZE_CLOSERS = (
    "Find the size of the angle marked {u}, in degrees.",
    "What is the size of angle {u}, in degrees?",
    "Work out the size of the angle marked {u}, in degrees.",
    "Calculate the size of angle {u}, in degrees.",
    "Find angle {u}, in degrees.",
)
EACH_CLOSERS = (
    "Find the size of each marked angle, in degrees.",
    "What is the size of each marked angle, in degrees?",
    "Work out the size of the marked angles, in degrees.",
    "Calculate the size of each of the marked angles, in degrees.",
    "Find the value of each marked angle, in degrees.",
)


def parse_linear_label(label: str) -> tuple[int, int]:
    """(coefficient, constant) of an angle label such as ``(3x + 10)°``."""
    m = _LINEAR.match(label.strip())
    check(m is not None, f"cannot read the expression {label!r}")
    coef_text, sign, mag = m.groups()
    coef = 1 if coef_text in ("", "+") else -1 if coef_text == "-" else int(coef_text)
    const = 0 if sign is None else int(mag) if sign == "+" else -int(mag)
    return coef, const


def oriented(points: dict[str, tuple[float, float]], orientation: int) -> dict[str, tuple[float, float]]:
    """Move ``points`` to the middle of the 100 x 100 frame, mirrored when ``orientation`` is 4-7 and turned
    counter-clockwise by 90 degrees ``orientation % 4`` times (the same eight orientations as
    ``space.right_triangle``)."""
    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    out = {}
    for key, (x, y) in points.items():
        x, y = x - cx, y - cy
        if orientation >= 4:
            x = -x
        for _ in range(orientation % 4):
            x, y = -y, x
        out[key] = (50 + x, 50 + y)
    return out


def point_on_segment(point, a, b) -> bool:
    """Is ``point`` on the segment a-b (the figure validator's own test)?"""
    return space._on_segment(point, a, b)
