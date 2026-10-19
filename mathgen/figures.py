"""Figure specs: a picture in a question, described as data.

A generator that needs a figure builds one here and sets ``Question.figure``.
The spec is a plain JSON-able dict; a front-end component draws it as
SVG and never evaluates maths itself. Tests assert on the spec (the intercept
drawn is the intercept in the answer), never on pixels.

Other kinds register themselves with:func:`register_kind` so every kind shares one validator and one alt-text
path.

Envelope, common to every kind::

    {"kind": str, "alt": str, ...kind-specific keys}

``alt`` is the accessible description. For a question that asks the learner to
READ a value off the figure it names only the object ("a straight line on a
grid"), never the value; :func:`assert_no_leak` checks labels and alt text.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

Number = int | float

# kind -> validator(spec) -> list of problems (empty when valid)
_KINDS: dict[str, Callable[[dict], list[str]]] = {}

# kind -> texts(spec) -> every string that kind draws beyond the shared keys
# (``alt``, ``title``, axis labels). Feeds :func:`labels` and so the leak check.
_TEXTS: dict[str, Callable[[dict], list[str]]] = {}

# topic -> reader(question) -> answer_text, re-derived from `question.figure`
# alone (never from `question.answer`/`answer_text`). See `register_reader`.
_READERS: dict[str, Callable[[Any], str]] = {}

# A curve is sampled here, not in the browser; this caps the spec's size.
MAX_CURVE_POINTS = 400


def register_kind(
    kind: str,
    validator: Callable[[dict], list[str]],
    texts: Callable[[dict], list[str]] | None = None,
) -> None:
    """Register a figure kind's validator. A kind is registered once.

    ``texts(spec)`` returns every string the kind draws beyond ``alt``,
    ``title``, ``x_label`` and ``y_label`` (category names, tick captions,
    vertex names, a key such as "each circle = 2"), so :func:`assert_no_leak`
    sees everything a learner sees. A kind that draws no other text omits it.
    """
    if kind in _KINDS:
        raise ValueError(f"figure kind {kind!r} is already registered")
    _KINDS[kind] = validator
    if texts is not None:
        _TEXTS[kind] = texts


def kinds() -> tuple[str, ...]:
    return tuple(sorted(_KINDS))


def register_reader(topic: str, reader: Callable[[Any], str]) -> None:
    """Register a plausibility reader for a `mathgen` topic that draws a figure.

    ``reader(question)`` must re-derive the answer from ``question.figure``
    alone (the spec a learner actually sees), never from ``question.answer``
    or ``question.answer_text`` -- otherwise it would only restate the
    generator's own construction and could never catch it disagreeing with
    the figure. It returns the re-derived answer in whatever text form the
    skill's grader expects (the same shape as ``answer_text``).

    A topic registers at most one reader. `scripts/regrade_sweep.py` calls every registered reader on every
    generated question of that topic and checks the result grades correct against the real key -- the
    "generator calls its own plausibility check" rule ("Plausibility is part of the generator, not the
    review").
    """
    if topic in _READERS:
        raise ValueError(f"topic {topic!r} already has a registered figure reader")
    _READERS[topic] = reader


def get_reader(topic: str) -> Callable[[Any], str] | None:
    """The registered reader for ``topic``, or ``None`` if it has none."""
    return _READERS.get(topic)


def validate(spec: Any) -> list[str]:
    """Problems with ``spec``; an empty list means it is a valid figure."""
    if not isinstance(spec, dict):
        return ["figure must be a dict"]
    kind = spec.get("kind")
    if kind not in _KINDS:
        return [f"unknown figure kind {kind!r} (known: {', '.join(kinds())})"]
    problems = []
    alt = spec.get("alt")
    if not isinstance(alt, str) or not alt.strip():
        problems.append("figure needs non-empty alt text")
    problems.extend(_KINDS[kind](spec))
    return problems


def check(spec: dict) -> dict:
    """Return ``spec`` unchanged, or raise ValueError listing its problems."""
    problems = validate(spec)
    if problems:
        raise ValueError("invalid figure: " + "; ".join(problems))
    return spec


def labels(spec: dict) -> list[str]:
    """Every piece of text the figure shows, including its alt text."""
    out = [spec.get("alt", "")]
    for key in ("title", "x_label", "y_label"):
        if spec.get(key):
            out.append(spec[key])
    for group in ("points", "segments", "lines", "curves", "marks"):
        for item in spec.get(group, ()):
            if item.get("label"):
                out.append(item["label"])
    canvas = spec.get("canvas")
    if isinstance(canvas, dict) and isinstance(canvas.get("vertex_labels"), list):
        out.extend(t for t in canvas["vertex_labels"] if isinstance(t, str))
    texts = _TEXTS.get(spec.get("kind"))
    if texts is not None:
        out.extend(t for t in texts(spec) if t)
    return out


def assert_no_leak(spec: dict, answer_texts: Iterable[str]) -> None:
    """Raise AssertionError if any answer text appears in the figure's labels or alt text.

    Pass the forms a learner would type (``"3"``, ``"(2, 5)"``, ``"y = 2x + 1"``).
    A per-skill test calls this for every seed it renders.
    """
    shown = [s.replace(" ", "") for s in labels(spec)]
    for ans in answer_texts:
        a = str(ans).replace(" ", "")
        if not a:
            continue
        for text in shown:
            if a == text or (len(a) > 1 and a in text):
                raise AssertionError(f"figure shows the answer {ans!r} in {text!r}")


# ---------------------------------------------------------------- helpers


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _xy(p: Any) -> bool:
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(_is_num(c) for c in p)


def _range_ok(r: Any) -> bool:
    return isinstance(r, (list, tuple)) and len(r) == 2 and all(_is_num(v) for v in r) and r[0] < r[1]


def _clean(v: Number) -> Number:
    """Integers stay integers; floats are rounded so specs are stable across platforms."""
    if isinstance(v, int):
        return v
    r = round(float(v), 6)
    return int(r) if r == int(r) else r


def sample_curve(
    f: Callable[[float], float], x_min: float, x_max: float, y_range: tuple[Number, Number], n: int = 200
) -> list[list[list[Number]]]:
    """Sample ``f`` on [x_min, x_max] into polyline pieces clipped to ``y_range``.

    Returns a list of pieces (each a list of [x, y]); a piece breaks where ``f``
    is undefined or leaves the visible window, so an asymptote is never joined
    across. Total points never exceed MAX_CURVE_POINTS.
    """
    n = max(2, min(n, MAX_CURVE_POINTS))
    lo, hi = y_range
    margin = (hi - lo) * 0.05
    pieces: list[list[list[Number]]] = []
    cur: list[list[Number]] = []
    for i in range(n):
        x = x_min + (x_max - x_min) * i / (n - 1)
        try:
            y = f(x)
        except (ValueError, ZeroDivisionError, OverflowError):
            y = None
        if y is None or not _is_num(y) or not (lo - margin <= y <= hi + margin):
            if len(cur) > 1:
                pieces.append(cur)
            cur = []
            continue
        cur.append([_clean(x), _clean(y)])
    if len(cur) > 1:
        pieces.append(cur)
    return pieces


# ---------------------------------------------------------------- cartesian


def cartesian(
    *,
    x_range: tuple[Number, Number],
    y_range: tuple[Number, Number],
    alt: str,
    grid: Number = 1,
    x_grid: Number | None = None,
    y_grid: Number | None = None,
    equal_scale: bool = True,
    x_tick_format: str = "number",
    points: Iterable[dict] = (),
    segments: Iterable[dict] = (),
    lines: Iterable[dict] = (),
    curves: Iterable[dict] = (),
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
    tick_labels: bool = True,
    canvas: dict | None = None,
) -> dict:
    """A coordinate plane.

    - ``points``: ``{"at": [x, y], "label"?: str, "open"?: bool}``
    - ``segments``: ``{"from": [x, y], "to": [x, y], "label"?: str,
      "open_start"?: bool, "open_end"?: bool}`` (step graphs use the open ends)
    - ``lines``: infinite lines through two points ``{"through": [[x, y], [x, y]], "label"?: str}``
    - ``curves``: ``{"pieces": sample_curve(...), "label"?: str}``
    - ``x_grid``/``y_grid``: per-axis grid step, each defaulting to ``grid``
      (the shared default kept for backward compatibility -- a caller that
      only ever passed ``grid`` renders exactly as before). The 60-step
      validator limit applies to each axis against its own step.
    - ``equal_scale``: ``True`` (default) keeps today's behaviour -- one grid
      unit is the same drawn length on both axes. ``False`` scales each axis
      independently to fill the same fixed drawing box, for a contextual
      graph whose axes are different quantities (cost vs. items,
      distance-time, a conversion graph).
    - ``x_tick_format``: ``"number"`` (default) or ``"pi"``. With ``"pi"``,
      ``x`` values are in units of pi (``x=0.5`` means ``pi/2``) and x tick
      labels are written as reduced multiples of pi (``pi/2``, ``pi``,
      ``3pi/2``, a proper minus sign for negatives, ``0`` stays ``0``), for a
      trig graph whose generator samples ``f(pi * u)``.
    - ``canvas``: how the learner answers by working on the graph (plotting points, drawing a line, shading a
      region, ...); see :mod:`mathgen.figure_kinds.canvas`. Left out of the spec when ``None``.
    """
    x_grid = grid if x_grid is None else x_grid
    y_grid = grid if y_grid is None else y_grid
    spec = {
        "kind": "cartesian",
        "alt": alt,
        "x_range": [_clean(v) for v in x_range],
        "y_range": [_clean(v) for v in y_range],
        "grid": _clean(grid),
        "x_grid": _clean(x_grid),
        "y_grid": _clean(y_grid),
        "equal_scale": bool(equal_scale),
        "x_tick_format": x_tick_format,
        "x_label": x_label,
        "y_label": y_label,
        "tick_labels": tick_labels,
        "points": [dict(p) for p in points],
        "segments": [dict(s) for s in segments],
        "lines": [dict(ln) for ln in lines],
        "curves": [dict(c) for c in curves],
    }
    if title:
        spec["title"] = title
    if canvas is not None:
        spec["canvas"] = dict(canvas)
    return check(spec)


def _validate_cartesian(spec: dict) -> list[str]:
    problems = []
    if not _range_ok(spec.get("x_range")):
        problems.append("x_range must be [min, max] with min < max")
    if not _range_ok(spec.get("y_range")):
        problems.append("y_range must be [min, max] with min < max")
    grid = spec.get("grid")
    if not _is_num(grid) or grid <= 0:
        problems.append("grid must be a positive number")
    x_grid = spec.get("x_grid")
    if not _is_num(x_grid) or x_grid <= 0:
        problems.append("x_grid must be a positive number")
    y_grid = spec.get("y_grid")
    if not _is_num(y_grid) or y_grid <= 0:
        problems.append("y_grid must be a positive number")
    if not isinstance(spec.get("equal_scale"), bool):
        problems.append("equal_scale must be a bool")
    if spec.get("x_tick_format") not in ("number", "pi"):
        problems.append("x_tick_format must be 'number' or 'pi'")
    if problems:
        return problems
    (x0, x1), (y0, y1) = spec["x_range"], spec["y_range"]
    if (x1 - x0) / x_grid > 60:
        problems.append("more than 60 grid steps on the x-axis; widen x_grid")
    if (y1 - y0) / y_grid > 60:
        problems.append("more than 60 grid steps on the y-axis; widen y_grid")

    def inside(p) -> bool:
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    for i, p in enumerate(spec.get("points", [])):
        if not _xy(p.get("at")):
            problems.append(f"points[{i}].at must be [x, y]")
        elif not inside(p["at"]):
            problems.append(f"points[{i}] at {p['at']} is outside the plane")
    for i, s in enumerate(spec.get("segments", [])):
        if not (_xy(s.get("from")) and _xy(s.get("to"))):
            problems.append(f"segments[{i}] needs from and to as [x, y]")
        elif not (inside(s["from"]) and inside(s["to"])):
            problems.append(f"segments[{i}] leaves the plane")
    for i, ln in enumerate(spec.get("lines", [])):
        t = ln.get("through")
        if not (isinstance(t, (list, tuple)) and len(t) == 2 and all(_xy(p) for p in t)):
            problems.append(f"lines[{i}].through must be two [x, y] points")
        elif list(t[0]) == list(t[1]):
            problems.append(f"lines[{i}].through points must differ")
    total = 0
    for i, c in enumerate(spec.get("curves", [])):
        pieces = c.get("pieces")
        if not isinstance(pieces, list) or not pieces:
            problems.append(f"curves[{i}].pieces must be a non-empty list")
            continue
        for piece in pieces:
            if not isinstance(piece, list) or len(piece) < 2 or not all(_xy(p) for p in piece):
                problems.append(f"curves[{i}] has a piece that is not a list of 2+ [x, y] points")
                break
            total += len(piece)
    if total > MAX_CURVE_POINTS * 4:
        problems.append(f"curves carry {total} points; sample fewer")
    if "canvas" in spec:
        from mathgen.figure_kinds.canvas import validate_canvas

        problems.extend(validate_canvas(spec["canvas"], "plane"))
    return problems


# ---------------------------------------------------------------- number line


def number_line(
    *,
    range_: tuple[Number, Number],
    alt: str,
    tick: Number = 1,
    marks: Iterable[dict] = (),
    rays: Iterable[dict] = (),
    intervals: Iterable[dict] = (),
    tick_labels: bool = True,
    canvas: dict | None = None,
) -> dict:
    """A number line.

    - ``marks``: ``{"at": x, "open"?: bool, "label"?: str}`` (a dot; open = hollow)
    - ``rays``: ``{"from": x, "direction": "left" | "right", "open"?: bool}``
    - ``intervals``: ``{"from": x, "to": x, "open_start"?: bool, "open_end"?: bool}``
    - ``canvas``: how the learner answers by marking the line; see :mod:`mathgen.figure_kinds.canvas`.
    """
    spec = {
        "kind": "number_line",
        "alt": alt,
        "range": [_clean(v) for v in range_],
        "tick": _clean(tick),
        "tick_labels": tick_labels,
        "marks": [dict(m) for m in marks],
        "rays": [dict(r) for r in rays],
        "intervals": [dict(iv) for iv in intervals],
    }
    if canvas is not None:
        spec["canvas"] = dict(canvas)
    return check(spec)


def _validate_number_line(spec: dict) -> list[str]:
    problems = []
    r = spec.get("range")
    if not _range_ok(r):
        return ["range must be [min, max] with min < max"]
    tick = spec.get("tick")
    if not _is_num(tick) or tick <= 0:
        return ["tick must be a positive number"]
    if (r[1] - r[0]) / tick > 40:
        problems.append("more than 40 ticks; widen the tick")
    for i, m in enumerate(spec.get("marks", [])):
        if not _is_num(m.get("at")) or not (r[0] <= m["at"] <= r[1]):
            problems.append(f"marks[{i}].at must be a number on the line")
    for i, ray in enumerate(spec.get("rays", [])):
        if not _is_num(ray.get("from")) or not (r[0] <= ray["from"] <= r[1]):
            problems.append(f"rays[{i}].from must be a number on the line")
        if ray.get("direction") not in ("left", "right"):
            problems.append(f"rays[{i}].direction must be 'left' or 'right'")
    for i, iv in enumerate(spec.get("intervals", [])):
        a, b = iv.get("from"), iv.get("to")
        if not (_is_num(a) and _is_num(b) and r[0] <= a < b <= r[1]):
            problems.append(f"intervals[{i}] must satisfy min <= from < to <= max")
    if "canvas" in spec:
        from mathgen.figure_kinds.canvas import validate_canvas

        problems.extend(validate_canvas(spec["canvas"], "number_line"))
    return problems


register_kind("cartesian", _validate_cartesian)
register_kind("number_line", _validate_number_line)

# The other kinds register themselves on import; they use the helpers above.
from mathgen import figure_kinds as _figure_kinds  # noqa: E402,F401
