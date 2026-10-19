"""Answer kinds for interactive graph answers: what a front end submits when the learner works on a graph.

A front end draws a canvas from the question's figure ``canvas`` key (`mathgen.figure_kinds.canvas`) and, when the
learner presses "check", sends the state of the canvas back as a reply. Every reply is canonical JSON written by the
widget, or a typed form for a platform that has no widget. The JSON is read with `json.loads` and nothing else, and
no reply is ever evaluated. Every coordinate is an exact Fraction (an integer, a decimal such as ``0.5`` or a
string such as ``"1/3"``), compared with no tolerance. Anything unreadable, off the canvas, off the snap grid or
too large makes the reply wrong, never an error.

Six kinds, each documented with its schema in `docs/design/interactive-graph-answers.md`:

* ``half_planes``: shade the region of a linear inequality (or of a system of them).
* ``number_line_set``: open and closed circles, rays and segments marked on a number line.
* ``polygon``: the vertices of a shape drawn or moved on a grid.
* ``curve``: a parabola or an exponential curve, given by points on it.
* ``bars``: the height of each bar of a bar chart or histogram.
* ``edges``: the edges drawn between the labelled vertices of a network.

``Question.answer`` is a dict that also carries the canvas the learner worked on: ``x_range`` / ``y_range`` (or
``range`` for a number line), as ``[low, high]``, and ``snap``, the grid step the widget snaps to. When present,
a reply point outside the ranges or off the snap lattice is wrong.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction

import sympy

from .inequality import read_solution_set, set_text
from .typed import _POINT_FILLER_RE, _POINT_RE, GradeResult, _multi_part_aware

#: Longest reply read at all. The largest legitimate reply (60 edges, 60 points) is under 3,000 characters.
MAX_REPLY_LENGTH = 8_000
#: Largest absolute coordinate read: a canvas is a few dozen units across.
MAX_COORDINATE = 10**6
#: Most items (points, vertices, bars, half-planes, pieces, edges) a reply may hold.
MAX_ITEMS = 60
MAX_HALF_PLANES = 6
MAX_VERTICES = 12
MAX_EXPONENT = 60
#: Longest typed inequality read.
MAX_TYPED_LENGTH = 400

Point = tuple[Fraction, Fraction]


# ---------------------------------------------------------------------------------------------- reading numbers


def _reject_constant(name):
    raise ValueError(f"{name} is not a number")


def _load(text):
    """The JSON value in `text`, or None. Floats stay as text so ``0.1`` is exactly one tenth."""
    if not isinstance(text, str) or not text or len(text) > MAX_REPLY_LENGTH:
        return None
    try:
        return json.loads(text, parse_float=str, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None


def _is_jsonish(text: str) -> bool:
    return text.lstrip()[:1] in ("{", "[")


def _unbracketed(text: str) -> str:
    """`text` without one outer pair of [] or {} (the typed pairs may sit in list brackets: ``[(0, 0), (1, 1)]``)."""
    text = text.strip()
    if len(text) >= 2 and text[0] + text[-1] in ("[]", "{}"):
        return text[1:-1]
    return text


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def _coord(value) -> Fraction | None:
    """A reply coordinate as an exact Fraction, or None. Plain numbers only: ``3``, ``-1.5``, ``"7/3"``. Unlike the
    ``points`` kind's reader this does not round to six decimals, so a third-step grid must send ``"1/3"``."""
    if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
        return None
    text = str(value).strip().replace("\u2212", "-")
    if len(text) > 40 or not _NUMBER_RE.fullmatch(text):
        return None
    try:
        number = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return None if abs(number) > MAX_COORDINATE else number


def _typed_pts(text: str, limit: int = MAX_ITEMS) -> list[Point] | None:
    """The points typed in `text` (``(1, 2) (3, -4)``, with only separators between), or None."""
    found: list[Point] = []
    for match in _POINT_RE.finditer(text):
        x, y = _coord(match.group(2)), _coord(match.group(3))
        if match.group(1) or x is None or y is None or len(found) >= limit:
            return None  # an end mark, a bad number, or more points than any answer holds
        found.append((x, y))
    if not found or not _POINT_FILLER_RE.fullmatch(_POINT_RE.sub(" ", text)):
        return None
    return found


def _pt(value) -> Point | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = _coord(value[0]), _coord(value[1])
    return None if x is None or y is None else (x, y)


def _pts(values, limit: int = MAX_ITEMS) -> list[Point] | None:
    if not isinstance(values, (list, tuple)) or len(values) > limit:
        return None
    out = [_pt(v) for v in values]
    return None if any(p is None for p in out) else out


def _exact(value) -> Fraction | None:
    """A number in a generator's answer: an int, a Fraction, ``"a/b"`` text or ``{"p": a, "q": b}``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    if isinstance(value, dict) and set(value) == {"p", "q"}:
        try:
            return Fraction(value["p"], value["q"])
        except (TypeError, ZeroDivisionError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return Fraction(value.strip())
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _exact_pt(value) -> Point | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = _exact(value[0]), _exact(value[1])
    return None if x is None or y is None else (x, y)


def fraction_json(value):
    """A Fraction as the JSON an answer dict holds: an int when whole, else ``"a/b"`` text."""
    value = Fraction(value)
    return value.numerator if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def point_json(point) -> list:
    return [fraction_json(point[0]), fraction_json(point[1])]


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _point_text(point) -> str:
    return f"({_fraction_text(point[0])}, {_fraction_text(point[1])})"


# ---------------------------------------------------------------------------------------------- the canvas


class _Canvas:
    """The ranges and snap step the learner worked in, read from the answer dict. Missing means unchecked."""

    def __init__(self, answer):
        answer = answer if isinstance(answer, dict) else {}
        self.x = self._range(answer.get("x_range", answer.get("range")))
        self.y = self._range(answer.get("y_range"))
        snap = _exact(answer.get("snap")) if answer.get("snap") is not None else None
        self.snap = snap if snap is not None and snap > 0 else None

    @staticmethod
    def _range(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        low, high = _exact(value[0]), _exact(value[1])
        return None if low is None or high is None or low >= high else (low, high)

    def has_x(self, x: Fraction) -> bool:
        return self.x is None or self.x[0] <= x <= self.x[1]

    def has_y(self, y: Fraction) -> bool:
        return self.y is None or self.y[0] <= y <= self.y[1]

    def has(self, point: Point) -> bool:
        return self.has_x(point[0]) and self.has_y(point[1])

    def snapped(self, value: Fraction) -> bool:
        return self.snap is None or (value / self.snap).denominator == 1

    def snapped_point(self, point: Point) -> bool:
        return self.snapped(point[0]) and self.snapped(point[1])

    def accepts(self, point: Point) -> bool:
        """A placed point: on the canvas and on the snap lattice."""
        return self.has(point) and self.snapped_point(point)


def _result(question, correct: bool, given_normalised: str | None = None) -> GradeResult:
    return GradeResult(
        correct=bool(correct), canonical=getattr(question, "answer_text", ""), given_normalised=given_normalised
    )


def _answer_dict(question) -> dict | None:
    answer = getattr(question, "answer", None)
    return answer if isinstance(answer, dict) else None


def _only_keys(obj, allowed) -> bool:
    return isinstance(obj, dict) and set(obj) <= set(allowed)


# ---------------------------------------------------------------------------------------------- half_planes
#
# A half-plane is "a x + b y + c > 0" (or ">=" when the boundary is solid), known up to a positive scale. Its key is
# (a, b, c, closed) scaled so that the first non-zero of a, b is 1 or -1: two spellings of one region have one key.

HalfPlane = tuple[Fraction, Fraction, Fraction, bool]

_BOUNDARIES = {"solid": True, "dashed": False}


def _half_plane_key(a: Fraction, b: Fraction, c: Fraction, closed: bool) -> HalfPlane | None:
    scale = abs(a) if a else abs(b)
    if not scale:
        return None
    return (a / scale, b / scale, c / scale, closed)


def _key_from_shading(p: Point, q: Point, shade: Point, boundary: str) -> HalfPlane | None:
    """The key of the half-plane bounded by the line through `p` and `q`, containing `shade`."""
    if boundary not in _BOUNDARIES or p == q:
        return None
    a, b = q[1] - p[1], p[0] - q[0]
    c = -(a * p[0] + b * p[1])
    side = a * shade[0] + b * shade[1] + c
    if side == 0:
        return None  # the shading point is on the line: it says which side
    if side < 0:
        a, b, c = -a, -b, -c
    return _half_plane_key(a, b, c, _BOUNDARIES[boundary])


def _expected_half_planes(answer) -> set[HalfPlane] | None:
    items = answer.get("half_planes") if isinstance(answer, dict) else None
    if not isinstance(items, list) or not items or len(items) > MAX_HALF_PLANES:
        return None
    keys = set()
    for item in items:
        through = item.get("through") if isinstance(item, dict) else None
        if not (isinstance(through, list) and len(through) == 2):
            return None
        p, q, shade = _exact_pt(through[0]), _exact_pt(through[1]), _exact_pt(item.get("shade"))
        if p is None or q is None or shade is None:
            return None
        key = _key_from_shading(p, q, shade, item.get("boundary"))
        if key is None:
            return None
        keys.add(key)
    return keys


def _json_half_planes(value, canvas: _Canvas) -> set[HalfPlane] | None:
    if isinstance(value, dict):
        if not _only_keys(value, {"half_planes"}):
            return None
        value = value.get("half_planes")
    if not isinstance(value, list) or not value or len(value) > MAX_HALF_PLANES:
        return None
    keys = set()
    for item in value:
        if not _only_keys(item, {"through", "boundary", "shade"}) or set(item) != {"through", "boundary", "shade"}:
            return None
        through = item["through"]
        if not (isinstance(through, list) and len(through) == 2):
            return None
        p, q, shade = _pt(through[0]), _pt(through[1]), _pt(item["shade"])
        if p is None or q is None or shade is None:
            return None
        if not (canvas.accepts(p) and canvas.accepts(q) and canvas.has(shade)):
            return None
        key = _key_from_shading(p, q, shade, item["boundary"])
        if key is None:
            return None
        keys.add(key)
    return keys


_OPERATOR_RE = re.compile(r"<=|>=|=<|=>|<|>")
_LINEAR_TERM_RE = re.compile(
    r"\s*([+-]?)\s*(\d+(?:\.\d+)?(?:/\d+)?|\(\s*\d+(?:\.\d+)?\s*/\s*\d+\s*\))?\s*\*?\s*([xy])?"
)
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[;,\n]|\band\b|∧)\s*", re.IGNORECASE)


def _linear_side(text: str) -> tuple[Fraction, Fraction, Fraction] | None:
    """``a x + b y + c`` from text such as ``2x - 3``, ``y``, ``1/2 x + 4`` or ``-x - y + 6``; None if it is not one."""
    a = b = c = Fraction(0)
    pos, first, text = 0, True, text.strip()
    if not text:
        return None
    while pos < len(text):
        m = _LINEAR_TERM_RE.match(text, pos)
        if m is None or m.end() == pos:
            return None
        sign, coef_text, var = m.groups()
        if coef_text is None and var is None:
            return None
        if not first and not sign:
            return None  # "2x 3y" or "2 3": terms need a sign between them
        first = False
        try:
            coef = Fraction(re.sub(r"[()\s]", "", coef_text)) if coef_text else Fraction(1)
        except (ValueError, ZeroDivisionError):
            return None
        if sign == "-":
            coef = -coef
        if var == "x":
            a += coef
        elif var == "y":
            b += coef
        else:
            c += coef
        pos = m.end()
    return a, b, c


def _typed_half_planes(text: str) -> set[HalfPlane] | None:
    text = text.strip()
    for old, new in (("≤", "<="), ("≥", ">="), ("⩽", "<="), ("⩾", ">="), ("≦", "<="), ("≧", ">=")):
        text = text.replace(old, new)
    if not text or len(text) > MAX_TYPED_LENGTH:
        return None
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
    if not clauses or len(clauses) > MAX_HALF_PLANES:
        return None
    keys = set()
    for clause in clauses:
        ops = _OPERATOR_RE.findall(clause)
        if len(ops) != 1:
            return None
        left_text, right_text = _OPERATOR_RE.split(clause)
        left, right = _linear_side(left_text), _linear_side(right_text)
        if left is None or right is None:
            return None
        op = ops[0]
        # left op right  <=>  g > 0 (or >= 0), where g is right - left for "<" and left - right for ">"
        sign = -1 if op in ("<", "<=", "=<") else 1
        a, b, c = (sign * (l - r) for l, r in zip(left, right))
        key = _half_plane_key(a, b, c, op in ("<=", ">=", "=<", "=>"))
        if key is None:
            return None
        keys.add(key)
    return keys


def half_plane_text(key: HalfPlane) -> str:
    """A half-plane key as text: ``2x - y + 3 > 0`` with whole-number coefficients."""
    a, b, c, closed = key
    scale = math.lcm(a.denominator, b.denominator, c.denominator)
    parts = []
    for coef, name in ((a * scale, "x"), (b * scale, "y"), (c * scale, "")):
        if coef == 0:
            continue
        n = int(coef)
        mag = "" if (abs(n) == 1 and name) else str(abs(n))
        piece = f"{mag}{name}"
        parts.append(("- " if n < 0 else "+ ") + piece if parts else ("-" if n < 0 else "") + piece)
    return f"{' '.join(parts)} {'>=' if closed else '>'} 0"


def half_planes_answer(items, *, x_range, y_range, snap=1) -> dict:
    """`Question.answer` for a ``half_planes`` question. Each item is ``(p, q, shade, "solid" | "dashed")``: two
    points on the boundary and a point strictly inside the shaded side. Generators call this."""
    out = []
    for p, q, shade, boundary in items:
        out.append({"through": [point_json(p), point_json(q)], "boundary": boundary, "shade": point_json(shade)})
    answer = {
        "half_planes": out,
        "x_range": [fraction_json(v) for v in x_range],
        "y_range": [fraction_json(v) for v in y_range],
        "snap": fraction_json(snap),
    }
    if _expected_half_planes(answer) is None:
        raise ValueError("not a valid half-plane answer")
    return answer


@_multi_part_aware
def grade_half_planes(question, given: str) -> GradeResult:
    answer = _answer_dict(question)
    expected = _expected_half_planes(answer)
    if expected is None:
        return _result(question, False)
    canvas = _Canvas(answer)
    if not isinstance(given, str) or not given.strip() or len(given) > MAX_REPLY_LENGTH:
        return _result(question, False)
    got = _json_half_planes(_load(given), canvas) if _is_jsonish(given) else _typed_half_planes(given)
    if got is None:
        return _result(question, False)
    text = "; ".join(half_plane_text(k) for k in sorted(got, key=lambda k: tuple(map(str, k))))
    return _result(question, got == expected, text)


# ---------------------------------------------------------------------------------------------- number_line_set
#
# A set of the real line as disjoint pieces (low, low_closed, high, high_closed), merged and sorted, with -inf / +inf
# for a ray. A single marked point is the piece [p, p].

_INF = float("inf")
Piece = tuple[object, bool, object, bool]


def _merge_pieces(pieces: list[Piece]) -> tuple[Piece, ...]:
    ordered = sorted(pieces, key=lambda p: (p[0], not p[1]))
    out: list[list] = []
    for lo, lo_closed, hi, hi_closed in ordered:
        if out:
            last = out[-1]
            joined = last[2] > lo or (last[2] == lo and (last[3] or lo_closed))
            if joined:
                if hi > last[2]:
                    last[2], last[3] = hi, hi_closed
                elif hi == last[2]:
                    last[3] = last[3] or hi_closed
                continue
        out.append([lo, lo_closed, hi, hi_closed])
    return tuple((lo, lc, hi, hc) for lo, lc, hi, hc in out)


def _piece(low, low_closed: bool, high, high_closed: bool) -> Piece | None:
    """A validated piece: `low`/`high` are Fractions or None for unbounded; None if it is empty or backwards."""
    lo, hi = (-_INF if low is None else low), (_INF if high is None else high)
    lc, hc = (False if low is None else low_closed), (False if high is None else high_closed)
    if lo > hi or (lo == hi and not (lc and hc)):
        return None
    return (lo, lc, hi, hc)


def _json_pieces(value, canvas: _Canvas) -> tuple[Piece, ...] | None:
    if isinstance(value, dict):
        if len(value) != 1 or next(iter(value)) not in ("pieces", "marks"):
            return None
        kind, value = next(iter(value.items()))
    else:
        kind = "pieces"
    if not isinstance(value, list) or not value or len(value) > MAX_ITEMS:
        return None
    pieces: list[Piece] = []
    for item in value:
        if kind == "marks":
            x = _coord(item)
            if x is None or not (canvas.has_x(x) and canvas.snapped(x)):
                return None
            pieces.append((x, True, x, True))
            continue
        if not _only_keys(item, {"from", "to", "from_closed", "to_closed"}) or not {"from", "to"} <= set(item):
            return None
        ends = []
        for end in ("from", "to"):
            raw, closed = item[end], item.get(f"{end}_closed")
            if raw is None:
                ends.extend((None, False))
                continue
            x = _coord(raw)
            if x is None or not (canvas.has_x(x) and canvas.snapped(x)) or not isinstance(closed, bool):
                return None
            ends.extend((x, closed))
        piece = _piece(*ends)
        if piece is None:
            return None
        pieces.append(piece)
    return _merge_pieces(pieces)


def _pieces_from_set(solution) -> tuple[Piece, ...] | None:
    """The merged pieces of a SymPy set of rational intervals and points, or None when a bound is not rational."""

    def bound(value, unbounded):
        if value == unbounded:
            return None
        if value == -unbounded:
            return None
        return Fraction(int(value.p), int(value.q)) if getattr(value, "is_Rational", False) else False

    if solution == sympy.S.EmptySet:
        return ()
    members = solution.args if isinstance(solution, sympy.Union) else (solution,)
    pieces: list[Piece] = []
    for member in members:
        if isinstance(member, sympy.FiniteSet):
            for v in member:
                x = bound(v, sympy.oo)
                if x is None or x is False:
                    return None
                pieces.append((x, True, x, True))
        elif isinstance(member, sympy.Interval):
            lo, hi = bound(member.start, sympy.oo), bound(member.end, sympy.oo)
            if lo is False or hi is False:
                return None
            if member.start == sympy.oo or member.end == -sympy.oo:
                return None
            piece = _piece(
                None if member.start == -sympy.oo else lo,
                not member.left_open,
                None if member.end == sympy.oo else hi,
                not member.right_open,
            )
            if piece is None:
                return None
            pieces.append(piece)
        else:
            return None
    return _merge_pieces(pieces)


def _typed_pieces(text: str, variable: str) -> tuple[Piece, ...] | None:
    if len(text) > MAX_TYPED_LENGTH:
        return None
    # "x = 7" (one marked point) is not a comparison the inequality reader knows: it is "x >= 7 and x <= 7".
    equals = re.compile(rf"(?<![<>=!\w]){re.escape(variable)}\s*=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)(?![\d./])")
    text = equals.sub(rf"{variable} >= \1 and {variable} <= \1", text)
    read = read_solution_set(text)
    if read is None:
        return None
    solution, name = read
    if name is not None and name != variable:
        return None
    return _pieces_from_set(solution)


def _expected_pieces(answer) -> tuple[Piece, ...] | None:
    items = answer.get("pieces") if isinstance(answer, dict) else None
    if not isinstance(items, list) or not items or len(items) > MAX_ITEMS:
        return None
    pieces = []
    for item in items:
        if not isinstance(item, dict):
            return None
        ends = []
        for end in ("from", "to"):
            raw = item.get(end)
            if raw is None:
                ends.extend((None, False))
                continue
            value = _exact(raw)
            if value is None:
                return None
            ends.extend((value, bool(item.get(f"{end}_closed"))))
        piece = _piece(*ends)
        if piece is None:
            return None
        pieces.append(piece)
    return _merge_pieces(pieces)


def _pieces_to_set(pieces) -> sympy.Set:
    members = []
    for lo, lc, hi, hc in pieces:
        low = -sympy.oo if lo == -_INF else sympy.Rational(lo.numerator, lo.denominator)
        high = sympy.oo if hi == _INF else sympy.Rational(hi.numerator, hi.denominator)
        members.append(sympy.FiniteSet(low) if lo == hi else sympy.Interval(low, high, not lc, not hc))
    return sympy.Union(*members)


def pieces_text(pieces, variable: str = "x") -> str:
    """Pieces as the typed inequality that spells the same set: ``x < 3 or x >= 7``."""
    return set_text(_pieces_to_set(pieces), variable)


def number_line_answer(pieces, *, range_, snap=1, variable: str = "x") -> dict:
    """`Question.answer` for a ``number_line_set`` question. Each piece is ``(low, low_closed, high, high_closed)``
    with ``None`` for an unbounded end (a ray); a single point is ``(p, True, p, True)``."""
    out = []
    for lo, lc, hi, hc in pieces:
        out.append(
            {
                "from": None if lo is None else fraction_json(lo),
                "to": None if hi is None else fraction_json(hi),
                "from_closed": bool(lc),
                "to_closed": bool(hc),
            }
        )
    answer = {
        "pieces": out,
        "range": [fraction_json(v) for v in range_],
        "snap": fraction_json(snap),
        "variable": variable,
    }
    if _expected_pieces(answer) is None:
        raise ValueError("not a valid number-line answer")
    return answer


@_multi_part_aware
def grade_number_line_set(question, given: str) -> GradeResult:
    answer = _answer_dict(question)
    expected = _expected_pieces(answer)
    if expected is None:
        return _result(question, False)
    canvas = _Canvas(answer)
    variable = answer.get("variable") if isinstance(answer.get("variable"), str) else "x"
    if not isinstance(given, str) or not given.strip() or len(given) > MAX_REPLY_LENGTH:
        return _result(question, False)
    if _is_jsonish(given):
        got = _json_pieces(_load(given), canvas)
    else:
        got = _typed_pieces(given, variable)
        if got is not None and any(
            (p[0] != -_INF and not canvas.has_x(p[0])) or (p[2] != _INF and not canvas.has_x(p[2])) for p in got
        ):
            got = None
    if got is None:
        return _result(question, False)
    return _result(question, got == expected, pieces_text(got, variable) if got else "no solution")


# ---------------------------------------------------------------------------------------------- polygon


def _cross(o: Point, a: Point, b: Point) -> Fraction:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _cleaned_polygon(vertices: list[Point]) -> list[Point] | None:
    """The corners of the outline: repeated clicks, a closing repeat of the first point, and points partway along a
    straight side removed. None when fewer than three corners remain."""
    out: list[Point] = []
    for v in vertices:
        if not out or out[-1] != v:
            out.append(v)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    changed = True
    while changed and len(out) >= 3:
        changed = False
        for i in range(len(out)):
            if _cross(out[i - 1], out[i], out[(i + 1) % len(out)]) == 0:
                del out[i]
                changed = True
                break
    return out if len(out) >= 3 else None


def _cyclic_form(vertices: list[Point]) -> tuple:
    """One tuple for every spelling of the same outline: any start vertex, either direction."""
    n = len(vertices)
    spellings = []
    for seq in (vertices, vertices[::-1]):
        for i in range(n):
            spellings.append(tuple(seq[i:] + seq[:i]))
    return min(spellings)


def _json_vertices(value, canvas: _Canvas) -> list[Point] | None:
    if isinstance(value, dict):
        if not _only_keys(value, {"vertices"}):
            return None
        value = value.get("vertices")
    points = _pts(value, MAX_VERTICES + 1)
    if points is None or any(not canvas.accepts(p) for p in points):
        return None
    return points


def polygon_answer(vertices, *, x_range, y_range, snap=1, labelled: bool = True) -> dict:
    """`Question.answer` for a ``polygon`` question. ``labelled`` means vertex *i* of the reply must be vertex *i* of
    the answer (the image of A, B, C, ... is A', B', C', ...); otherwise the reply is the same outline starting
    anywhere and going either way round."""
    verts = [point_json(v) for v in vertices]
    if len(verts) < 3 or len(verts) > MAX_VERTICES:
        raise ValueError("a polygon has 3 to 12 vertices")
    return {
        "vertices": verts,
        "labelled": bool(labelled),
        "x_range": [fraction_json(v) for v in x_range],
        "y_range": [fraction_json(v) for v in y_range],
        "snap": fraction_json(snap),
    }


@_multi_part_aware
def grade_polygon(question, given: str) -> GradeResult:
    answer = _answer_dict(question)
    expected = _pts_exact(answer.get("vertices")) if answer else None
    if not expected or not 3 <= len(expected) <= MAX_VERTICES:
        return _result(question, False)
    labelled = bool(answer.get("labelled", True))
    canvas = _Canvas(answer)
    if not isinstance(given, str) or not given.strip() or len(given) > MAX_REPLY_LENGTH:
        return _result(question, False)
    loaded = _load(given) if _is_jsonish(given) else None
    if loaded is not None:
        got = _json_vertices(loaded, canvas)
    else:
        got = _typed_pts(_unbracketed(given), MAX_VERTICES + 1)
        if got is not None and any(not canvas.accepts(p) for p in got):
            got = None
    if got is None:
        return _result(question, False)
    text = " ".join(_point_text(p) for p in got)
    if labelled:
        return _result(question, got == expected, text)
    clean_got, clean_expected = _cleaned_polygon(got), _cleaned_polygon(expected)
    if clean_got is None or clean_expected is None:
        return _result(question, False, text)
    return _result(question, _cyclic_form(clean_got) == _cyclic_form(clean_expected), text)


def _pts_exact(values) -> list[Point] | None:
    if not isinstance(values, (list, tuple)) or len(values) > MAX_ITEMS:
        return None
    out = [_exact_pt(v) for v in values]
    return None if any(p is None for p in out) else out


# ---------------------------------------------------------------------------------------------- curve
#
# The target is a parabola y = a (x - h)^2 + k or an exponential y = a b^x + c. A reply is points on it, enough to
# fix it: three with different x for the parabola (or its vertex and one other point), two for the exponential.

_CURVE_FAMILIES = ("quadratic", "exponential")


def _curve_params(answer) -> dict | None:
    if not isinstance(answer, dict):
        return None
    family = answer.get("family")
    if family == "quadratic":
        a, h, k = (_exact(answer.get(n)) for n in ("a", "h", "k"))
        return None if a is None or h is None or k is None or a == 0 else {"family": family, "a": a, "h": h, "k": k}
    if family == "exponential":
        a, b, c = _exact(answer.get("a")), _exact(answer.get("b")), _exact(answer.get("c", 0))
        if a is None or b is None or c is None or a == 0 or b <= 0 or b == 1:
            return None
        return {"family": family, "a": a, "b": b, "c": c}
    return None


def _on_curve(params: dict, point: Point) -> bool:
    x, y = point
    if params["family"] == "quadratic":
        return y == params["a"] * (x - params["h"]) ** 2 + params["k"]
    if x.denominator != 1 or abs(x) > MAX_EXPONENT:
        return False  # b**x is rational only for a whole-number x, and a huge one is not on any canvas
    return y == params["a"] * params["b"] ** int(x) + params["c"]


def curve_value(params: dict, x) -> Fraction | None:
    """The curve's value at `x` (a Fraction), or None when it is not rational. Used by generators and tests."""
    x = Fraction(x)
    if params["family"] == "quadratic":
        return params["a"] * (x - params["h"]) ** 2 + params["k"]
    if x.denominator != 1 or abs(x) > MAX_EXPONENT:
        return None
    return params["a"] * params["b"] ** int(x) + params["c"]


def _json_curve(value, canvas: _Canvas, family: str):
    """``("points", [p, ...])`` or ``("vertex", vertex, point)`` from a reply object, or None."""
    if isinstance(value, list):
        value = {"points": value}
    if not isinstance(value, dict):
        return None
    if set(value) == {"points"}:
        points = _pts(value["points"], MAX_ITEMS)
        if points is None or any(not canvas.accepts(p) for p in points):
            return None
        return ("points", points)
    if set(value) == {"vertex", "point"} and family == "quadratic":
        vertex, point = _pt(value["vertex"]), _pt(value["point"])
        if vertex is None or point is None or not (canvas.accepts(vertex) and canvas.accepts(point)):
            return None
        return ("vertex", vertex, point)
    return None


_TYPED_VERTEX_RE = re.compile(r"^\s*vertex\s*(\(.*\))\s*(?:[,;]|and)?\s*point\s*(\(.*\))\s*$", re.IGNORECASE)


def _typed_curve(text: str, canvas: _Canvas, family: str):
    m = _TYPED_VERTEX_RE.match(text) if family == "quadratic" and len(text) <= MAX_TYPED_LENGTH else None
    if m:
        vertex, point = _typed_pts(m.group(1), 1), _typed_pts(m.group(2), 1)
        if vertex is None or point is None:
            return None
        return ("vertex", vertex[0], point[0]) if canvas.accepts(vertex[0]) and canvas.accepts(point[0]) else None
    points = _typed_pts(_unbracketed(text))
    return ("points", points) if points is not None and all(canvas.accepts(p) for p in points) else None


def curve_answer(family: str, *, x_range, y_range, snap=1, **params) -> dict:
    """`Question.answer` for a ``curve`` question: ``a, h, k`` for ``"quadratic"``, ``a, b, c`` (``c`` the
    asymptote, default 0) for ``"exponential"``."""
    answer = {
        "family": family,
        **{k: fraction_json(v) for k, v in params.items()},
        "x_range": [fraction_json(v) for v in x_range],
        "y_range": [fraction_json(v) for v in y_range],
        "snap": fraction_json(snap),
    }
    if _curve_params(answer) is None:
        raise ValueError("not a valid curve answer")
    return answer


@_multi_part_aware
def grade_curve(question, given: str) -> GradeResult:
    answer = _answer_dict(question)
    params = _curve_params(answer)
    if params is None:
        return _result(question, False)
    canvas = _Canvas(answer)
    if not isinstance(given, str) or not given.strip() or len(given) > MAX_REPLY_LENGTH:
        return _result(question, False)
    family = params["family"]
    loaded = _load(given) if _is_jsonish(given) else None
    reply = _json_curve(loaded, canvas, family) if loaded is not None else _typed_curve(given, canvas, family)
    if reply is None:
        return _result(question, False)
    if reply[0] == "vertex":
        _, vertex, point = reply
        text = f"vertex {_point_text(vertex)} point {_point_text(point)}"
        ok = vertex == (params["h"], params["k"]) and point[0] != vertex[0] and _on_curve(params, point)
        return _result(question, ok, text)
    points = list(dict.fromkeys(reply[1]))
    text = " ".join(_point_text(p) for p in points)
    needed = 3 if family == "quadratic" else 2
    ok = len({x for x, _ in points}) >= needed and all(_on_curve(params, p) for p in points)
    return _result(question, ok, text)


# ---------------------------------------------------------------------------------------------- bars


def _expected_bars(answer):
    if not isinstance(answer, dict):
        return None
    heights, cats = answer.get("heights"), answer.get("categories")
    if not (isinstance(heights, list) and isinstance(cats, list) and len(heights) == len(cats)):
        return None
    if not 1 <= len(heights) <= MAX_ITEMS or len(set(map(str, cats))) != len(cats):
        return None
    values = [_exact(h) for h in heights]
    return None if any(v is None or v < 0 for v in values) else (list(map(str, cats)), values)


def _bars_ok(values, answer, canvas: _Canvas) -> bool:
    top = _exact(answer.get("max")) if answer.get("max") is not None else None
    return all(v >= 0 and canvas.snapped(v) and (top is None or v <= top) for v in values)


def _json_bars(value, cats: list[str]) -> list[Fraction] | None:
    if isinstance(value, list):
        value = {"heights": value}
    if not isinstance(value, dict) or len(value) != 1:
        return None
    if "heights" in value:
        raw = value["heights"]
        if not isinstance(raw, list) or len(raw) != len(cats):
            return None
        out = [_coord(v) for v in raw]
    elif "bars" in value:
        raw = value["bars"]
        if not isinstance(raw, dict) or set(raw) != set(cats):
            return None
        out = [_coord(raw[c]) for c in cats]
    else:
        return None
    return None if any(v is None for v in out) else out


_BAR_ITEM_RE = re.compile(r"^\s*(?:(.+?)\s*[=:]\s*)?(-?\d+(?:\.\d+)?(?:/\d+)?)\s*$")


def _typed_bars(text: str, cats: list[str]) -> list[Fraction] | None:
    if len(text) > MAX_TYPED_LENGTH:
        return None
    items = [i for i in re.split(r"\s*[,;\n]\s*", text.strip()) if i]
    if len(items) != len(cats):
        return None
    labelled: dict[str, Fraction] = {}
    plain: list[Fraction] = []
    for item in items:
        m = _BAR_ITEM_RE.match(item)
        value = _coord(m.group(2)) if m else None
        if m is None or value is None:
            return None
        if m.group(1):
            labelled[m.group(1).strip()] = value
        else:
            plain.append(value)
    if labelled and not plain and set(labelled) == set(cats) and len(labelled) == len(cats):
        return [labelled[c] for c in cats]
    return plain if plain and not labelled else None


def bars_answer(categories, heights, *, top, snap=1) -> dict:
    """`Question.answer` for a ``bars`` question: bar heights in category order; ``top`` is the axis maximum."""
    answer = {
        "categories": [str(c) for c in categories],
        "heights": [fraction_json(h) for h in heights],
        "max": fraction_json(top),
        "snap": fraction_json(snap),
    }
    if _expected_bars(answer) is None:
        raise ValueError("not a valid bars answer")
    return answer


@_multi_part_aware
def grade_bars(question, given: str) -> GradeResult:
    answer = _answer_dict(question)
    expected = _expected_bars(answer)
    if expected is None:
        return _result(question, False)
    cats, want = expected
    if not isinstance(given, str) or not given.strip() or len(given) > MAX_REPLY_LENGTH:
        return _result(question, False)
    got = _json_bars(_load(given), cats) if _is_jsonish(given) else _typed_bars(given, cats)
    if got is None or not _bars_ok(got, answer, _Canvas(answer)):
        return _result(question, False)
    text = ", ".join(f"{c} = {_fraction_text(v)}" for c, v in zip(cats, got))
    return _result(question, got == want, text)


# ---------------------------------------------------------------------------------------------- edges

_LABEL = r"[A-Za-z][A-Za-z0-9_]{0,7}"
_EDGE_RE = re.compile(rf"^\s*({_LABEL})\s*(?:-+>?|→|–|—)\s*({_LABEL})\s*(?:[:=]\s*(-?\d+(?:\.\d+)?(?:/\d+)?))?\s*$")


def _edge_key(a: str, b: str, directed: bool):
    return (a, b) if directed or a <= b else (b, a)


def _expected_edges(answer):
    if not isinstance(answer, dict):
        return None
    verts, edges = answer.get("vertices"), answer.get("edges")
    if not (isinstance(verts, list) and isinstance(edges, list)) or not 2 <= len(verts) <= MAX_ITEMS:
        return None
    if not all(isinstance(v, str) and re.fullmatch(_LABEL, v) for v in verts) or len(set(verts)) != len(verts):
        return None
    directed, weighted = bool(answer.get("directed")), bool(answer.get("weighted"))
    out: dict = {}
    for edge in edges:
        if not (isinstance(edge, list) and len(edge) == (3 if weighted else 2)):
            return None
        a, b = edge[0], edge[1]
        w = _exact(edge[2]) if weighted else None
        if a not in verts or b not in verts or a == b or (weighted and w is None):
            return None
        key = _edge_key(a, b, directed)
        if key in out:
            return None
        out[key] = w
    return (verts, directed, weighted, out) if out else None


def _read_edges(items, verts, directed: bool, weighted: bool):
    """A dict key -> weight from (a, b, weight-or-None) triples, or None when any is invalid or repeats."""
    out: dict = {}
    for a, b, w in items:
        if a not in verts or b not in verts or a == b or (w is not None) != weighted:
            return None
        key = _edge_key(a, b, directed)
        if key in out:
            if out[key] != w:
                return None
            continue
        out[key] = w
    return out if out else None


def _json_edges(value, verts, directed: bool, weighted: bool):
    if isinstance(value, dict):
        if set(value) != {"edges"}:
            return None
        value = value["edges"]
    if not isinstance(value, list) or not value or len(value) > MAX_ITEMS:
        return None
    triples = []
    for edge in value:
        if not isinstance(edge, list) or len(edge) != (3 if weighted else 2):
            return None
        if not all(isinstance(e, str) for e in edge[:2]):
            return None
        w = _coord(edge[2]) if weighted else None
        if weighted and w is None:
            return None
        triples.append((edge[0], edge[1], w))
    return _read_edges(triples, verts, directed, weighted)


def _typed_edges(text: str, verts, directed: bool, weighted: bool):
    if len(text) > MAX_TYPED_LENGTH:
        return None
    items = [i for i in re.split(r"\s*[,;\n]\s*", text.strip()) if i]
    if not items or len(items) > MAX_ITEMS:
        return None
    triples = []
    for item in items:
        m = _EDGE_RE.match(item)
        if m is None:
            return None
        w = _coord(m.group(3)) if m.group(3) is not None else None
        if m.group(3) is not None and w is None:
            return None
        triples.append((m.group(1), m.group(2), w))
    return _read_edges(triples, verts, directed, weighted)


def edges_text(edges: dict, directed: bool = False) -> str:
    """Edges as typed: ``A-B, B-C`` (``A->B`` when directed, ``A-B:4`` when weighted)."""
    arrow = "->" if directed else "-"
    parts = []
    for (a, b), w in sorted(edges.items()):
        parts.append(f"{a}{arrow}{b}" + ("" if w is None else f":{_fraction_text(w)}"))
    return ", ".join(parts)


def edges_answer(vertices, edges, *, directed: bool = False, weighted: bool = False) -> dict:
    """`Question.answer` for an ``edges`` question: ``edges`` are ``[a, b]`` (``[a, b, weight]`` when weighted)."""
    answer = {
        "vertices": list(vertices),
        "edges": [[e[0], e[1], fraction_json(e[2])] if weighted else [e[0], e[1]] for e in edges],
        "directed": directed,
        "weighted": weighted,
    }
    if _expected_edges(answer) is None:
        raise ValueError("not a valid edges answer")
    return answer


@_multi_part_aware
def grade_edges(question, given: str) -> GradeResult:
    answer = _answer_dict(question)
    expected = _expected_edges(answer)
    if expected is None:
        return _result(question, False)
    verts, directed, weighted, want = expected
    if not isinstance(given, str) or not given.strip() or len(given) > MAX_REPLY_LENGTH:
        return _result(question, False)
    if _is_jsonish(given):
        got = _json_edges(_load(given), verts, directed, weighted)
    else:
        got = _typed_edges(given, verts, directed, weighted)
    if got is None:
        return _result(question, False)
    return _result(question, got == want, edges_text(got, directed))


__all__ = [
    "grade_half_planes",
    "grade_number_line_set",
    "grade_polygon",
    "grade_curve",
    "grade_bars",
    "grade_edges",
    "half_planes_answer",
    "half_plane_text",
    "number_line_answer",
    "pieces_text",
    "polygon_answer",
    "curve_answer",
    "curve_value",
    "bars_answer",
    "edges_answer",
    "edges_text",
    "fraction_json",
    "point_json",
]
