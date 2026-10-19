"""Linear inequalities and pairs of lines read from a drawn graph (Year 10A, graphical solutions).

Three topics, all on the existing ``cartesian`` figure (labelled straight lines through lattice points, labelled
points). Nothing is shaded and nothing is dashed: the region is always described in words ("on or above the line"),
so the learner reads the line's equation off the grid and tests points. Every key is a typed set, a typed pair or a
choice among six inequalities.

``y910_graph_inequality_region`` (answer: a set of point labels)

* easy: one line, its equation in the stem, five labelled points on the grid; which lie in the region
  ``y > mx + c`` (or ``<``, ``>=``, ``<=``). One point is on the line, so the strict/inclusive rule decides it.
* medium: one line, no equation given; six points are listed by coordinates, some far off the grid, so the
  learner must read ``y = mx + c`` and test each point against it.
* hard: two lines; the region is "on or above A and strictly below B"; six listed points include ones that meet
  only one condition.

``y910_graph_system_solution`` (answer: the point where two lines meet, a pair)

* easy: two lines cross at a grid intersection that can be read straight off the graph.
* medium: the lines cross between grid lines (a half-integer point), so each equation must be read and the pair
  solved.
* hard: one line is drawn, the other is described by a gradient and a point; they meet outside the grid.

``y910_graph_inequality_match`` (answer: one of six inequalities)

A line is drawn; the stem says which side of it a stated point is on and whether the boundary is included.

Each key is computed exactly with fractions and checked a second way (SymPy relationals, or substitution in both
equations). The figure readers registered here re-derive each answer from the drawn lines alone (plus the numbers
the stem states).
"""

from __future__ import annotations

import re
from fractions import Fraction as F

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, phrase, pick_distractors
from mathgen.generators.fg1_line_reading_build import eq_text
from mathgen.generators.pv1a_line_figure_build import Window, lattice_points, line_figure, read_intersection, read_line
from mathgen.grading.typed import set_answer_text

TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in TIERS}
_ALT = "straight lines on a coordinate grid"
_POINT_LABELS = "PQRSTU"
_OPEN_SET = "Write the letters of those points in braces."


def _rhs(m: F, c: F) -> str:
    """The right-hand side of ``y = mx + c`` as a student writes it."""
    return eq_text(m, c)[4:]


def _frac_pair(x: F, y: F) -> tuple[str, dict]:
    def one(v: F) -> str:
        v = F(v)
        if v.denominator == 1:
            return str(v.numerator)
        check(v.denominator == 2, "intersections are at whole or half numbers")
        return f"{float(v):.1f}"

    return f"({one(x)}, {one(y)})", {
        "x": float(x) if F(x).denominator != 1 else int(x),
        "y": float(y) if F(y).denominator != 1 else int(y),
    }


def _as_value(v: F):
    return int(v) if F(v).denominator == 1 else float(v)


# ------------------------------------------------------------------------------------------ drawing helpers


def _window(rng) -> Window:
    nu, nv = rng.randint(9, 10), rng.randint(9, 10)
    u0 = rng.randint(-(nu - 3), -2)
    v0 = rng.randint(-(nv - 3), -2)
    return Window(u0, u0 + nu - 1, v0, v0 + nv - 1)


def _two_points(win: Window, m: F, c: F):
    pts = lattice_points(win, m, c)
    pairs = [(a, b) for i, a in enumerate(pts) for b in pts[i + 1 :] if b[0] - a[0] >= 2]
    return pairs


def _line_in_window(win: Window, m: F, c: F) -> bool:
    return bool(_two_points(win, m, c))


def _inside(win: Window, x: F, y: F, margin: int = 1) -> bool:
    return win.u0 + margin <= x <= win.u1 - margin and win.v0 + margin <= y <= win.v1 - margin


_SLOPES_EASY = [F(s) for s in (-3, -2, -1, 1, 2, 3)]
_SLOPES_ALL = _SLOPES_EASY + [F(1, 2), F(-1, 2), F(3, 2), F(-3, 2)]


def _one_line(rng, slopes, c_range=(-4, 4)):
    for _ in attempts(1000, "a line and a window"):
        win = _window(rng)
        m = rng.choice(slopes)
        c = F(rng.randint(*c_range))
        if c == 0 or not win.shows_y_axis:
            continue
        if _line_in_window(win, m, c) and len(lattice_points(win, m, c)) >= 3:
            return win, m, c
    raise AssertionError  # pragma: no cover


def _draw_one(win, m, c, rng, label="L", points=()):
    a, b = rng.choice(_two_points(win, m, c))
    return line_figure(win, [(a, b)], _ALT, labels=[label], points=points)


def _two_lines(rng, kind: str):
    """Two labelled lines with integer gradients and intercepts that meet inside the grid.

    kind "lattice": at a grid intersection. kind "half": at a half-integer point."""
    for _ in attempts(5000, "two lines that meet in the window"):
        win = _window(rng)
        ma, mb = rng.sample(_SLOPES_EASY + [F(0)] * 0, 2)
        ca, cb = F(rng.randint(-5, 5)), F(rng.randint(-5, 5))
        if ma == mb or ca == cb:
            continue
        x = (cb - ca) / (ma - mb)
        y = ma * x + ca
        if kind == "lattice" and not (x.denominator == 1 and y.denominator == 1):
            continue
        if kind == "half":
            if x.denominator != 2 and y.denominator != 2:
                continue
            if x.denominator not in (1, 2) or y.denominator not in (1, 2):
                continue
        if not _inside(win, x, y, 2):
            continue
        if not (_line_in_window(win, ma, ca) and _line_in_window(win, mb, cb)):
            continue
        if abs(ma - mb) == 0:
            continue
        return win, (ma, ca), (mb, cb), (x, y)
    raise AssertionError  # pragma: no cover


# ------------------------------------------------------------------------------------------ region (set) topic


def _relation(op: str, lhs: F, rhs: F) -> bool:
    return {">": lhs > rhs, ">=": lhs >= rhs, "<": lhs < rhs, "<=": lhs <= rhs}[op]


_OP_WORDS = {
    ">": "strictly above",
    ">=": "on or above",
    "<": "strictly below",
    "<=": "on or below",
}
_OP_SIGN = {">": ">", ">=": "≥", "<": "<", "<=": "≤"}


def _cond(m: F, c: F, op: str, point) -> bool:
    return _relation(op, F(point[1]), m * F(point[0]) + c)


def _integer_cond(m: F, c: F, op: str, point) -> bool:
    """Second route: clear the denominators and compare whole numbers (no fractions involved)."""
    d = m.denominator * c.denominator
    lhs = point[1] * d
    rhs = (m * d).numerator * point[0] + (c * d).numerator
    return {">": lhs > rhs, ">=": lhs >= rhs, "<": lhs < rhs, "<=": lhs <= rhs}[op]


def _letters(points: dict, test) -> list[str]:
    return [label for label, p in points.items() if test(p)]


def _set_candidates(points, key, tests):
    """Wrong sets as (letters, note). `tests` maps a note to a test function giving a wrong membership."""
    out = []
    for note, fn in tests:
        letters = _letters(points, fn)
        if letters and letters != key and len(letters) < len(points) + 1:
            out.append((letters, note))
    return out


def _set_distractors(key: list[str], candidates, labels: str):
    answer_text = set_answer_text(key)
    order = {c: i for i, c in enumerate(labels)}
    made = []
    for letters, note in candidates:
        if len(letters) == len(labels):
            continue  # "all of them" is never a believable wrong answer
        letters = sorted(letters, key=lambda s: order[s])
        made.append(Distractor(set_answer_text(letters), note, letters))
    return answer_text, pick_distractors(answer_text, made, 3, unordered=True)


def _easy_region(rng):
    for _ in attempts(1000, "an easy region question"):
        win, m, c = _one_line(rng, _SLOPES_EASY)
        op = rng.choice(list(_OP_WORDS))
        on_line = rng.choice(lattice_points(win, m, c))
        spots = [
            (u, v)
            for u in range(win.u0, win.u1 + 1)
            for v in range(win.v0, win.v1 + 1)
            if abs(v - (m * u + c)) >= 1 and (u, v) != on_line and v != m * u + c
        ]
        if len(spots) < 6:
            continue
        chosen = rng.sample(spots, 4) + [on_line]
        us = [p[0] for p in chosen]
        if len(set(us)) < 4:
            continue
        rng.shuffle(chosen)
        points = dict(zip(_POINT_LABELS, chosen))
        key = _letters(points, lambda p: _cond(m, c, op, p))
        check(key == _letters(points, lambda p: _integer_cond(m, c, op, p)), "the two membership routes disagree")
        if not 1 <= len(key) <= 3:
            continue
        flip_op = {">": ">=", ">=": ">", "<": "<=", "<=": "<"}[op]
        opposite = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}[op]
        cands = _set_candidates(
            points,
            key,
            [
                ("boundary-point-handled-the-wrong-way", lambda p: _cond(m, c, flip_op, p)),
                ("region-on-the-wrong-side-of-the-line", lambda p: _cond(m, c, opposite, p)),
                (
                    "region-on-the-wrong-side-of-the-line",
                    lambda p: _cond(m, c, {">": "<", ">=": "<=", "<": ">", "<=": ">="}[op], p),
                ),
                (
                    "coordinates-read-the-wrong-way-round",
                    lambda p: _cond(m, c, op, (p[1], p[0])) if _inside(win, F(p[1]), F(p[0]), 0) else False,
                ),
            ],
        )
        answer_text, distractors = _set_distractors(key, cands, _POINT_LABELS[:5])
        if len(distractors) < 3:
            continue
        break
    fig = _draw_one(
        win,
        m,
        c,
        rng,
        points=[{"at": [p[0], p[1]], "label": lab} for lab, p in points.items()],
    )
    sign = _OP_SIGN[op]
    inequality = f"y {sign} {_rhs(m, c)}"
    text = (
        f"The line L is the graph of {eq_text(m, c)}. The five points P, Q, R, S and T are marked on the grid. "
        f"Which of the points satisfy the inequality {inequality}? {_OPEN_SET}"
    )
    working = [
        f"{lab} = ({p[0]}, {p[1]}): the line has y = {m * p[0] + c} at x = {p[0]}, so the test y {sign} {m * p[0] + c} is {'true' if lab in key else 'false'} for this point."
        for lab, p in points.items()
    ]
    return dict(
        text=text,
        answer=key,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
        figure=fig,
        answer_kind="set",
    )


def _list_points(points: dict) -> str:
    return "; ".join(f"{lab} ({p[0]}, {p[1]})" for lab, p in points.items())


def _medium_region(rng):
    for _ in attempts(2000, "a medium region question"):
        win, m, c = _one_line(rng, _SLOPES_ALL)
        op = rng.choice(list(_OP_WORDS))
        # one point on the line, others within a few units of it, two beyond the grid
        xs = rng.sample(range(-9, 10), 6)
        on_x = next((x for x in xs if (m * x + c).denominator == 1), None)
        if on_x is None:
            continue
        chosen = []
        for x in xs:
            base = m * x + c
            if x == on_x:
                chosen.append((x, int(base)))
            else:
                off = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                chosen.append((x, int(base) + off if base.denominator == 1 else int(base // 1) + off))
        if len(set(chosen)) < 6:
            continue
        if sum(1 for p in chosen if not (win.u0 <= p[0] <= win.u1 and win.v0 <= p[1] <= win.v1)) < 2:
            continue
        rng.shuffle(chosen)
        points = dict(zip(_POINT_LABELS, chosen))
        key = _letters(points, lambda p: _cond(m, c, op, p))
        check(key == _letters(points, lambda p: _integer_cond(m, c, op, p)), "the two membership routes disagree")
        if not 2 <= len(key) <= 4:
            continue
        flip_op = {">": ">=", ">=": ">", "<": "<=", "<=": "<"}[op]
        cands = _set_candidates(
            points,
            key,
            [
                ("boundary-point-handled-the-wrong-way", lambda p: _cond(m, c, flip_op, p)),
                ("region-on-the-wrong-side-of-the-line", lambda p: not _cond(m, c, op, p) and not _on(m, c, p)),
                ("gradient-sign-misread", lambda p: _cond(-m, c, op, p)),
                ("intercept-sign-error", lambda p: _cond(m, -c, op, p)),
                ("coordinates-read-the-wrong-way-round", lambda p: _cond(m, c, op, (p[1], p[0]))),
            ],
        )
        answer_text, distractors = _set_distractors(key, cands, _POINT_LABELS)
        if len(distractors) < 3:
            continue
        break
    fig = _draw_one(win, m, c, rng)
    where = _OP_WORDS[op]
    text = (
        f"The line L is drawn on the grid and continues in both directions. A region is made up of the points "
        f"{where} L. Six points are given: {_list_points(points)}. "
        f"Which of the points lie in the region? {_OPEN_SET}"
    )
    mm, cc = m, c
    working = [
        f"Read two points from the grid: the line is {eq_text(mm, cc)}.",
        *[
            f"{lab} = ({p[0]}, {p[1]}): at x = {p[0]} the line has y = {mm * p[0] + cc}, and {p[1]} is {_side(mm, cc, p)} that."
            for lab, p in points.items()
        ],
    ]
    return dict(
        text=text,
        answer=key,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
        figure=fig,
        answer_kind="set",
    )


def _on(m, c, p) -> bool:
    return F(p[1]) == m * F(p[0]) + c


def _side(m, c, p) -> str:
    v = F(p[1]) - (m * F(p[0]) + c)
    return "on" if v == 0 else ("above" if v > 0 else "below")


def _hard_region(rng):
    for _ in attempts(5000, "a hard region question"):
        win, (ma, ca), (mb, cb), (xi, yi) = _two_lines(rng, "lattice")
        if ma == mb:
            continue
        op_a = rng.choice([">=", ">"])
        op_b = rng.choice(["<", "<="])
        xs = rng.sample(range(-6, 7), 7)
        points = []
        for x in xs:
            band = rng.randint(-4, 4)
            line = rng.choice([(ma, ca), (mb, cb)])
            y = (
                int(line[0] * x + line[1] + band)
                if (line[0] * x + line[1]).denominator == 1
                else int((line[0] * x + line[1]) // 1) + band
            )
            points.append((x, y))
        # one point exactly on each line, to make the boundary rule matter
        xa = next(
            (x for x in range(-8, 9) if (ma * x + ca).denominator == 1 and x not in [p[0] for p in points[:5]]), 0
        )
        points = points[:5] + [(xa, int(ma * xa + ca))]
        xb = next((x for x in range(8, -9, -1) if (mb * x + cb).denominator == 1 and x != xa), 1)
        points = points[:6]
        points[0] = (xb, int(mb * xb + cb))
        if len(set(points)) < 6 or any(abs(p[1]) > 18 for p in points):
            continue
        rng.shuffle(points)
        pts = dict(zip(_POINT_LABELS, points))

        def both(p):
            return _cond(ma, ca, op_a, p) and _cond(mb, cb, op_b, p)

        key = _letters(pts, both)
        second = _letters(pts, lambda p: _integer_cond(ma, ca, op_a, p) and _integer_cond(mb, cb, op_b, p))
        check(key == second, "the two membership routes disagree")
        only_a = _letters(pts, lambda p: _cond(ma, ca, op_a, p))
        only_b = _letters(pts, lambda p: _cond(mb, cb, op_b, p))
        either = _letters(pts, lambda p: _cond(ma, ca, op_a, p) or _cond(mb, cb, op_b, p))
        if not 1 <= len(key) <= 3:
            continue
        cands = [
            (only_a, "only-one-condition-checked"),
            (only_b, "only-one-condition-checked"),
            (either, "either-condition-taken-for-both"),
            (
                _letters(
                    pts,
                    lambda p: (
                        _cond(ma, ca, {">": ">=", ">=": ">"}[op_a], p)
                        and _cond(mb, cb, {"<": "<=", "<=": "<"}[op_b], p)
                    ),
                ),
                "boundary-point-handled-the-wrong-way",
            ),
            (
                _letters(pts, lambda p: _cond(ma, ca, {">": "<=", ">=": "<"}[op_a], p) and _cond(mb, cb, op_b, p)),
                "region-on-the-wrong-side-of-the-line",
            ),
        ]
        cands = [(letters, note) for letters, note in cands if letters and letters != key]
        answer_text, distractors = _set_distractors(key, cands, _POINT_LABELS)
        if len(distractors) < 3:
            continue
        break
    fig = line_figure(
        win,
        [_two_points(win, ma, ca)[0], _two_points(win, mb, cb)[0]],
        _ALT,
        labels=["A", "B"],
    )
    text = (
        f"The lines A and B are drawn on the grid and continue in both directions. A region is made up of the points "
        f"{_OP_WORDS[op_a]} A that are also {_OP_WORDS[op_b]} B. Six points are given: {_list_points(pts)}. "
        f"Which of the points lie in the region? {_OPEN_SET}"
    )
    working = [
        f"Read the lines: A is {eq_text(ma, ca)} and B is {eq_text(mb, cb)}.",
        *[
            f"{lab} = ({p[0]}, {p[1]}): {_side(ma, ca, p)} A and {_side(mb, cb, p)} B, so it is {'in' if lab in key else 'not in'} the region."
            for lab, p in pts.items()
        ],
    ]
    return dict(
        text=text,
        answer=key,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
        figure=fig,
        answer_kind="set",
    )


_REGION = {"easy": _easy_region, "medium": _medium_region, "hard": _hard_region}


@generator("y910_graph_inequality_region", level=10, presets=_PRESETS)
def y910_graph_inequality_region(rng, tier="easy"):
    """Which points lie in a region bounded by drawn lines; see the module docstring."""
    check(tier in _REGION, f"unknown tier {tier!r}")
    d = _REGION[tier](rng)
    figures.assert_no_leak(d["figure"], [d["answer_text"], *d["distractors"]])
    return d


# ---- the reader: re-derives the set from the drawn lines and the points (figure or stem)

_POINT_RE = re.compile(r"([P-U]) \((-?\d+), (-?\d+)\)")
_OP_BY_WORDS = {words: op for op, words in _OP_WORDS.items()}
_WORDS_RE = re.compile(r"(strictly above|on or above|strictly below|on or below)")


def _read_region(question) -> str:
    fig, text = question.figure, question.text
    pts = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in _POINT_RE.finditer(text)}
    if not pts:  # easy: the points are marked on the grid
        pts = {p["label"]: tuple(p["at"]) for p in fig.get("points", ())}
    words = _WORDS_RE.findall(text)
    if words:
        ops = [_OP_BY_WORDS[w] for w in words]
    else:
        ineq = re.search(r"inequality y ([<>≤≥]) ", text)
        ops = [{"≥": ">=", "≤": "<=", ">": ">", "<": "<"}[ineq.group(1)]]
    lines = [read_line(fig, i) for i in range(len(fig["lines"]))]
    members = [
        lab for lab, p in pts.items() if all(_relation(op, F(p[1]), m * F(p[0]) + c) for op, (m, c) in zip(ops, lines))
    ]
    return set_answer_text(members)


figures.register_reader("y910_graph_inequality_region", _read_region)


# ------------------------------------------------------------------------------------------ systems (pair) topic


def _pair_wrong(wrongs, key_xy):
    """Distractors for a pair answer from (x, y, note) triples."""
    out = []
    for x, y, note in wrongs:
        x, y = F(x), F(y)
        if x.denominator not in (1, 2) or y.denominator not in (1, 2):
            continue
        text, value = _frac_pair(x, y)
        out.append(Distractor(text, note, value))
    key_text, _ = _frac_pair(*key_xy)
    return pick_distractors(key_text, out, 3, tolerance=GRADER_TOLERANCE)


def _system_result(text, key_xy, wrongs, working, fig):
    key_text, value = _frac_pair(*key_xy)
    return dict(
        text=text,
        answer=value,
        answer_text=key_text,
        answer_latex=key_text,
        working=working,
        distractors=_pair_wrong(wrongs, key_xy),
        figure=fig,
        answer_kind="pair",
    )


_SOLVE_ASK = [
    "Find the point where the lines A and B cross, as (x, y).",
    "The lines A and B are drawn on the grid. What is the solution of the simultaneous equations given by the two lines? Write it as (x, y).",
    "Solve the pair of simultaneous equations whose graphs are the lines A and B. Write the solution as (x, y).",
]


def _easy_system(rng):
    for _ in attempts(500, "an easy system"):
        win, (ma, ca), (mb, cb), (x, y) = _two_lines(rng, "lattice")
        wrongs = [
            (y, x, "x-and-y-swapped"),
            (0, ca, "read-an-intercept-as-the-solution"),
            (0, cb, "read-an-intercept-as-the-solution"),
            (x, 0, "read-an-intercept-as-the-solution"),
            (x + 1, y, "misread-the-crossing-by-one-square"),
            (x, y + 1, "misread-the-crossing-by-one-square"),
        ]
        res = _system_result(
            phrase(rng, _SOLVE_ASK),
            (x, y),
            wrongs,
            [
                f"Read the lines: A is {eq_text(ma, ca)} and B is {eq_text(mb, cb)}.",
                f"They cross at ({x}, {y}): at x = {x}, line A gives y = {ma * x + ca} and line B gives y = {mb * x + cb}.",
            ],
            line_figure(win, [_two_points(win, ma, ca)[0], _two_points(win, mb, cb)[0]], _ALT, labels=["A", "B"]),
        )
        if len(res["distractors"]) >= 3:
            return res
    raise AssertionError  # pragma: no cover


def _medium_system(rng):
    for _ in attempts(500, "a medium system"):
        win, (ma, ca), (mb, cb), (x, y) = _two_lines(rng, "half")
        if x.denominator == 1 and y.denominator == 1:
            continue
        wrongs = [
            (
                round(float(x)) if x.denominator == 2 else x,
                round(float(y)) if y.denominator == 2 else y,
                "rounded-the-crossing-to-a-grid-point",
            ),
            (y, x, "x-and-y-swapped"),
            *(
                [((cb - ca) / (ma + mb), ma * ((cb - ca) / (ma + mb)) + ca, "sign-error-when-setting-equal")]
                if ma + mb
                else []
            ),
            ((ca - cb) / (ma - mb), ma * ((ca - cb) / (ma - mb)) + ca, "sign-error-when-setting-equal"),
            (x, mb * x + ca, "mixed-the-two-equations"),
        ]
        res = _system_result(
            phrase(rng, _SOLVE_ASK) + " The lines cross between grid lines, so read each equation and solve.",
            (x, y),
            wrongs,
            [
                f"Read the lines: A is {eq_text(ma, ca)} and B is {eq_text(mb, cb)}.",
                f"Set them equal: {_rhs(ma, ca)} = {_rhs(mb, cb)}, so x = {x}.",
                f"Putting x = {x} into line A gives y = {y}.",
            ],
            line_figure(win, [_two_points(win, ma, ca)[0], _two_points(win, mb, cb)[0]], _ALT, labels=["A", "B"]),
        )
        if len(res["distractors"]) >= 3:
            return res
    raise AssertionError  # pragma: no cover


def _hard_system(rng):
    for _ in attempts(5000, "a hard system"):
        win = _window(rng)
        ma = rng.choice(_SLOPES_EASY)
        ca = F(rng.randint(-5, 5))
        if ca == 0 or not _line_in_window(win, ma, ca):
            continue
        mb = rng.choice([m for m in _SLOPES_EASY if m != ma])
        x = F(rng.choice([v for v in range(-14, 15) if v < win.u0 - 1 or v > win.u1 + 1]))
        y = ma * x + ca
        if abs(y) > 40:
            continue
        p = rng.randint(win.u0 + 1, win.u1 - 1)
        q = y + mb * (p - x)
        if abs(q) > 30 or (p, q) == (0, 0):
            continue
        cb = q - mb * p
        # the second line described in words; the sign of its equation matters
        wrongs = []
        for note, (m2, c2) in [
            ("sign-error-on-point-substitution", (mb, q + mb * p)),
            ("point-used-as-the-intercept", (mb, q)),
            ("gradient-sign-misread", (-mb, q + mb * p)),
        ]:
            if m2 != ma:
                x2 = (c2 - ca) / (ma - m2)
                wrongs.append((x2, ma * x2 + ca, note))
        wrongs.append((x, ca, "y-replaced-by-the-intercept"))
        wrongs.append((y, x, "x-and-y-swapped"))
        res = _system_result(
            phrase(
                rng,
                (
                    "The line A is drawn on the grid and continues in both directions. The line B has gradient {mb} and passes through the point ({p}, {q}). "
                    "Find the point where A and B cross, as (x, y).",
                    "Line A is drawn on the grid and extends in both directions. Line B has gradient {mb} and passes through the point ({p}, {q}). "
                    "Where do the two lines meet? Give the point as (x, y).",
                    "On the grid, the line A continues in both directions. A second line, B, has gradient {mb} and passes through the point ({p}, {q}). "
                    "What are the coordinates (x, y) of the point where A and B intersect?",
                ),
                mb=mb,
                p=p,
                q=q,
            ),
            (x, y),
            wrongs,
            [
                f"Read line A: {eq_text(ma, ca)}.",
                f"Line B passes through ({p}, {q}) with gradient {mb}, so {eq_text(mb, cb)}.",
                f"Set equal: {_rhs(ma, ca)} = {_rhs(mb, cb)} gives x = {x}, and y = {y}.",
            ],
            line_figure(win, [_two_points(win, ma, ca)[0]], _ALT, labels=["A"]),
        )
        if len(res["distractors"]) >= 3:
            return res
    raise AssertionError  # pragma: no cover


_SYSTEM = {"easy": _easy_system, "medium": _medium_system, "hard": _hard_system}


@generator("y910_graph_system_solution", level=10, presets=_PRESETS)
def y910_graph_system_solution(rng, tier="easy"):
    """Where two lines cross, read from a graph; see the module docstring."""
    check(tier in _SYSTEM, f"unknown tier {tier!r}")
    d = _SYSTEM[tier](rng)
    figures.assert_no_leak(d["figure"], [d["answer_text"], *d["distractors"]])
    return d


_LINE_B_RE = re.compile(r"gradient (-?\d+) and passes through the point \((-?\d+), (-?\d+)\)")


def _read_system(question) -> str:
    fig = question.figure
    if len(fig["lines"]) == 2:
        x, y = read_intersection(fig)
    else:
        m2, p, q = (F(int(g)) for g in _LINE_B_RE.search(question.text).groups())
        ma, ca = read_line(fig, 0)
        cb = q - m2 * p
        x = (cb - ca) / (ma - m2)
        y = ma * x + ca
    return _frac_pair(x, y)[0]


figures.register_reader("y910_graph_system_solution", _read_system)


# ------------------------------------------------------------------------------------------ matching topic


def _ineq(m: F, c: F, op: str) -> str:
    return f"y {_OP_SIGN[op]} {_rhs(m, c)}"


@generator("y910_graph_inequality_match", level=10, presets=_PRESETS)
def y910_graph_inequality_match(rng, tier="easy"):
    """Which of six inequalities describes the region on a stated side of a drawn line.

    easy: whole-number gradient, and the test point is on the grid shown; medium: fractional gradients as well;
    hard: the test point is well off the grid, so the side has to be found from the equation."""
    check(tier in TIERS, f"unknown tier {tier!r}")
    for _ in attempts(1000, "an inequality to match"):
        win, m, c = _one_line(rng, _SLOPES_EASY if tier == "easy" else _SLOPES_ALL)
        side = rng.choice(["above", "below"])
        include = rng.random() < 0.5
        # a test point clearly on the stated side, possibly off the grid
        for _ in attempts(200, "a test point"):
            px = (
                rng.randint(win.u0 + 1, win.u1 - 1)
                if tier == "easy"
                else rng.randint(-8, 8)
                if tier == "medium"
                else rng.choice([-1, 1]) * rng.randint(10, 14)
            )
            base = m * px + c
            off = rng.randint(2, 3 if tier == "easy" else 6) * (1 if side == "above" else -1)
            py = (int(base) if base.denominator == 1 else int(base // 1)) + off
            if (px, py) != (0, 0):
                break
        op = ({"above": ">=", "below": "<="} if include else {"above": ">", "below": "<"})[side]
        check(_cond(m, c, op, (px, py)) and _integer_cond(m, c, op, (px, py)), "the test point is not in the region")
        other = {">=": "<=", "<=": ">=", ">": "<", "<": ">"}
        strict = {">=": ">", "<=": "<", ">": ">=", "<": "<="}
        key = _ineq(m, c, op)
        cands = [
            (_ineq(m, c, other[op]), "region-on-the-wrong-side-of-the-line"),
            (_ineq(m, c, strict[op]), "boundary-point-handled-the-wrong-way"),
            (_ineq(m, c, other[strict[op]]), "region-on-the-wrong-side-of-the-line"),
            (_ineq(m, -c, op), "intercept-sign-error"),
            (_ineq(-m, c, op), "gradient-sign-misread"),
            (_ineq(m, c, strict[op]).replace("y", "y"), "boundary-point-handled-the-wrong-way"),
        ]
        seen = {key}
        texts: list[tuple[str, str]] = []
        for t, note in cands:
            if t not in seen:
                seen.add(t)
                texts.append((t, note))
        if len(texts) < 5:
            continue
        break
    rng.shuffle(texts)
    notes = {t: n for t, n in texts[:5]}
    boundary = (
        "The boundary line is included in the region."
        if include
        else "The boundary line is not included in the region."
    )
    fig = _draw_one(win, m, c, rng)
    text = phrase(
        rng,
        (
            "The line L is drawn on the grid. A region lies on the same side of L as the point ({px}, {py}). "
            "{boundary} Which inequality describes the region?",
            "The grid shows the line L. A region is bounded by L and contains the point ({px}, {py}) on one side of it. "
            "{boundary} Which of the inequalities describes the region?",
            "Look at the line L on the grid. Take the region on the same side of L as the point ({px}, {py}). "
            "{boundary} Choose the inequality that gives this region.",
        ),
        px=px,
        py=py,
        boundary=boundary,
    )
    working = [
        f"Read the line: L is {eq_text(m, c)}.",
        f"At x = {px} the line has y = {m * px + c}, and {py} is {_side(m, c, (px, py))} that, so the region is {side} L, which means y {'>' if side == 'above' else '<'} the line.",
        f"{'Included' if include else 'Excluded'} boundary uses {_OP_SIGN[op]}: {key}.",
    ]
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=f"plain:{key}",
        working=working,
        distractors=[Distractor(t, notes[t]) for t in notes],
        figure=fig,
        answer_kind="mcq",
    )


_MATCH_POINT_RE = re.compile(r"point \((-?\d+), (-?\d+)\)")


def _read_match(question) -> str:
    fig, text = question.figure, question.text
    px, py = (int(g) for g in _MATCH_POINT_RE.search(text).groups())
    m, c = read_line(fig)
    above = F(py) > m * px + c
    included = "is included" in text
    op = (">=" if included else ">") if above else ("<=" if included else "<")
    return _ineq(m, c, op)


figures.register_reader("y910_graph_inequality_match", _read_match)
