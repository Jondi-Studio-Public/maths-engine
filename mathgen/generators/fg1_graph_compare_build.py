"""Read lines, curves and number lines in order to solve or compare. Four topics; all read from the figure.

- ``fg1_solve_graphically`` (skill ``solve-linear-equations-graphically``, AC9M8A02): the solution of a
  linear equation, read off the graph, an mcq over ``x = k``. Easy: two labelled lines, solve
  ``f(x) = g(x)`` at their crossing. Medium: one line, solve ``f(x) = k`` for a stated k. Hard: three
  labelled lines, solve for the named pair (the other crossings are the traps).
- ``fg1_compare_linear`` (skill ``compare-linear-representations``, AC9M8A02): relationship A is a graph,
  relationship B an equation or a list of (x, y) pairs; an mcq over A, B or "the same". The class of the
  answer is drawn first and uniformly, and in an unequal class the intercepts point at the wrong
  relationship. Easy: greater gradient, B an equation. Medium: greater gradient, B a list of pairs, any
  sign. Hard: which has the greater y-value at a far x (positive or negative), so neither the gradient
  nor the intercept alone decides it.
- ``fg1_compare_functions`` (skill ``compare-functions-representations``, AC9M10A05 area): function f is
  a graph, g is a verbal rule, a vertex-form equation or a list of pairs. Easy: line f against a verbal
  rule, greater output at a stated input. Medium: parabola f against a vertex-form g, greater maximum
  value. Hard: parabola f against a linear g given as pairs, greater value at a stated x beyond the pairs.
- ``fg1_inequality_number_line`` (skill ``inequality-solutions-number-line``, AC9M10A02): a shown
  number line, an mcq over inequalities written as text. Easy: a ray, four options (open or closed end,
  left or right). Medium: a shaded interval, four endpoint combinations. Hard: each option is an
  unsolved inequality ``ax + b < c`` and the learner solves it (reversing the sign for a negative a) to
  find the one whose solution set is shown.

Each curve or line is built from lattice points. Every reader re-derives the answer from ``question.figure``
and the numbers the prompt states, never from the key.
"""

from __future__ import annotations

import re
from fractions import Fraction as F

from mathgen import figures
from mathgen.core import check, fmt_signed, generator, paren_negative, phrase, rational_text
from mathgen.generators.fg1_line_reading_build import (
    ATT,
    draw_line,
    eq_text,
    mcq_result,
    no_leak,
    on_line,
    parse_pairs,
    pairs_text,
    pick_pair,
    real_mc,
    real_pt,
    take_distractors,
)
from mathgen.generators.pv1a_line_figure_build import Window, line_figure, read_line

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_ALT_LINES = "straight lines on a coordinate grid"
_ALT_LINE = "a straight line on a coordinate grid"
_ALT_CURVE = "a parabola on a coordinate grid"
_ALT_NUMBER_LINE = "a number line"


def _x_eq(v) -> str:
    return f"x = {rational_text(v)}"


def _window(rng) -> Window:
    """A ten-tick square window with both axes strictly inside."""
    u0, v0 = rng.randint(-5, -4), rng.randint(-5, -4)
    return Window(u0, u0 + 9, v0, v0 + 9)


def _lattice_through(win: Window, M: int, C: int, first):
    """A second lattice point on ``v = M u + C`` in the window, as near ``first`` as possible."""
    pts = [p for p in ((first[0] + d, M * (first[0] + d) + C) for d in (1, -1, 2, -2)) if win.u0 <= p[0] <= win.u1]
    pts = [p for p in pts if win.v0 <= p[1] <= win.v1]
    return pts[0] if pts else None


# ----------------------------------------------------------------------------- solve graphically

_S_OPEN = {
    "easy": [
        "Line A is the graph of y = f(x) and line B is the graph of y = g(x).",
        "The graphs of y = f(x) and y = g(x) are the lines A and B on the grid.",
        "On the grid, line A shows y = f(x) and line B shows y = g(x).",
    ],
    "medium": [
        "The graph of y = f(x) is a straight line.",
        "A straight line is the graph of y = f(x).",
        "The line on the grid is the graph of y = f(x).",
    ],
    "hard": [
        "Lines A, B and C are the graphs of y = f(x), y = g(x) and y = h(x).",
        "On the grid, line A shows y = f(x), line B shows y = g(x) and line C shows y = h(x).",
    ],
}
_S_ASK = {
    "easy": [
        "Use the graphs to solve f(x) = g(x).",
        "Solve f(x) = g(x) using the graphs.",
        "Use the graphs to find x when f(x) = g(x).",
    ],
    "medium": [
        "Use the graph to solve f(x) = {k}.",
        "Solve f(x) = {k} using the graph.",
        "Use the graph to find x when f(x) = {k}.",
    ],
    "hard": [
        "Use the graphs to solve {p} = {q}.",
        "Solve {p} = {q} using the graphs.",
        "Use the graphs to find x when {p} = {q}.",
    ],
}


def _solve_easy(rng, notes):
    for _ in range(ATT):
        win = _window(rng)
        xi, yi = rng.randint(-3, 3), rng.randint(-3, 3)
        m1, m2 = rng.sample([-3, -2, -1, 1, 2, 3], 2)
        c1, c2 = yi - m1 * xi, yi - m2 * xi
        p2s = [_lattice_through(win, m, c, (xi, yi)) for m, c in ((m1, c1), (m2, c2))]
        if None in p2s or not (win.u0 < xi < win.u1 and win.v0 < yi < win.v1):
            continue
        fig = figures.cartesian(
            x_range=win.x_range,
            y_range=win.y_range,
            alt=_ALT_LINES,
            grid=1,
            lines=[
                {"through": [[xi, yi], list(p2s[0])], "label": "A"},
                {"through": [[xi, yi], list(p2s[1])], "label": "B"},
            ],
        )
        key = _x_eq(xi)
        xa, xb = F(-c1, m1), F(-c2, m2)
        cands = [
            (_x_eq(yi), "reads-the-y-value-instead-of-x"),
            (_x_eq(-xi), "sign-error-on-the-solution"),
            (_x_eq(xi + rng.choice((-1, 1))), "off-by-one-gridline"),
            (_x_eq(c1), "reads-where-a-line-meets-the-y-axis"),
            (_x_eq(c2), "reads-where-a-line-meets-the-y-axis"),
        ]
        cands += [(_x_eq(v), "reads-where-a-line-meets-the-x-axis") for v in (xa, xb) if v.denominator == 1]
        texts, note_map = take_distractors(rng, key, cands)
        if len(texts) < 3:
            continue
        try:
            no_leak(fig, key, texts)
        except AssertionError:
            continue
        text = f"{phrase(rng, _S_OPEN['easy'])} {phrase(rng, _S_ASK['easy'])}"
        working = [
            f"f(x) = g(x) where the two lines cross, at ({xi}, {yi}).",
            f"The x-coordinate of the crossing is {xi}, so {key}.",
        ]
        return mcq_result(
            text=text, key_text=key, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a two-line question")


def _solve_medium(rng, notes):
    for _ in range(ATT):
        win, M, C, pts = draw_line(
            rng,
            slopes=[F(s) for s in (-3, -2, -1, 1, 2, 3)]
            + [F(1, 2), F(-1, 2), F(1, 3), F(-1, 3), F(2, 3), F(-2, 3), F(3, 2)],
            c_range=(-4, 4),
            y_axis=None,
            min_points=3,
        )
        m, c = real_mc(win, M, C)
        readable = [p for p in pts if p[0] != 0 and p[1] != 0 and p[0] != p[1]]
        if not readable:
            continue
        x0, k = rng.choice(readable)
        pa, pb = pick_pair(rng, pts)
        fig = line_figure(win, [(pa, pb)], _ALT_LINE)
        fig["lines"][0]["label"] = "y = f(x)"
        rm, rc = read_line(fig)
        check((rm, rc) == (m, c) and rm * x0 + rc == k, "the line does not pass through the keyed point")
        key = _x_eq(x0)
        fk = m * k + c
        cands = [
            (_x_eq(k), "reads-the-y-value-instead-of-x"),
            (_x_eq(-x0), "sign-error-on-the-solution"),
            (_x_eq(x0 + rng.choice((-1, 1))), "off-by-one-gridline"),
            (_x_eq(c), "reads-where-a-line-meets-the-y-axis"),
        ]
        if fk.denominator == 1:
            cands.append((_x_eq(fk), "finds-f-of-k-instead-of-solving"))
        texts, note_map = take_distractors(rng, key, cands)
        if len(texts) < 3:
            continue
        try:
            no_leak(fig, key, texts)
        except AssertionError:
            continue
        text = f"{phrase(rng, _S_OPEN['medium'])} {phrase(rng, _S_ASK['medium'], k=k)}"
        working = [f"Find the height y = {k} on the graph.", f"The line is at that height where x = {x0}, so {key}."]
        return mcq_result(
            text=text, key_text=key, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a one-line question")


def _solve_hard(rng, notes):
    for _ in range(ATT * 2):
        win = _window(rng)
        ms = rng.sample([-3, -2, -1, 1, 2, 3], 3)
        cs = [rng.randint(-4, 4) for _ in range(3)]
        cross = {}
        ok = True
        for i, j in ((0, 1), (1, 2), (0, 2)):
            dm = ms[i] - ms[j]
            if (cs[j] - cs[i]) % dm:
                ok = False
                break
            x = (cs[j] - cs[i]) // dm
            y = ms[i] * x + cs[i]
            if not (win.u0 < x < win.u1 and win.v0 < y < win.v1):
                ok = False
                break
            cross[(i, j)] = (x, y)
        if not ok:
            continue
        if len(set(cross.values())) < 3 or len({p[0] for p in cross.values()}) < 3:
            continue
        # each line passes through the two crossings that involve it
        through = []
        for i in range(3):
            mine = [p for (a, b), p in cross.items() if i in (a, b)]
            through.append([list(mine[0]), list(mine[1])])
        letters = "ABC"
        fig = figures.cartesian(
            x_range=win.x_range,
            y_range=win.y_range,
            alt=_ALT_LINES,
            grid=1,
            lines=[{"through": through[i], "label": letters[i]} for i in range(3)],
        )
        for i in range(3):
            check(
                on_line(F(ms[i]), F(cs[i]), [tuple(q) for q in through[i]]),
                "a line does not pass through its crossings",
            )
        pair = rng.choice(list(cross))
        names = "fgh"
        key = _x_eq(cross[pair][0])
        others = [p for p in cross if p != pair]
        cands = [(_x_eq(cross[o][0]), "solves-for-the-wrong-pair-of-lines") for o in others]
        cands.append((_x_eq(cross[pair][1]), "reads-the-y-value-instead-of-x"))
        cands.append((_x_eq(-cross[pair][0]), "sign-error-on-the-solution"))
        texts, note_map = take_distractors(rng, key, cands)
        if len(texts) < 3:
            continue
        try:
            no_leak(fig, key, texts)
        except AssertionError:
            continue
        p_, q_ = (f"{names[pair[0]]}(x)", f"{names[pair[1]]}(x)")
        text = f"{phrase(rng, _S_OPEN['hard'])} {phrase(rng, _S_ASK['hard'], p=p_, q=q_)}"
        working = [
            f"{p_} = {q_} where lines {letters[pair[0]]} and {letters[pair[1]]} cross, at {tuple(cross[pair])}.",
            f"So {key}.",
        ]
        return mcq_result(
            text=text, key_text=key, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a three-line question")


@generator("fg1_solve_graphically", level=12, presets=_PRESETS)
def fg1_solve_graphically(rng, tier="easy", notes=False):
    if tier == "easy":
        return _solve_easy(rng, notes)
    if tier == "medium":
        return _solve_medium(rng, notes)
    if tier == "hard":
        return _solve_hard(rng, notes)
    raise ValueError(f"unknown tier {tier!r}")


def _read_solve(question) -> str:
    fig, text = question.figure, question.text
    lines = fig["lines"]
    if len(lines) == 1:
        k = F(re.search(r"f\(x\) = (-?\d+)", text).group(1))
        (x1, y1), (x2, y2) = lines[0]["through"]
        m = F(y2 - y1, x2 - x1)
        x0 = (k - (y1 - m * x1)) / m
        return _x_eq(x0)
    if len(lines) == 2:
        pair = (0, 1)
    else:
        named = re.search(r"([fgh])\(x\) = ([fgh])\(x\)", text)
        pair = ("fgh".index(named.group(1)), "fgh".index(named.group(2)))
    by_label = {ln["label"]: ln for ln in lines}
    a, b = by_label["ABC"[pair[0]]], by_label["ABC"[pair[1]]]

    def mc(ln):
        (x1, y1), (x2, y2) = ln["through"]
        m = F(y2 - y1, x2 - x1)
        return m, y1 - m * x1

    (m1, c1), (m2, c2) = mc(a), mc(b)
    return _x_eq((c2 - c1) / (m1 - m2))


figures.register_reader("fg1_solve_graphically", _read_solve)


# ----------------------------------------------------------------------------- compare linear relationships

_NOTE_INTERCEPT = "compares-intercepts-not-gradients"
_NOTE_GRADIENT = "compares-gradients-only"
_NOTE_SAME = "treats-the-relationships-as-the-same"
_NOTE_LOOK = "picks-by-the-look-of-the-graph"

_CL_OPEN = [
    "Relationship A is shown in the graph.",
    "The graph shows relationship A.",
    "Relationship A is the straight line on the grid.",
]
_CL_B_EQ = ["Relationship B is {eq}.", "Relationship B has the equation {eq}.", "Relationship B is given by {eq}."]
_CL_B_TABLE = [
    "Relationship B has these (x, y) pairs: {pairs}.",
    "Relationship B is given by this table, written as (x, y) pairs: {pairs}.",
    "The (x, y) pairs for relationship B are {pairs}.",
]
_CL_GRAD_ASK = [
    "Which relationship has the greater gradient?",
    "Which relationship has the greater rate of change?",
    "Which relationship has the greater slope?",
]
_CL_VALUE_ASK = [
    "Which relationship has the greater value of y when x = {X}?",
    "When x = {X}, which relationship gives the greater y-value?",
]
_CL_OPTIONS_GRAD = ("Relationship A", "Relationship B", "They have the same gradient")
_CL_OPTIONS_VALUE = ("Relationship A", "Relationship B", "They have the same y-value")


def _sgn(v) -> int:
    return (v > 0) - (v < 0)


def _cl_choose_b(rng, cls, mA, cA, tier):
    """B = (mB, cB) integers realising ``cls`` ('A', 'B' or 'same') against A, with the intercept pointing the
    wrong way in an unequal class. ``None`` when this A does not allow it."""
    for _ in range(200):
        mB = rng.randint(-3, 4) if tier != "easy" else rng.randint(1, 4)
        cB = rng.randint(-6, 6)
        if mB == 0:
            continue
        if cls == "same":
            if F(mB) == mA and cB != cA:
                return mB, cB
            continue
        if F(mB) == mA or cB == cA:
            continue
        a_wins = mA > mB
        if (cls == "A") == a_wins and (cA > cB) != a_wins:
            return mB, cB
    return None


def _cl_hard_b(rng, cls, mA, cA):
    X = rng.choice((-12, -10, -8, 8, 10, 12))
    for _ in range(300):
        mB = rng.randint(-3, 3)
        cB = rng.randint(-6, 6)
        if mB == 0 or F(mB) == mA:
            continue
        dX = (mA - mB) * X + (cA - cB)  # A(X) - B(X)
        d0 = cA - cB
        if cls == "same":
            if dX == 0 and d0 != 0:
                return X, mB, cB
            continue
        if d0 == 0 or dX == 0:
            continue
        if (_sgn(dX) > 0) == (cls == "A") and _sgn(d0) == -_sgn(dX):
            return X, mB, cB
    return None


def _table_pairs(mB, cB, xs):
    return [(x, mB * x + cB) for x in xs]


@generator("fg1_compare_linear", level=12, presets=_PRESETS)
def fg1_compare_linear(rng, tier="easy", notes=False):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    slopes = {
        "easy": [F(1), F(2), F(3)],
        "medium": [F(s) for s in (-3, -2, -1, 1, 2, 3)] + [F(1, 2), F(-1, 2), F(3, 2), F(-3, 2)],
        "hard": [F(s) for s in (-3, -2, -1, 1, 2, 3)] + [F(1, 2), F(-1, 2)],
    }[tier]
    cls = rng.choice(["A", "B", "same"])  # the class of the answer first, uniformly
    for _ in range(1500):
        win, M, C, pts = draw_line(rng, slopes=slopes, c_range=(-4, 4), y_axis=True, c_visible=True)
        mA, cA = real_mc(win, M, C)
        X = None
        if tier == "hard":
            got = _cl_hard_b(rng, cls, mA, cA)
            if got is None:
                continue
            X, mB, cB = got
        else:
            got = _cl_choose_b(rng, cls, mA, cA, tier)
            if got is None:
                continue
            mB, cB = got
        pa, pb = pick_pair(rng, pts)
        fig = line_figure(win, [(pa, pb)], _ALT_LINE)
        rm, rc = read_line(fig)
        check((rm, rc) == (mA, cA), "the drawn line is not relationship A")
        if tier == "easy":
            b_text = phrase(rng, _CL_B_EQ, eq=eq_text(mB, cB))
        else:
            xs = rng.choice(((0, 1, 2, 3), (0, 2, 4, 6), (1, 2, 3, 4)))
            b_text = phrase(rng, _CL_B_TABLE, pairs=pairs_text(_table_pairs(mB, cB, xs)))
        if tier == "hard":
            options = _CL_OPTIONS_VALUE
            ask = phrase(rng, _CL_VALUE_ASK, X=X)
            diff = (mA * X + cA) - (mB * X + cB)
        else:
            options = _CL_OPTIONS_GRAD
            ask = phrase(rng, _CL_GRAD_ASK)
            diff = mA - mB
        winner = "A" if diff > 0 else ("B" if diff < 0 else "same")
        check(winner == cls, "the constructed class is not the true class")
        key = options[("A", "B", "same").index(cls)]
        intercept_pick = "A" if cA > cB else ("B" if cA < cB else "same")
        gradient_pick = "A" if mA > mB else ("B" if mA < mB else "same")
        distractors, note_of = [], {}
        for label, text in zip(("A", "B", "same"), options):
            if label == cls:
                continue
            distractors.append(text)
            if label == "same":
                note_of[text] = _NOTE_SAME
            elif label == intercept_pick:
                note_of[text] = _NOTE_INTERCEPT
            elif label == gradient_pick and tier == "hard":
                note_of[text] = _NOTE_GRADIENT
            else:
                note_of[text] = _NOTE_LOOK
        rng.shuffle(distractors)
        no_leak(fig, key, distractors)
        text = f"{phrase(rng, _CL_OPEN)} {b_text} {ask}"
        if tier == "hard":
            working = [
                f"A: gradient {rational_text(mA)}, y-intercept {rational_text(cA)}, so y = {rational_text(mA * X + cA)} at x = {X}.",
                f"B: y = {mB * X + cB} at x = {X}.",
            ]
        else:
            working = [f"A has gradient {rational_text(mA)}; B has gradient {mB}.", f"The greater gradient is: {key}."]
        return mcq_result(
            text=text, key_text=key, texts=distractors, notes=note_of, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a comparison question")


def _parse_b(text: str):
    """(mB, cB) from relationship B's equation or its (x, y) pairs in ``text``."""
    eq = re.search(r"y = (-?)(\d*)x(?: ([+-]) (\d+))?", text)
    if eq and "pairs" not in text and "table" not in text:
        sign, mag, csign, cval = eq.groups()
        m = int(mag) if mag else 1
        m = -m if sign else m
        c = int(cval) if cval else 0
        return F(m), F(-c if csign == "-" else c)
    body = text.split("pairs")[-1] if "pairs" in text else text.split("table")[-1]
    pairs = parse_pairs(body)
    (x1, y1), (x2, y2) = pairs[0], pairs[1]
    m = F(y2 - y1, x2 - x1)
    check(all(F(y) == m * x + (y1 - m * x1) for x, y in pairs), "relationship B is not linear")
    return m, y1 - m * x1


def _read_compare_linear(question) -> str:
    mA, cA = read_line(question.figure)
    mB, cB = _parse_b(question.text)
    if any("y-value" in o for o in [question.answer_text, *question.distractors]):
        X = int(re.search(r"x = (-?\d+)", question.text).group(1))
        diff = (mA * X + cA) - (mB * X + cB)
        names = _CL_OPTIONS_VALUE
    else:
        diff = mA - mB
        names = _CL_OPTIONS_GRAD
    return names[0] if diff > 0 else (names[1] if diff < 0 else names[2])


figures.register_reader("fg1_compare_linear", _read_compare_linear)


# ----------------------------------------------------------------------------- compare functions

_CF_OPEN_LINE = [
    "Function f is shown in the graph.",
    "The graph shows function f.",
    "Function f is the straight line on the grid.",
]
_CF_OPEN_CURVE = [
    "Function f is shown in the graph.",
    "The graph shows the function f.",
    "Function f is the curve on the grid.",
]
_CF_OPTIONS = ("Function f", "Function g", "They are equal")
_CF_NOTE_INTERCEPT = "compares-starting-values-only"
_CF_NOTE_VERTEX_X = "compares-the-x-coordinates-of-the-turning-points"
_CF_NOTE_EQUAL = "treats-the-functions-as-equal"
_CF_NOTE_LOOK = "picks-by-the-look-of-the-graph"


def vertex_form(a: int, h: int, k: int) -> str:
    sq = f"(x - {h})^2" if h > 0 else f"(x + {-h})^2"
    lead = "" if a == 1 else "-" if a == -1 else str(a)
    body = f"{lead}{sq}"
    return body if k == 0 else f"{body} {'+' if k > 0 else '-'} {abs(k)}"


def _parabola(win: Window, a, h, k):
    n = (win.u1 - win.u0) * 8 + 1  # step 1/8, so every integer x is an exact sample
    return figures.sample_curve(lambda x: a * (x - h) ** 2 + k, win.u0, win.u1, (win.v0, win.v1), n=n)


def _curve_figure(win: Window, a, h, k) -> dict:
    return figures.cartesian(
        x_range=win.x_range,
        y_range=win.y_range,
        alt=_ALT_CURVE,
        grid=1,
        curves=[{"pieces": _parabola(win, a, h, k)}],
    )


def _class_of(diff) -> str:
    return "f" if diff > 0 else ("g" if diff < 0 else "same")


def _cf_easy(rng, notes):
    cls = rng.choice(["f", "g", "same"])  # the answer class first, uniformly
    for _ in range(ATT):
        win, M, C, pts = draw_line(
            rng,
            slopes=[F(s) for s in (-2, -1, 1, 2, 3)],
            c_range=(-4, 4),
            y_axis=True,
            c_visible=True,
            c_nonzero=True,
            min_points=3,
        )
        readable = [p for p in pts if p[0] != 0]
        if not readable:
            continue
        x0, v = rng.choice(readable)
        a = rng.choice((-3, -2, 2, 3, 4))
        d = rng.randint(1, 3)
        target = v + {"f": -d, "g": d, "same": 0}[cls]
        b = target - a * x0
        if abs(b) > 9 or b == 0:
            continue
        if C == b or (cls != "same" and ((C > b) == (cls == "f"))):
            continue  # comparing the starting values would give the right answer
        pa, pb = pick_pair(rng, pts)
        fig = line_figure(win, [(pa, pb)], _ALT_LINE)
        rm, rc = read_line(fig)
        check(rm * x0 + rc == v, "f(x0) is not the keyed value")
        key = _CF_OPTIONS[("f", "g", "same").index(cls)]
        verb = f"adds {b}" if b > 0 else f"subtracts {-b}"
        rule = f"Function g takes an input, multiplies it by {a} and then {verb}."
        intercept_pick = "f" if C > b else "g"
        distractors = [t for lab, t in zip(("f", "g", "same"), _CF_OPTIONS) if lab != cls]
        note_of = {}
        for lab, t in zip(("f", "g", "same"), _CF_OPTIONS):
            if lab == cls:
                continue
            note_of[t] = (
                _CF_NOTE_EQUAL if lab == "same" else (_CF_NOTE_INTERCEPT if lab == intercept_pick else _CF_NOTE_LOOK)
            )
        rng.shuffle(distractors)
        no_leak(fig, key, distractors)
        text = f"{phrase(rng, _CF_OPEN_LINE)} {rule} Which function gives the greater output when the input is {x0}?"
        working = [
            f"f({x0}) = {v} from the graph.",
            f"g({x0}) = {a} × {paren_negative(x0)}{f' {chr(43) if b > 0 else chr(45)} {abs(b)}' if b else ''} = {a * x0 + b}.",
            f"So {key}.",
        ]
        return mcq_result(
            text=text, key_text=key, texts=distractors, notes=note_of, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a function-comparison question")


def _cf_medium(rng, notes):
    cls = rng.choice(["f", "g", "same"])  # the answer class first, uniformly
    for _ in range(ATT):
        u0, v0 = rng.randint(-6, -3), rng.randint(-7, -4)
        win = Window(u0, u0 + 9, v0, v0 + 9)
        af, ag = rng.choice((-1, -2)), rng.choice((-1, -2, -3))
        hf, hg = rng.randint(win.u0 + 1, win.u1 - 1), rng.choice([h for h in range(-5, 6) if h != 0])
        kf = rng.randint(max(win.v0 + 2, -3), min(win.v1 - 1, 4))
        d = rng.randint(1, 3)
        kg = kf + {"f": -d, "g": d, "same": 0}[cls]
        if kf == 0 or kg == 0 or hf == hg or abs(hg - hf) < 1:
            continue
        if cls != "same" and ((hf > hg) == (cls == "f")):
            continue  # comparing the turning points' x-coordinates would give the right answer
        if hf == 0:
            continue
        fig = _curve_figure(win, af, hf, kf)
        key = _CF_OPTIONS[("f", "g", "same").index(cls)]
        g_text = f"g(x) = {vertex_form(ag, hg, kg)}"
        distractors = [t for lab, t in zip(("f", "g", "same"), _CF_OPTIONS) if lab != cls]
        vertex_pick = "f" if hf > hg else "g"
        note_of = {}
        for lab, t in zip(("f", "g", "same"), _CF_OPTIONS):
            if lab == cls:
                continue
            note_of[t] = (
                _CF_NOTE_EQUAL if lab == "same" else (_CF_NOTE_VERTEX_X if lab == vertex_pick else _CF_NOTE_LOOK)
            )
        rng.shuffle(distractors)
        no_leak(fig, key, distractors)
        text = f"{phrase(rng, _CF_OPEN_CURVE)} Function g is given by {g_text}. Which function has the greater maximum value?"
        working = [
            f"The graph of f has its highest point at y = {kf}.",
            f"g has maximum value {kg} (its turning point).",
            f"So {key}.",
        ]
        return mcq_result(
            text=text, key_text=key, texts=distractors, notes=note_of, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a maximum-value question")


def _cf_hard(rng, notes):
    cls = rng.choice(["f", "g", "same"])  # the answer class first, uniformly
    for _ in range(ATT):
        u0, v0 = rng.randint(-3, -2), rng.randint(-6, -3)
        win = Window(u0, u0 + 9, v0, v0 + 9)
        a = rng.choice((-2, -1, 1, 2))
        h = rng.randint(max(win.u0 + 1, -1), 2)
        k = rng.randint(-2, 2)
        xs = [x for x in range(win.u0, win.u1 + 1) if abs(x) >= 3 and win.v0 <= a * (x - h) ** 2 + k <= win.v1]
        if not xs:
            continue
        X = rng.choice(xs)
        fX = a * (X - h) ** 2 + k
        f0 = a * h * h + k
        m = rng.choice((-3, -2, -1, 1, 2, 3))
        d = rng.randint(1, 3)
        target = fX + {"f": -d, "g": d, "same": 0}[cls]
        c = target - m * X
        if abs(c) > 9 or m == 0:
            continue
        if f0 == c or (cls != "same" and ((f0 > c) == (cls == "f"))):
            continue  # comparing the values at x = 0 would give the right answer
        fig = _curve_figure(win, a, h, k)
        key = _CF_OPTIONS[("f", "g", "same").index(cls)]
        pairs = pairs_text(_table_pairs(m, c, (0, 1, 2)))
        distractors = [t for lab, t in zip(("f", "g", "same"), _CF_OPTIONS) if lab != cls]
        start_pick = "f" if f0 > c else "g"
        note_of = {}
        for lab, t in zip(("f", "g", "same"), _CF_OPTIONS):
            if lab == cls:
                continue
            note_of[t] = (
                _CF_NOTE_EQUAL if lab == "same" else (_CF_NOTE_INTERCEPT if lab == start_pick else _CF_NOTE_LOOK)
            )
        rng.shuffle(distractors)
        no_leak(fig, key, distractors)
        text = (
            f"{phrase(rng, _CF_OPEN_CURVE)} Function g is linear and has these (x, y) pairs: {pairs}. "
            f"Which function has the greater value when x = {X}?"
        )
        working = [
            f"f({X}) = {fX} from the graph.",
            f"g has gradient {m} and g(0) = {c}, so g({X}) = {m * X + c}.",
            f"So {key}.",
        ]
        return mcq_result(
            text=text, key_text=key, texts=distractors, notes=note_of, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a linear-against-parabola question")


@generator("fg1_compare_functions", level=13, presets=_PRESETS)
def fg1_compare_functions(rng, tier="easy", notes=False):
    if tier == "easy":
        return _cf_easy(rng, notes)
    if tier == "medium":
        return _cf_medium(rng, notes)
    if tier == "hard":
        return _cf_hard(rng, notes)
    raise ValueError(f"unknown tier {tier!r}")


def _curve_samples(fig):
    return [tuple(p) for piece in fig["curves"][0]["pieces"] for p in piece]


def _read_compare_functions(question) -> str:
    fig, text = question.figure, question.text
    if fig["lines"]:
        mf, cf = read_line(fig)
        x0 = int(re.search(r"input is (-?\d+)", text).group(1))
        mul = re.search(r"multiplies it by (-?\d+) and then (adds|subtracts) (\d+)", text)
        a = int(mul.group(1))
        b = int(mul.group(3)) * (1 if mul.group(2) == "adds" else -1)
        diff = (mf * x0 + cf) - (a * x0 + b)
    else:
        samples = _curve_samples(fig)
        if "greater maximum value" in text:
            fmax = max(y for _x, y in samples)
            g = re.search(r"g\(x\) = (-?\d*)\(x ([+-]) (\d+)\)\^2(?: ([+-]) (\d+))?", text)
            gmax = int(g.group(5) or 0) * (-1 if g.group(4) == "-" else 1)
            diff = fmax - gmax
        else:
            X = int(re.search(r"when x = (-?\d+)", text).group(1))
            fX = [y for x, y in samples if x == X][0]
            pairs = parse_pairs(text.split("pairs:")[1])
            (x1, y1), (x2, y2) = pairs[0], pairs[1]
            m = F(y2 - y1, x2 - x1)
            diff = fX - (m * (X - x1) + y1)
    return _CF_OPTIONS[0] if diff > 0 else (_CF_OPTIONS[1] if diff < 0 else _CF_OPTIONS[2])


figures.register_reader("fg1_compare_functions", _read_compare_functions)


# ----------------------------------------------------------------------------- inequality on a number line

_SYM = {"<": "<", "<=": "≤", ">": ">", ">=": "≥"}
_NL_OPEN = [
    "The number line shows the solution of an inequality.",
    "The solution of an inequality is shown on the number line.",
    "The shaded part of the number line is the solution of an inequality.",
]
_NL_ASK = {
    "easy": [
        "Which inequality has this solution?",
        "Which inequality does the number line show?",
        "Which inequality is shown on the number line?",
    ],
    "medium": ["Which inequality does the number line show?", "Which inequality has this solution?"],
    "hard": [
        "Which of these inequalities has the solution shown?",
        "Solve each inequality. Which one has the solution shown on the number line?",
        "Which inequality has exactly the solution shown on the number line?",
    ],
}
_NL_RANGE = (-8, 8)
_N_CLOSED = "closed-endpoint-for-a-strict-inequality"
_N_OPEN = "open-endpoint-for-an-inclusive-inequality"
_N_DIR = "shading-direction-reversed"
_N_BOTH = "direction-and-endpoint-both-wrong"
_N_NOFLIP = "forgets-to-reverse-the-sign-when-dividing-by-a-negative"
_N_REV = "inequality-direction-reversed"
_N_STRICT = "strict-and-inclusive-confused"
_N_CONST = "sign-error-on-the-constant"


def _ray_text(p, direction, open_):
    sym = ("<" if open_ else "<=") if direction == "left" else (">" if open_ else ">=")
    return f"x {_SYM[sym]} {p}"


_NL_VALUE_ASK = [
    "Which of these numbers is a solution of the inequality?",
    "Which number is in the solution shown on the number line?",
    "Which of these values makes the inequality true?",
]


def _in_ray(v, p, direction, open_):
    if direction == "right":
        return v > p if open_ else v >= p
    return v < p if open_ else v <= p


def _nl_easy_value(rng, notes):
    """The ray is shown and four numbers are offered: which one is in the solution? The endpoint itself is the
    deciding case, a solution only when the circle is closed."""
    lo, hi = _NL_RANGE
    for _ in range(ATT):
        p = rng.choice([v for v in range(-6, 7) if v != 0])
        direction, open_ = rng.choice(("left", "right")), rng.choice((True, False))
        sgn = 1 if direction == "right" else -1
        inside = [v for v in range(lo, hi + 1) if _in_ray(v, p, direction, open_)]
        outside = [v for v in range(lo, hi + 1) if not _in_ray(v, p, direction, open_)]
        if not inside or len(outside) < 3:
            continue
        key_v = p if (not open_ and rng.random() < 0.4) else rng.choice(inside)
        cands = [(str(v), _N_DIR) for v in outside if v != p]
        if open_:
            cands.append((str(p), _N_CLOSED))
            cands.append((str(p), _N_CLOSED))  # the endpoint is the likeliest slip, so it is drawn more often
        else:
            cands.append((str(p - sgn), _N_OPEN))
        texts, note_map = take_distractors(rng, str(key_v), cands)
        if len(texts) < 3:
            continue
        fig = figures.number_line(
            range_=_NL_RANGE, alt=_ALT_NUMBER_LINE, tick=1, rays=[{"from": p, "direction": direction, "open": open_}]
        )
        try:
            no_leak(fig, str(key_v), texts)
        except AssertionError:
            continue
        text = f"{phrase(rng, _NL_OPEN)} {phrase(rng, _NL_VALUE_ASK)}"
        word = "an open circle, so the end is not included" if open_ else "a closed circle, so the end is included"
        working = [
            f"The shading goes {direction} from {p}, with {word}.",
            f"{key_v} is in the shaded part, so it is a solution.",
        ]
        return mcq_result(
            text=text, key_text=str(key_v), texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a solution-value question")


def _nl_easy(rng, notes):
    if rng.random() < 0.4:
        return _nl_easy_value(rng, notes)
    p = rng.choice([v for v in range(-7, 8) if v != 0])
    direction, open_ = rng.choice(("left", "right")), rng.choice((True, False))
    fig = figures.number_line(
        range_=_NL_RANGE, alt=_ALT_NUMBER_LINE, tick=1, rays=[{"from": p, "direction": direction, "open": open_}]
    )
    key = _ray_text(p, direction, open_)
    other = {"left": "right", "right": "left"}
    cands = [
        (_ray_text(p, direction, not open_), _N_CLOSED if open_ else _N_OPEN),
        (_ray_text(p, other[direction], open_), _N_DIR),
        (_ray_text(p, other[direction], not open_), _N_BOTH),
    ]
    texts, note_map = take_distractors(rng, key, cands)
    check(len(texts) == 3, "the four options are the four combinations")
    no_leak(fig, key, texts)
    text = f"{phrase(rng, _NL_OPEN)} {phrase(rng, _NL_ASK['easy'])}"
    word = "an open circle, so the end is not included" if open_ else "a closed circle, so the end is included"
    working = [f"The shading goes {direction} from {p}, with {word}.", f"So the inequality is {key}."]
    return mcq_result(
        text=text, key_text=key, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
    )


def _interval_text(a, b, open_start, open_end):
    return f"{a} {'<' if open_start else '≤'} x {'<' if open_end else '≤'} {b}"


def _nl_medium(rng, notes):
    for _ in range(ATT):
        a = rng.randint(-6, 3)
        b = rng.randint(a + 2, 6)
        os_, oe = rng.choice((True, False)), rng.choice((True, False))
        fig = figures.number_line(
            range_=_NL_RANGE,
            alt=_ALT_NUMBER_LINE,
            tick=1,
            intervals=[{"from": a, "to": b, "open_start": os_, "open_end": oe}],
        )
        key = _interval_text(a, b, os_, oe)
        cands = []
        for ws, we in ((not os_, oe), (os_, not oe), (not os_, not oe)):
            if (ws, we) == (not os_, oe):
                note = _N_CLOSED if os_ else _N_OPEN
            elif (ws, we) == (os_, not oe):
                note = _N_CLOSED if oe else _N_OPEN
            else:
                note = _N_STRICT
            cands.append((_interval_text(a, b, ws, we), note))
        texts, note_map = take_distractors(rng, key, cands)
        check(len(texts) == 3, "the four options are the four endpoint combinations")
        no_leak(fig, key, texts)
        text = f"{phrase(rng, _NL_OPEN)} {phrase(rng, _NL_ASK['medium'])}"
        working = [
            f"The shading runs from {a} to {b}.",
            f"The end at {a} is {'open' if os_ else 'closed'} and the end at {b} is {'open' if oe else 'closed'}.",
            f"So the inequality is {key}.",
        ]
        return mcq_result(
            text=text, key_text=key, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build an interval question")


def _lin_text(a: int, b: int, c: int, op: str) -> str:
    lead = "" if a == 1 else "-" if a == -1 else str(a)
    return f"{lead}x {'+' if b > 0 else '-'} {abs(b)} {_SYM[op]} {c}"


def _flip(op: str) -> str:
    return {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[op]


def _toggle(op: str) -> str:
    return {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}[op]


def _solve(a: int, b: int, c: int, op: str):
    """Solution of ``a x + b op c`` as ``(direction, threshold, open)``."""
    t = F(c - b, a)
    sym = _flip(op) if a < 0 else op
    return ("left" if sym in ("<", "<=") else "right"), t, sym in ("<", ">")


def _nl_hard(rng, notes):
    for _ in range(ATT):
        a = rng.choice((-4, -3, -2, -1, 2, 3, 4))
        t = rng.choice([v for v in range(-5, 6) if v != 0])
        b = rng.choice([v for v in range(-6, 7) if v != 0])
        c = a * t + b
        op = rng.choice(("<", "<=", ">", ">="))
        direction, thr, open_ = _solve(a, b, c, op)
        check(thr == t, "the threshold is not the keyed one")
        fig = figures.number_line(
            range_=_NL_RANGE,
            alt=_ALT_NUMBER_LINE,
            tick=1,
            rays=[{"from": int(thr), "direction": direction, "open": open_}],
        )
        key = _lin_text(a, b, c, op)
        cands = [
            (_lin_text(a, b, c, _flip(op)), _N_NOFLIP if a < 0 else _N_REV),
            (_lin_text(a, b, c, _toggle(op)), _N_STRICT),
            (_lin_text(a, -b, c, op), _N_CONST),
            (_lin_text(a, b, -c, op), _N_CONST),
        ]
        cands = [
            (s, n)
            for (s, n), args in zip(
                cands, ((a, b, c, _flip(op)), (a, b, c, _toggle(op)), (a, -b, c, op), (a, b, -c, op))
            )
            if _solve(*args) != (direction, thr, open_)
        ]
        texts, note_map = take_distractors(rng, key, cands)
        if len(texts) < 3:
            continue
        no_leak(fig, key, texts)
        text = f"{phrase(rng, _NL_OPEN)} {phrase(rng, _NL_ASK['hard'])}"
        working = [
            f"Solve {key}: subtract {paren_negative(b)} from both sides to get {fmt_signed(a, 'x')} {_SYM[op]} {c - b}.",
            f"Divide by {a}{', reversing the inequality sign' if a < 0 else ''}: x {_SYM[_flip(op) if a < 0 else op]} {t}.",
            "This matches the shaded part of the number line.",
        ]
        return mcq_result(
            text=text, key_text=key, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a solve-and-match question")


@generator("fg1_inequality_number_line", level=11, presets=_PRESETS)
def fg1_inequality_number_line(rng, tier="easy", notes=False):
    if tier == "easy":
        return _nl_easy(rng, notes)
    if tier == "medium":
        return _nl_medium(rng, notes)
    if tier == "hard":
        return _nl_hard(rng, notes)
    raise ValueError(f"unknown tier {tier!r}")


_LIN_RE = re.compile(r"^(-?)(\d*)x ([+-]) (\d+) (<|≤|>|≥) (-?\d+)$")
_REV_SYM = {v: k for k, v in _SYM.items()}


def _read_number_line(question) -> str:
    fig = question.figure
    options = [question.answer_text, *question.distractors]
    if fig["rays"]:
        ray = fig["rays"][0]
        shown = (ray["direction"], F(ray["from"]), bool(ray.get("open")))
        if all(re.fullmatch(r"-?\d+", o) for o in options):
            inside = [o for o in options if _in_ray(int(o), int(ray["from"]), ray["direction"], bool(ray.get("open")))]
            check(len(inside) == 1, "exactly one offered number must be in the shaded part")
            return inside[0]
        for o in options:
            m = _LIN_RE.match(o)
            if m:
                sign, mag, bsign, bval, sym, cval = m.groups()
                a = int(mag or 1) * (-1 if sign else 1)
                b = int(bval) * (1 if bsign == "+" else -1)
                if _solve(a, b, int(cval), _REV_SYM[sym]) == shown:
                    return o
            elif o == _ray_text(int(ray["from"]), ray["direction"], bool(ray.get("open"))):
                return o
        raise AssertionError("no option matches the shaded ray")
    iv = fig["intervals"][0]
    want = _interval_text(int(iv["from"]), int(iv["to"]), bool(iv.get("open_start")), bool(iv.get("open_end")))
    check(want in options, "no option matches the shaded interval")
    return want


figures.register_reader("fg1_inequality_number_line", _read_number_line)
