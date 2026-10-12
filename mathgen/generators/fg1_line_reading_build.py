"""Read a straight line, or a few plotted points, off a shown coordinate grid.

Five topics, each a "read from a shown figure" question. None of them asks the learner to draw, plot or
sketch anything, and nothing the key needs is printed: the figure is the whole stimulus.

- ``fg1_gradient_from_graph`` (skill ``gradient-from-graph``, AC9M9A03): the gradient of a drawn line, a
  ``fraction`` answer. Easy: two marked points, whole-number gradient. Medium: two marked points,
  negative or fractional gradient. Hard: no marked points, the two axes use different scales (so
  counting squares is wrong); the learner picks two grid crossings and reads rise and run in axis units.
- ``fg1_equation_from_graph`` (skill ``equation-from-graph``, AC9M8A02): which equation ``y = mx + c``
  matches the line, an mcq. Easy: whole-number gradient, y-axis drawn. Medium: fractional or negative
  gradient. Hard: the y-axis is outside the window, so the intercept is found by running the line back.
- ``fg1_proportional_graph`` (skill ``graph-y-mx``, AC9M8A02): which equation ``y = mx`` matches a line
  through the origin, an mcq. Easy: whole-number gradient. Medium: negative or fractional gradient. Hard:
  the axes use different scales.
- ``fg1_line_table`` (skill ``graph-y-mx-plus-b``, AC9M8A02): which list of (x, y) pairs lies on the
  shown line, an mcq. Easy: whole-number gradient and consecutive x. Medium: negative or fractional
  gradient. Hard: the y-axis is outside the window.
- ``fg1_plotted_points`` (skill ``plot-linear-points``, AC9M8A02): read plotted points. Easy: the
  coordinates of one labelled point. Medium: which list of pairs matches three plotted points on a line.
  Hard: which further ordered pair lies on the same straight line.

Every drawn line passes through lattice points (grid intersections), so every value a key needs is read
from a labelled gridline. Distractors come from named misconceptions (rise and run inverted, a sign error,
the intercept read off the wrong axis, a gradient counted in squares, coordinates reversed) and every one
is a different answer from the key by construction. Each topic registers a plausibility reader that
re-derives the answer from ``question.figure`` alone (plus the numbers the prompt states, where it states
any).
"""

from __future__ import annotations

import re
from fractions import Fraction as F

from mathgen import figures
from mathgen.core import check, check_pairwise_distinct, generator, phrase, pick_distractors, rational_text
from mathgen.generators.pv1a_line_figure_build import Window, lattice_points, line_figure, read_line

ATT = 3000
_ALT_LINE = "a straight line on a coordinate grid"
_ALT_POINTS = "points plotted on a coordinate grid"
_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}


# ----------------------------------------------------------------------------- shared helpers


def eq_text(m, c) -> str:
    """``y = mx + c`` the way a student writes it: ``y = 2x + 1``, ``y = -x/2 + 3``, ``y = 2x/3 - 1``."""
    m, c = F(m), F(c)
    num, den = m.numerator, m.denominator
    body = ("x" if num > 0 else "-x") if abs(num) == 1 else f"{num}x"
    if den != 1:
        body += f"/{den}"
    if c == 0:
        return f"y = {body}"
    return f"y = {body} {'+' if c > 0 else '-'} {rational_text(abs(c))}"


def pairs_text(points) -> str:
    return ", ".join(f"({x}, {y})" for x, y in points)


def _frac_answer(v: F):
    """The ``answer`` value a ``fraction`` grader reads: an int, or ``{"p", "q"}``."""
    v = F(v)
    return int(v) if v.denominator == 1 else {"p": v.numerator, "q": v.denominator}


def _frac_latex(v: F) -> str:
    v = F(v)
    if v.denominator == 1:
        return str(v.numerator)
    return rf"{'-' if v < 0 else ''}\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"


def draw_line(
    rng,
    *,
    slopes,
    c_range=(-4, 4),
    y_axis=None,
    scales=((1, 1),),
    min_points=2,
    min_du=2,
    origin=False,
    c_visible=None,
    c_nonzero=False,
):
    """Sample a window and a line ``v = M u + C`` (grid units) with enough lattice points.

    ``y_axis``: True draws x = 0 strictly inside the window, False keeps it out, None leaves it free.
    ``c_visible``: True needs the intercept inside the window. Returns ``(win, M, C, pts)``."""
    for _ in range(ATT):
        nu, nv = rng.randint(8, 10), rng.randint(8, 10)
        if y_axis is True:
            u0 = rng.randint(-(nu - 2), -1)
        elif y_axis is False:
            k = rng.randint(1, 4)
            u0 = k if rng.random() < 0.5 else -k - (nu - 1)
        else:
            u0 = rng.randint(-(nu - 2), 0)
        v0 = rng.randint(-(nv - 2), -1)
        xs, ys = rng.choice(scales)
        win = Window(u0, u0 + nu - 1, v0, v0 + nv - 1, xs=xs, ys=ys)
        M = rng.choice(slopes)
        C = 0 if origin else rng.randint(*c_range)
        if c_nonzero and C == 0:
            continue
        if c_visible and not (win.v0 <= C <= win.v1 and win.shows_y_axis):
            continue
        pts = lattice_points(win, M, C)
        if len(pts) < min_points:
            continue
        if not any(b[0] - a[0] >= min_du for i, a in enumerate(pts) for b in pts[i + 1 :]):
            continue
        return win, M, C, pts
    raise ValueError("no plausible line and window")


def pick_pair(rng, pts, min_du=2):
    pairs = [(a, b) for i, a in enumerate(pts) for b in pts[i + 1 :] if b[0] - a[0] >= min_du]
    return rng.choice(pairs)


def real_pt(win: Window, p):
    return (p[0] * win.xs, p[1] * win.ys)


def real_mc(win: Window, M, C):
    """Real-unit gradient and intercept of the grid-unit line ``v = M u + C``."""
    return F(M) * win.ys / win.xs, F(C) * win.ys


def take_distractors(rng, key_text: str, cands, n=3, *, rank=None):
    """Distinct distractor texts from ``[(text, note_id), ...]``. The list is shuffled first so no
    misconception is always the first one offered; ``rank(text)`` (lower first) then moves the
    implausible ones to the back. Returns ``(texts, {text: note})``."""
    cands = list(cands)
    rng.shuffle(cands)
    if rank is not None:
        cands.sort(key=lambda c: rank(c[0]))
    texts = pick_distractors(key_text, [t for t, _ in cands], n=n)
    note_of = dict(cands)
    return texts, {t: note_of[t] for t in texts}


def no_leak(fig, key_text, texts):
    figures.assert_no_leak(fig, [key_text, *texts])


def mcq_result(*, text, key_text, texts, notes, working, fig, option_notes):
    return dict(
        text=text,
        answer=key_text,
        answer_text=key_text,
        answer_latex=f"plain:{key_text}",
        distractors=texts,
        working=working,
        figure=fig,
        **({"option_notes": notes} if option_notes else {}),
    )


# ----------------------------------------------------------------------------- gradient from graph

_G_SLOPES = {
    "easy": [F(2), F(3)],
    "medium": [F(-3), F(-2), F(1, 2), F(-1, 2), F(1, 3), F(-1, 3), F(2, 3), F(-2, 3), F(3, 2), F(-3, 2)],
    "hard": [F(s) for s in (-3, -2, -1, 1, 2, 3)] + [F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(1, 3), F(-1, 3)],
}
_UNEQUAL = ((1, 2), (1, 5), (2, 1), (5, 1), (2, 5), (5, 2), (1, 10))

_G_OPEN = [
    "The graph shows a straight line on a coordinate grid.",
    "A straight line is shown on the coordinate grid below.",
    "Look at the straight line shown on this grid.",
    "The diagram shows the graph of a straight line.",
    "Study the straight line on this coordinate grid.",
]
_G_ASK = [
    "What is the gradient of the line?",
    "Find the gradient of the line.",
    "Work out the gradient of this line.",
    "What is the slope of the line?",
]
_G_NOTE = {
    "easy": ["Two points on the line are marked.", "The line passes through the two marked points.", ""],
    "medium": [
        "Two points on the line are marked. Give the gradient as a whole number or a fraction.",
        "The line passes through the two marked points. Write the gradient as a whole number or a fraction.",
    ],
    "hard": [
        "Use two points where the line crosses grid intersections, and read the scale on each axis.",
        "The axes do not use the same scale. Pick two points on grid intersections and use the axis labels.",
    ],
}


def _gradient_candidates(m: F, M: F, dy: int, tier: str):
    c = [(1 / m, "inverted-gradient"), (-m, "sign-error-on-gradient")]
    if tier == "hard":
        c.append((M, "gradient-counted-in-squares"))
    c.append((F(dy), "rise-without-run"))
    c.append((-1 / m, "inverted-and-sign-error"))
    return c


@generator("fg1_gradient_from_graph", level=11, presets=_PRESETS)
def fg1_gradient_from_graph(rng, tier="easy", notes=False):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        scales = _UNEQUAL if tier == "hard" else ((1, 1),)
        win, M, C, pts = draw_line(
            rng,
            slopes=_G_SLOPES[tier],
            c_range=(-4, 4),
            y_axis=True if tier == "easy" else None,
            scales=scales,
            min_points=3 if tier == "hard" else 2,
        )
        m, _c = real_mc(win, M, C)
        if tier == "hard" and (m.denominator > 12 or abs(m) > 20):
            continue
        pa, pb = pick_pair(rng, pts)
        marks = [{"at": list(real_pt(win, p))} for p in (pa, pb)] if tier != "hard" else []
        fig = line_figure(win, [(pa, pb)], _ALT_LINE, points=marks)
        rm, _ = read_line(fig)
        check(rm == m, "the drawn line does not have the keyed gradient")
        check(m != 0 and (tier != "hard" or win.xs != win.ys), "gradient must be non-zero; hard needs unequal scales")
        dy = (pb[1] - pa[1]) * win.ys
        key_text = rational_text(m)
        cands = [(rational_text(v), n) for v, n in _gradient_candidates(m, M, dy, tier) if v != m and v != 0]
        texts, note_map = take_distractors(rng, key_text, cands)
        if len(texts) < 3:
            continue
        vals = [F(t) for t in (key_text, *texts)]
        check_pairwise_distinct(*vals)
        try:
            no_leak(fig, key_text, texts)
        except AssertionError:
            continue
        break
    else:
        raise ValueError("could not build a gradient question")
    text = " ".join(p for p in (phrase(rng, _G_OPEN), phrase(rng, _G_ASK), phrase(rng, _G_NOTE[tier])) if p)
    (x1, y1), (x2, y2) = real_pt(win, pa), real_pt(win, pb)
    working = [
        f"Read two points on the line: ({x1}, {y1}) and ({x2}, {y2}).",
        f"rise = {y2 - y1}, run = {x2 - x1}.",
        f"gradient = rise / run = {rational_text(m)}.",
    ]
    return dict(
        text=text,
        answer=_frac_answer(m),
        answer_text=key_text,
        answer_latex=_frac_latex(m),
        distractors=texts,
        working=working,
        figure=fig,
        **({"option_notes": note_map} if notes else {}),
    )


def _read_gradient(question) -> str:
    m, _c = read_line(question.figure)
    return rational_text(m)


figures.register_reader("fg1_gradient_from_graph", _read_gradient)


# ----------------------------------------------------------------------------- equation from graph

_E_SLOPES = {
    "easy": [F(1), F(2), F(3)],
    "medium": [F(-3), F(-2), F(-1), F(1, 2), F(-1, 2), F(1, 3), F(-1, 3), F(2, 3), F(-2, 3), F(3, 2), F(-3, 2)],
    "hard": [F(-3), F(-2), F(-1), F(2), F(3), F(1, 2), F(-1, 2), F(1, 3), F(-1, 3), F(2, 3), F(-2, 3)],
}
_E_OPEN = [
    "The graph shows a straight line on a coordinate grid.",
    "A straight line is shown on the coordinate grid below.",
    "Look at the straight line shown on this grid.",
    "The diagram shows the graph of a straight line.",
]
_E_ASK = [
    "Which equation is the equation of the line?",
    "Which of these equations matches the graph?",
    "Which equation describes this line?",
    "Choose the equation that matches the line.",
]
_E_NOTE = {
    "easy": ["", "The equation has the form y = mx + c."],
    "medium": ["The equation has the form y = mx + c.", "Each equation is written in the form y = mx + c."],
    "hard": [
        "The y-axis is not shown, so find where the line would meet it. Each equation has the form y = mx + c.",
        "Not all of the grid is shown. Each equation has the form y = mx + c.",
    ],
}


def _equation_candidates(m: F, c: F):
    out = [((-m, c), "sign-error-on-gradient"), ((m, -c), "sign-error-on-intercept")]
    if m.denominator == 1 and c.denominator == 1:
        out.append(((c, m), "gradient-and-intercept-swapped"))
    if abs(m) != 1:
        out.append(((1 / m, c), "inverted-gradient"))
    xi = -c / m
    if xi.denominator == 1:
        out.append(((m, xi), "intercept-read-off-the-x-axis"))
    return out


@generator("fg1_equation_from_graph", level=12, presets=_PRESETS)
def fg1_equation_from_graph(rng, tier="easy", notes=False):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        win, M, C, pts = draw_line(
            rng,
            slopes=_E_SLOPES[tier],
            c_range=(-4, 4) if tier != "hard" else (-9, 9),
            y_axis=False if tier == "hard" else True,
            c_visible=tier != "hard",
            c_nonzero=True,
        )
        m, c = real_mc(win, M, C)
        pa, pb = pick_pair(rng, pts)
        fig = line_figure(win, [(pa, pb)], _ALT_LINE)
        rm, rc = read_line(fig)
        check((rm, rc) == (m, c), "the drawn line is not the keyed one")
        key_text = eq_text(m, c)
        cands = [(eq_text(*p), n) for p, n in _equation_candidates(m, c)]
        cands.append((eq_text(m, c + rng.choice((-1, 1))), "intercept-misread-by-one-square"))
        texts, note_map = take_distractors(rng, key_text, cands)
        if len(texts) < 3:
            continue
        try:
            no_leak(fig, key_text, texts)
        except AssertionError:
            continue
        break
    else:
        raise ValueError("could not build an equation question")
    text = " ".join(p for p in (phrase(rng, _E_OPEN), phrase(rng, _E_ASK), phrase(rng, _E_NOTE[tier])) if p)
    (x1, y1), (x2, y2) = real_pt(win, pa), real_pt(win, pb)
    working = [
        f"Read two points on the line: ({x1}, {y1}) and ({x2}, {y2}).",
        f"gradient m = ({y2} - ({y1})) / ({x2} - ({x1})) = {rational_text(m)}.",
        f"Running the line to x = 0 gives c = {rational_text(c)}, so {key_text}.",
    ]
    return mcq_result(
        text=text, key_text=key_text, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
    )


def _read_equation(question) -> str:
    m, c = read_line(question.figure)
    return eq_text(m, c)


figures.register_reader("fg1_equation_from_graph", _read_equation)


# ----------------------------------------------------------------------------- proportional graph (y = mx)

_P_SLOPES = {
    "easy": [F(1), F(2), F(3)],
    "medium": [F(-3), F(-2), F(-1), F(1, 2), F(-1, 2), F(1, 3), F(-1, 3), F(2, 3), F(3, 2)],
    "hard": [F(-3), F(-2), F(-1), F(1), F(2), F(3)],
}
_P_SCALES = ((1, 2), (1, 5), (1, 10))
_P_OPEN = [
    "The graph shows a straight line through the origin of a coordinate grid.",
    "A straight line through the origin is shown on the coordinate grid below.",
    "Look at the straight line through the origin on this grid.",
    "The diagram shows a proportional relationship as a straight line through the origin.",
]
_P_ASK = [
    "Which equation matches the line?",
    "Which equation of the form y = mx describes the graph?",
    "Which of these equations is the rule for the line?",
    "Choose the equation that matches the graph.",
]
_P_NOTE = {
    "easy": [""],
    "medium": ["Each equation is written as y = mx or in a similar form."],
    "hard": [
        "The axes do not use the same scale, so use the axis labels and not the grid squares.",
        "Read the scale on each axis before you decide.",
    ],
}


@generator("fg1_proportional_graph", level=11, presets=_PRESETS)
def fg1_proportional_graph(rng, tier="easy", notes=False):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        win, M, C, pts = draw_line(
            rng,
            slopes=_P_SLOPES[tier],
            y_axis=True,
            origin=True,
            scales=_P_SCALES if tier == "hard" else ((1, 1),),
            min_points=2,
            min_du=1,
        )
        m, c = real_mc(win, M, C)
        check(c == 0, "a proportional line passes through the origin")
        others = [p for p in pts if p != (0, 0)]
        pb = rng.choice(others)
        fig = line_figure(win, [((0, 0), pb)], _ALT_LINE)
        rm, rc = read_line(fig)
        check((rm, rc) == (m, 0), "the drawn line is not the keyed one")
        key_text = eq_text(m, 0)
        cands = [
            (eq_text(-m, 0), "sign-error-on-gradient"),
            (eq_text(m, 1), "adds-an-unstated-intercept"),
            (eq_text(m + 1, 0), "gradient-off-by-one"),
        ]
        if abs(m).denominator == 1:
            cands.append((eq_text(1, abs(m)), "adds-an-unstated-intercept"))
        if abs(m) != 1:
            cands.append((eq_text(1 / m, 0), "inverted-gradient"))
        if tier == "hard":
            cands.append((eq_text(M, 0), "gradient-counted-in-squares"))
        texts, note_map = take_distractors(rng, key_text, cands)
        if len(texts) < 3:
            continue
        try:
            no_leak(fig, key_text, texts)
        except AssertionError:
            continue
        break
    else:
        raise ValueError("could not build a proportional-graph question")
    text = " ".join(p for p in (phrase(rng, _P_OPEN), phrase(rng, _P_ASK), phrase(rng, _P_NOTE[tier])) if p)
    x2, y2 = real_pt(win, pb)
    working = [
        f"The line passes through the origin and ({x2}, {y2}).",
        f"y is {rational_text(m)} times x, so the gradient is {rational_text(m)} and {key_text}.",
    ]
    return mcq_result(
        text=text, key_text=key_text, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
    )


def _read_proportional(question) -> str:
    m, c = read_line(question.figure)
    check(c == 0, "line does not pass through the origin")
    return eq_text(m, 0)


figures.register_reader("fg1_proportional_graph", _read_proportional)


# ----------------------------------------------------------------------------- table for the shown line

_T_SLOPES = {
    "easy": [F(1), F(2), F(3)],
    "medium": [F(-3), F(-2), F(-1), F(1, 2), F(-1, 2), F(2), F(3), F(1, 3), F(-1, 3)],
    "hard": [F(-3), F(-2), F(-1), F(2), F(3), F(1, 2), F(-1, 2), F(1, 3), F(-1, 3)],
}
_T_OPEN = [
    "The graph shows a straight line on a coordinate grid.",
    "A straight line is shown on the coordinate grid below.",
    "Look at the straight line shown on this grid.",
    "The diagram shows the graph of a straight line.",
]
_T_ASK = [
    "Which list of (x, y) pairs shows only points on the line?",
    "Which table of values, written as (x, y) pairs, matches the line?",
    "Every pair in one list lies on the line. Which list is it?",
    "Which set of (x, y) pairs belongs to the line?",
]
_T_NOTE = {
    "easy": [""],
    "medium": ["Each list has three pairs."],
    "hard": ["The y-axis is not shown, so check each pair against the grid you can see."],
}


def on_line(m: F, c: F, pts) -> bool:
    return all(F(y) == m * x + c for x, y in pts)


@generator("fg1_line_table", level=12, presets=_PRESETS)
def fg1_line_table(rng, tier="easy", notes=False):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(300):
        win, M, C, pts = draw_line(
            rng,
            slopes=_T_SLOPES[tier],
            c_range=(-3, 3),
            y_axis=False if tier == "hard" else True,
            c_visible=tier != "hard",
            c_nonzero=True,
            min_points=3,
            min_du=2,
        )
        m, c = real_mc(win, M, C)
        i = rng.randrange(len(pts) - 2)
        trio = [pts[i], pts[i + 1], pts[i + 2]]
        key_pts = [real_pt(win, p) for p in trio]
        check(on_line(m, c, key_pts), "the keyed pairs are not on the line")
        fig = line_figure(win, [(trio[0], trio[2])], _ALT_LINE)
        rm, rc = read_line(fig)
        check((rm, rc) == (m, c), "the drawn line is not the keyed one")
        xs = [x for x, _y in key_pts]

        def lst(fn, xs=xs):
            return [(x, int(fn(x))) for x in xs]

        cands = [
            (lst(lambda x: m * x), "intercept-left-out"),
            (lst(lambda x: -m * x + c), "sign-error-on-gradient"),
            (lst(lambda x: m * x - c), "sign-error-on-intercept"),
            ([(y, x) for x, y in key_pts], "coordinates-reversed"),
            ([(x, y + (1 if j == 1 else 0)) for j, (x, y) in enumerate(key_pts)], "one-pair-off-by-one-square"),
        ]
        if m.denominator == 1 and c.denominator == 1:
            cands.append((lst(lambda x: c * x + m), "gradient-and-intercept-swapped"))
        cands = [(pairs_text(p), n) for p, n in cands if not on_line(m, c, p)]
        key_text = pairs_text(key_pts)

        def outside(t, win=win):
            return any(
                not (win.x_range[0] <= x <= win.x_range[1] and win.y_range[0] <= y <= win.y_range[1])
                for x, y in parse_pairs(t)
            )

        texts, note_map = take_distractors(rng, key_text, cands, rank=outside)
        if len(texts) < 3:
            continue
        try:
            no_leak(fig, key_text, texts)
        except AssertionError:
            continue
        break
    else:
        raise ValueError("could not build a table question")
    text = " ".join(p for p in (phrase(rng, _T_OPEN), phrase(rng, _T_ASK), phrase(rng, _T_NOTE[tier])) if p)
    working = [
        f"Read the line: gradient {rational_text(m)}, meeting the y-axis at {rational_text(c)}.",
        f"So y = {rational_text(m)}x + {rational_text(c)}; check each pair against the line.",
        f"The pairs on the line are {key_text}.",
    ]
    return mcq_result(
        text=text, key_text=key_text, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
    )


def parse_pairs(s: str):
    nums = [int(v) for v in re.findall(r"-?\d+", s)]
    return list(zip(nums[0::2], nums[1::2]))


def _read_line_table(question) -> str:
    m, c = read_line(question.figure)
    options = [question.answer_text, *question.distractors]
    good = [o for o in options if on_line(m, c, parse_pairs(o))]
    check(len(good) == 1, "not exactly one option lies on the drawn line")
    return good[0]


figures.register_reader("fg1_line_table", _read_line_table)


# ----------------------------------------------------------------------------- plotted points


def _plane(rng):
    nu, nv = rng.randint(9, 10), rng.randint(9, 10)
    u0 = rng.randint(-(nu - 3), -2)
    v0 = rng.randint(-(nv - 3), -2)
    return Window(u0, u0 + nu - 1, v0, v0 + nv - 1)


def _points_figure(win: Window, points) -> dict:
    return figures.cartesian(
        x_range=win.x_range,
        y_range=win.y_range,
        alt=_ALT_POINTS,
        grid=1,
        equal_scale=True,
        points=points,
    )


_PP_ONE_OPEN = [
    "A point P is marked on the coordinate grid.",
    "The point P is shown on this coordinate grid.",
    "Look at the point P on the grid below.",
]
_PP_ONE_ASK = [
    "What are the coordinates of P?",
    "Which ordered pair gives the position of P?",
    "Which pair of coordinates (x, y) is the point P?",
]
_PP_TABLE_OPEN = [
    "Three points are marked on the coordinate grid.",
    "The grid shows three marked points that lie on a straight line.",
    "Three points on a straight line are marked on this grid.",
]
_PP_TABLE_ASK = [
    "Which list of (x, y) pairs gives the three points?",
    "Which table of values, written as (x, y) pairs, matches the marked points?",
    "Which set of ordered pairs matches the points on the grid?",
]
_PP_NEXT_OPEN = [
    "Three points are marked on the coordinate grid. They lie on a straight line.",
    "The grid shows three marked points on one straight line.",
    "Three marked points on this grid lie on the same straight line.",
]
_PP_NEXT_ASK = [
    "Which of these ordered pairs also lies on the same straight line?",
    "Which ordered pair is another point on the line through the marked points?",
    "Which of these points lies on the line that passes through the three marked points?",
]


def _one_point(rng, notes):
    for _ in range(ATT):
        win = _plane(rng)
        x = rng.randint(win.u0, win.u1)
        y = rng.randint(win.v0, win.v1)
        if x == 0 or y == 0 or abs(x) == abs(y):
            continue
        key = f"({x}, {y})"
        cands = [
            (f"({-x}, {y})", "sign-error-on-x"),
            (f"({x}, {-y})", "sign-error-on-y"),
            (f"({-x}, {-y})", "both-signs-wrong"),
        ]
        # reversed coordinates are the misconception this question is about: always offered
        texts, note_map = take_distractors(rng, key, cands, n=2)
        texts.insert(rng.randrange(3), f"({y}, {x})")
        note_map[f"({y}, {x})"] = "coordinates-reversed"
        fig = _points_figure(win, [{"at": [x, y], "label": "P"}])
        try:
            no_leak(fig, key, texts)
        except AssertionError:
            continue
        text = f"{phrase(rng, _PP_ONE_OPEN)} {phrase(rng, _PP_ONE_ASK)}"
        working = [f"From the origin, P is {abs(x)} across and {abs(y)} up or down: {key}."]
        return mcq_result(
            text=text, key_text=key, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a plotted-point question")


def _line_trio(rng, slopes):
    for _ in range(ATT):
        win = _plane(rng)
        M = rng.choice(slopes)
        C = rng.randint(-3, 3)
        xs = [x for x in range(win.u0, win.u1 + 1) if (M * x + C).denominator == 1 and win.v0 <= M * x + C <= win.v1]
        if len(xs) < 3:
            continue
        i = rng.randrange(len(xs) - 2)
        trio = [(x, int(M * x + C)) for x in xs[i : i + 3]]
        if trio[1][0] - trio[0][0] != trio[2][0] - trio[1][0]:
            continue
        return win, M, C, trio
    raise ValueError("no plotted trio")


def _three_points(rng, notes):
    for _ in range(ATT):
        win, M, C, trio = _line_trio(rng, [F(s) for s in (-3, -2, -1, 1, 2, 3)])
        fig = _points_figure(win, [{"at": [x, y]} for x, y in trio])
        key = pairs_text(trio)
        cands = [
            (pairs_text([(y, x) for x, y in trio]), "coordinates-reversed"),
            (pairs_text([(-x, y) for x, y in trio]), "sign-error-on-x"),
            (
                pairs_text([(x, y + (1 if j == 1 else 0)) for j, (x, y) in enumerate(trio)]),
                "one-point-off-by-one-square",
            ),
            (pairs_text([(x, y) if j != 1 else (y, x) for j, (x, y) in enumerate(trio)]), "one-pair-reversed"),
        ]
        texts, note_map = take_distractors(rng, key, cands)
        if len(texts) < 3:
            continue
        try:
            no_leak(fig, key, texts)
        except AssertionError:
            continue
        text = f"{phrase(rng, _PP_TABLE_OPEN)} {phrase(rng, _PP_TABLE_ASK)}"
        working = [f"Read each point as (across, up): {key}."]
        return mcq_result(
            text=text, key_text=key, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a plotted-points question")


def _next_point(rng, notes):
    for _ in range(ATT):
        win, M, C, trio = _line_trio(rng, [F(s) for s in (-3, -2, -1, 1, 2, 3)])
        fig = _points_figure(win, [{"at": [x, y]} for x, y in trio])
        step = trio[1][0] - trio[0][0]
        k = rng.randint(2, 5)
        side = rng.choice((-1, 1))
        xn = trio[-1][0] + k * step if side > 0 else trio[0][0] - k * step
        yn = int(M * xn + C)
        if abs(yn) > 25 or win.u0 <= xn <= win.u1:
            continue  # the further point lies off the grid, so it is found by continuing the pattern
        xl, yl = trio[-1] if side > 0 else trio[0]
        key = f"({xn}, {yn})"
        cands = [
            (f"({yn}, {xn})", "coordinates-reversed"),
            (f"({xn}, {-yn})", "sign-error-on-y"),
            (f"({xn}, {yl + int(M) * (1 if side > 0 else -1)})", "adds-the-rise-once"),
            (f"({xn}, {yn + int(M)})", "one-step-too-far"),
            (f"({xn}, {yn - int(M)})", "one-step-short"),
        ]
        cands = [(t, n) for t, n in cands if F(parse_pairs(t)[0][1]) != M * parse_pairs(t)[0][0] + C]
        texts, note_map = take_distractors(rng, key, cands)
        if len(texts) < 3:
            continue
        try:
            no_leak(fig, key, texts)
        except AssertionError:
            continue
        text = f"{phrase(rng, _PP_NEXT_OPEN)} {phrase(rng, _PP_NEXT_ASK)}"
        working = [
            f"Read the points: {pairs_text(trio)}.",
            f"Each step of {step} in x changes y by {int(M) * step}, so the gradient is {int(M)}.",
            f"At x = {xn}, y = {yn}: {key}.",
        ]
        return mcq_result(
            text=text, key_text=key, texts=texts, notes=note_map, working=working, fig=fig, option_notes=notes
        )
    raise ValueError("could not build a further-point question")


@generator("fg1_plotted_points", level=10, presets=_PRESETS)
def fg1_plotted_points(rng, tier="easy", notes=False):
    if tier == "easy":
        return _one_point(rng, notes)
    if tier == "medium":
        return _three_points(rng, notes)
    if tier == "hard":
        return _next_point(rng, notes)
    raise ValueError(f"unknown tier {tier!r}")


def _read_plotted(question) -> str:
    fig = question.figure
    pts = [tuple(p["at"]) for p in fig["points"]]
    options = [question.answer_text, *question.distractors]
    if len(pts) == 1:
        want = f"({pts[0][0]}, {pts[0][1]})"
        check(want in options, "the read point is not among the options")
        return want
    (x1, y1), (x2, y2), (x3, y3) = sorted(pts)
    check(x2 - x1 == x3 - x2 and (y2 - y1) == (y3 - y2), "the three points are not evenly spaced on a line")
    # a table question lists the points; a further-point question lists one pair that continues them
    listed = pairs_text(sorted(pts))
    if listed in options:
        return listed
    run, rise = x2 - x1, y2 - y1
    good = []
    for o in options:
        ((x, y),) = parse_pairs(o)
        if (y - y1) * run == rise * (x - x1):
            good.append(o)
    check(len(good) == 1, "not exactly one option lies on the line through the points")
    return good[0]


figures.register_reader("fg1_plotted_points", _read_plotted)
