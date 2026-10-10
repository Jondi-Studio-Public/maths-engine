"""Generates graph-interpretation questions for conversion graphs (ACMEM124), `answer: number`.

One skill: ``senior-interpret-and-use-graphs-in-practical-situations-including``
(ACMEM124), ``domain: senior-mathematics``, ``answer: number`` (an integer,
graded exactly; the prompt states the unit to answer in). A conversion graph is
drawn on a labelled grid; the prompt states ONE numeral (the input) followed by
its unit, names both units, and names the direction. The stated input and the
converted value both land on grid lines, so nothing is estimated.

Only true unit conversions (hours to minutes, feet to inches...) or an explicitly scenario-specific rule are
ever drawn: "an exchange desk today" (a rate and flat fee that are only that desk's), or "a hiker's rule of
thumb" for estimating degrees Fahrenheit (treated as exact for the question). Nothing is stated as a general
fact that is false.

Four tiers, escalating by the DECISION:

- ``easy``: a proportional unit conversion drawn through the origin; convert
  the stated input (read at a labelled tick) to the other unit.
- ``medium``: the same kind of graph read in REVERSE (the stated value is on
  the vertical axis, the answer on the horizontal); includes an exchange desk
  with no fee. The decision is which axis the stated value is on.
- ``hard``: an offset conversion (the line does not pass through the origin: a
  rule of thumb with a constant, or a desk that takes a flat fee), forward.
  Proportional reasoning from one point fails.
- ``exam``: the offset conversion read in reverse on stepped unequal scales
  (both axes step by more than one unit, and not equally).

Distractors are named misconceptions: reading the stated number on the wrong
axis; proportional reasoning ignoring the offset; repeating the stated number
unconverted; the adjacent grid line either side; counting grid squares instead
of reading the axis. All are integers different from the key.

The plausibility reader takes the line from ``question.figure`` and the stated
input and its unit from the prompt (the numeral followed by a unit named in an
axis label), so it re-derives the key as the learner does.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import figures
from mathgen.core import check, check_pairwise_distinct, generator, phrase
from mathgen.generators.pv1a_line_figure_build import (
    Window,
    lattice_points,
    line_figure,
    pick_two_points,
    read_line,
)

_ATTEMPTS = 3000
_ALT = "a straight line on a coordinate grid"
_F = Fraction
_ALLOWED_M = (_F(1, 2), _F(1), _F(3, 2), _F(2), _F(3))

# --- proportional TRUE unit conversions -------------------------------------
# (quantity, x unit, y unit, y units per x unit, x steps, y steps)
_UNITS = (
    ("Time", "hours", "minutes", 60, (1,), (20, 30, 40, 60, 120)),
    ("Length", "feet", "inches", 12, (1,), (4, 6, 8, 12, 24)),
    ("Length", "metres", "centimetres", 100, (1,), (50, 100, 200)),
    ("Distance", "kilometres", "metres", 1000, (1,), (500, 1000, 2000)),
    ("Mass", "kilograms", "grams", 1000, (1,), (500, 1000, 2000)),
    ("Volume", "litres", "millilitres", 1000, (1,), (500, 1000, 2000)),
    ("Time", "minutes", "seconds", 60, (1,), (20, 30, 40, 60, 120)),
    ("Time", "days", "hours", 24, (1,), (8, 12, 16, 24, 48)),
    ("Time", "weeks", "days", 7, (1, 2), (7, 14)),
    ("Length", "yards", "feet", 3, (1,), (1, 2, 3, 6)),
)

# --- scenario-specific exchange desks (rate k, y per x) ----------------------
# (x long name, x code, y long name, y code, k, x steps, y steps)
_DESKS = (
    ("euros", "EUR", "Australian dollars", "AUD", _F(3, 2), (10, 20), (5, 10, 15, 20, 30)),
    ("British pounds", "GBP", "Australian dollars", "AUD", _F(2), (5, 10), (5, 10, 20)),
    ("Australian dollars", "AUD", "euros", "EUR", _F(3, 5), (10, 20), (2, 3, 4, 6, 12)),
)

# --- the rule of thumb for estimating degrees Fahrenheit ---------------------
_RULE = dict(k=_F(2), b=30, xs=(5, 10), ys=(5, 10))

_STORY_RULE = (
    "The graph shows a hiker's rule of thumb for estimating a temperature in degrees Fahrenheit "
    "from a temperature in degrees Celsius; treat the rule as exact."
)
_STORY_DESK = "At one exchange desk today, the graph converts {xl} ({xc}) to {yl} ({yc})."
_STORY_FEE = (
    "At one exchange desk today, the graph shows the {yl} ({yc}) received for {xl} ({xc}) handed over, "
    "after the desk's flat fee is taken."
)

_T_EASY = [
    "The graph converts {xu} to {yu}. Use it to convert {n} {iu} to {ou}. Give your answer in {ou}.",
    "Use the conversion graph to change {n} {iu} into {ou}.",
    "How many {ou} are equal to {n} {iu}? Read it from the conversion graph.",
    "A conversion graph for {xu} and {yu} is shown. What is {n} {iu} in {ou}?",
]
_T_MEDIUM = [
    "Use the graph in reverse: how many {ou} are equal to {n} {iu}?",
    "An amount is {n} {iu}. Read the graph backwards to find the same amount in {ou}.",
    "Which number of {ou} matches {n} {iu} on this conversion graph?",
]
_T_HARD = [
    "Use the graph to convert {n} {iu} to {ou}.",
    "What value in {ou} does the graph give for {n} {iu}?",
    "Read the graph at {n} {iu} to find the matching value in {ou}.",
]
_T_EXAM = [
    "The value is {n} {iu}. Reading the graph the other way round, what is the matching value in {ou}?",
    "Which value in {ou} gives {n} {iu} on this graph? Answer in {ou}.",
    "Work backwards from {n} {iu} to the matching value in {ou}.",
]


def _contexts(tier: str):
    """Candidate contexts for the tier as dicts with k, b-choice, steps, labels, units, story."""
    out = []
    if tier in ("easy", "medium"):
        for q, xu, yu, k, xss, yss in _UNITS:
            out.append(
                dict(
                    kind="unit",
                    k=_F(k),
                    c_choices=(0,),
                    xs=xss,
                    ys=yss,
                    xu=xu,
                    yu=yu,
                    x_label=f"{q} ({xu})",
                    y_label=f"{q} ({yu})",
                    story="",
                )
            )
    if tier == "medium":
        for xl, xc, yl, yc, k, xss, yss in _DESKS:
            out.append(
                dict(
                    kind="desk",
                    k=k,
                    c_choices=(0,),
                    xs=xss,
                    ys=yss,
                    xu=xc,
                    yu=yc,
                    x_label=f"Amount handed over ({xc})",
                    y_label=f"Amount received ({yc})",
                    story=_STORY_DESK.format(xl=xl, xc=xc, yl=yl, yc=yc),
                )
            )
    if tier in ("hard", "exam"):
        out.append(
            dict(
                kind="rule",
                k=_RULE["k"],
                c_choices=("rule",),
                xs=_RULE["xs"],
                ys=_RULE["ys"],
                xu="degrees Celsius",
                yu="degrees Fahrenheit",
                x_label="Temperature (degrees Celsius)",
                y_label="Estimated temperature (degrees Fahrenheit)",
                story=_STORY_RULE,
            )
        )
        for xl, xc, yl, yc, k, xss, yss in _DESKS:
            out.append(
                dict(
                    kind="fee",
                    k=k,
                    c_choices=(-1, -2),
                    xs=xss,
                    ys=yss,
                    xu=xc,
                    yu=yc,
                    x_label=f"Amount handed over ({xc})",
                    y_label=f"Amount received ({yc})",
                    story=_STORY_FEE.format(xl=xl, xc=xc, yl=yl, yc=yc),
                )
            )
    return out


def _combos(ctx, tier: str):
    for xs in ctx["xs"]:
        for ys in ctx["ys"]:
            if tier == "exam" and (xs == 1 or ys == 1 or xs == ys):
                continue
            if ctx["k"] * xs / ys in _ALLOWED_M:
                yield xs, ys


def _build(rng, tier: str) -> dict:
    forward = tier in ("easy", "hard")
    contexts = _contexts(tier)
    for _ in range(_ATTEMPTS):
        ctx = rng.choice(contexts)
        combos = list(_combos(ctx, tier))
        if not combos:
            continue
        xs, ys = rng.choice(combos)
        M = ctx["k"] * xs / ys
        c_pick = rng.choice(ctx["c_choices"])
        if c_pick == "rule":
            C = _F(_RULE["b"], ys)
        else:
            C = _F(c_pick)
        nu, nv = rng.randint(8, 10), rng.randint(8, 10)
        win = Window(0, nu - 1, 0, nv - 1, xs=xs, ys=ys)
        pts = lattice_points(win, M, C)
        try:
            through = pick_two_points(rng, pts, min_du=2)
        except ValueError:
            continue
        queries = [p for p in pts if p not in through and p[0] >= 1 and p[1] >= 1]
        if not queries:
            continue
        uq, vq = rng.choice(queries)
        fig = line_figure(win, [through], _ALT, x_label=ctx["x_label"], y_label=ctx["y_label"])
        m_real, c_real = read_line(fig)
        in_val = uq * xs if forward else vq * ys
        key = vq * ys if forward else uq * xs
        if in_val == 1:
            continue  # "1 hours" would read badly; the stated number is always plural
        # Independent route: re-derive from the figure's line.
        got = (m_real * in_val + c_real) if forward else (in_val - c_real) / m_real
        check(got == key, "the drawn line does not give the key")

        cands = _candidates(rng, forward, in_val, key, m_real, c_real, win, through, (uq, vq))
        seen = {key}
        picked = []
        for val, note in cands:
            if val < 1 or val in seen:
                continue
            seen.add(val)
            picked.append((str(val), note))
        if len(picked) < 2:
            continue
        picked = picked[:3]
        check_pairwise_distinct(key, *[int(t) for t, _ in picked])
        texts = [t for t, _ in picked]
        try:
            figures.assert_no_leak(fig, [str(key), *texts])
        except AssertionError:
            continue
        break
    else:
        raise ValueError(f"pv1a_conversion: no valid {tier} draw")

    iu, ou = (ctx["xu"], ctx["yu"]) if forward else (ctx["yu"], ctx["xu"])
    slots = dict(n=in_val, iu=iu, ou=ou, xu=ctx["xu"], yu=ctx["yu"])
    bank = {"easy": _T_EASY, "medium": _T_MEDIUM, "hard": _T_HARD, "exam": _T_EXAM}[tier]
    prefix = f"{ctx['story']} " if ctx["story"] else ""
    text = prefix + phrase(rng, bank, **slots)
    (p1, p2) = through
    working = [
        f"The line passes through ({p1[0] * xs}, {p1[1] * ys}) and ({p2[0] * xs}, {p2[1] * ys}).",
        (
            f"Find {in_val} on the {'horizontal' if forward else 'vertical'} axis ({iu}), go to the line, "
            f"then across to the {'vertical' if forward else 'horizontal'} axis ({ou})."
        ),
        f"Reading the line: {iu} = {in_val} matches {ou} = {key}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=key,
        answer_text=str(key),
        answer_latex=str(key),
        working=working,
        distractors=texts,
        option_notes={t: n for t, n in picked},
        figure=fig,
    )


def _candidates(rng, forward, q, key, m, c, win, through, query):
    """(value, note id) per named misconception, in a seed-shuffled order of the
    main ones; integer values only (the key is an integer, graded exactly)."""
    xs, ys = win.xs, win.ys
    out = []
    # reads the stated number on the wrong axis
    if forward:
        if win.y_range[0] <= q <= win.y_range[1]:
            v = (Fraction(q) - c) / m
            if v.denominator == 1:
                out.append((int(v), "stated-number-read-on-the-wrong-axis"))
    elif win.x_range[0] <= q <= win.x_range[1]:
        v = m * q + c
        if v.denominator == 1:
            out.append((int(v), "stated-number-read-on-the-wrong-axis"))
    # proportional reasoning from a through point, ignoring the offset
    for u, v in through:
        x, y = u * xs, v * ys
        if x > 0 and y > 0:
            r = Fraction(q) * y / x if forward else Fraction(q) * x / y
            if r.denominator == 1:
                out.append((int(r), "proportional-reasoning-ignores-the-offset"))
                break
    out.append((q, "repeats-the-stated-number-unconverted"))
    step = ys if forward else xs
    out.append((key + step, "adjacent-grid-line-above"))
    out.append((key - step, "adjacent-grid-line-below"))
    count = query[1] if forward else query[0]
    if (ys if forward else xs) != 1:
        out.append((count, "counts-grid-squares-not-axis-units"))
    rng.shuffle(out)
    return out


@generator(
    "pv1a_conversion",
    level=10,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def pv1a_conversion(rng, tier="easy", notes=False):
    if tier not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"unknown tier {tier!r}")
    out = _build(rng, tier)
    if not notes:
        out.pop("option_notes")
    return out


def read_pv1a_conversion(question) -> str:
    """Plausibility reader: the line from the figure, the stated input and its
    unit from the prompt (the numeral followed by a unit named in an axis label)."""
    spec = question.figure
    m, c = read_line(spec)
    units = []
    for key in ("x_label", "y_label"):
        found = re.search(r"\(([^)]+)\)\s*$", spec[key])
        check(found is not None, "axis label names no unit")
        units.append(found.group(1))
    for axis, unit in enumerate(units):
        hit = re.search(r"(-?\d+)\s*" + re.escape(unit) + r"(?![A-Za-z])", question.text)
        if hit:
            n = Fraction(int(hit.group(1)))
            value = (m * n + c) if axis == 0 else (n - c) / m
            return str(int(value)) if value.denominator == 1 else str(value)
    raise ValueError("the prompt states no input in an axis unit")


figures.register_reader("pv1a_conversion", read_pv1a_conversion)
