"""Generates `w9_numberline_inequality` questions (easy, medium, hard) and the exam chain
`exd_x_w9_numberline_inequality`.

``inequality-on-number-line`` (``inequalities``, answer ``number_line_set``): the learner marks the solution of an
inequality on a number line with open and closed circles and shaded rays or segments. The front end sends the pieces
(``{"pieces": [{"from": null, "to": 3, "to_closed": false}]}``); the grader merges and compares them exactly, so a ray
drawn in two goes is the same as one ray. The typed form is the solution set written as inequalities (``x < 3``,
``-2 < x ≤ 5``, ``x < -2 or x ≥ 3``).

* easy: show a stated inequality (``x > 3``, ``x ≤ -2``) or a sentence ("all numbers less than 4").
* medium: solve a one- or two-step inequality whose coefficient is positive, then show it; or show a compound
  inequality ``a < x ≤ b``.
* hard: solve an inequality with a negative coefficient (the sign reverses), solve an absolute-value inequality, or solve two
  inequalities joined by "or" or "and".
* exam (``exd_x_w9_numberline_inequality``): solve, show the solution, show it combined with a second condition, and
  count the whole numbers that satisfy both.

Each answer is built from the bound first and the inequality derived from it, then checked a second way by testing
whole numbers either side of every bound against the original inequality. Every wrong picture is checked against
the `number_line_set` grader before it is offered.
"""

from __future__ import annotations

from fractions import Fraction

from .. import figures
from ..core import attempts, check, generator, phrase
from ..figure_kinds.canvas import canvas as _canvas
from ..grading import interactive as ia
from ._w6a_kit import num_dist, num_fields
from ._w9_graph_kit import dist, enough, frac_text, part, retry_draw

TOPIC = "w9_numberline_inequality"
EXAM_TOPIC = "exd_x_w9_numberline_inequality"

AREA = "inequality-on-number-line"
MISCONCEPTIONS = [
    (
        "w9-ray-wrong-direction",
        "Shades the ray towards the wrong end of the number line, so it shows the opposite inequality.",
        AREA,
    ),
    (
        "w9-circle-open-closed-swapped",
        "Uses an open circle where the bound is included (or a closed circle where it is not).",
        AREA,
    ),
    ("w9-ray-and-circle-wrong", "Shades the wrong way and also uses the wrong kind of circle.", AREA),
    ("w9-bound-sign-ignored", "Puts the bound on the wrong side of zero, as if its sign did not matter.", AREA),
    (
        "w9-bound-point-only",
        "Marks only the bound itself and does not shade the numbers that satisfy the inequality.",
        AREA,
    ),
    ("w9-ineq-forgets-to-divide", "Moves the constant across but does not divide by the coefficient of x.", AREA),
    (
        "w9-ineq-constant-sign",
        "Adds the constant to the right-hand side instead of subtracting it (or the reverse).",
        AREA,
    ),
    ("w9-ineq-divides-wrongly", "Multiplies by the coefficient of x instead of dividing by it.", AREA),
    ("w9-ineq-sign-not-reversed", "Divides by a negative number without reversing the inequality sign.", AREA),
    (
        "w9-ineq-negative-bound-sign",
        "Divides by a negative number and keeps the sign of the bound, so the bound is on the wrong side of zero.",
        AREA,
    ),
    (
        "w9-compound-wrong-end-open",
        "Draws a compound inequality with the circles the wrong way round: closed where open and open where closed.",
        AREA,
    ),
    (
        "w9-compound-one-ray",
        "Shades only a ray from one end of a compound inequality, not the segment between the two bounds.",
        AREA,
    ),
    ("w9-compound-outside", "Shades outside the two bounds instead of between them.", AREA),
    (
        "w9-abs-centre-sign",
        "Takes the centre of |x - a| as -a, so the interval is centred on the wrong side of zero.",
        AREA,
    ),
    ("w9-abs-centre-ignored", "Ignores the shift in |x - a| and centres the interval on zero.", AREA),
    (
        "w9-abs-union-for-intersection",
        "Shades the two outer rays for a 'less than' absolute-value inequality, which needs the segment between them.",
        AREA,
    ),
    (
        "w9-abs-intersection-for-union",
        "Shades the segment between the bounds for a 'greater than' absolute-value inequality, which needs the two outer rays.",
        AREA,
    ),
    (
        "w9-abs-one-side-only",
        "Solves only one of the two cases of an absolute-value inequality, so only one bound or ray is shown.",
        AREA,
    ),
    (
        "w9-or-drawn-as-and",
        "Shades only where both inequalities hold when the word was 'or', or shades only one of the two rays.",
        AREA,
    ),
    ("w9-and-drawn-as-or", "Shades everywhere either inequality holds when the word was 'and'.", AREA),
    ("w9-integer-count-includes-open-bound", "Counts the whole number at an open circle as a solution.", AREA),
    (
        "w9-integer-count-misses-closed-bound",
        "Leaves out the whole number at a closed circle when counting solutions.",
        AREA,
    ),
    (
        "w9-integer-count-fencepost",
        "Counts the gaps between the bounds instead of the whole numbers, so is one out.",
        AREA,
    ),
    (
        "w9-integer-greatest-open-bound",
        "Gives the bound itself as the greatest whole number even though the inequality is strict.",
        AREA,
    ),
]

LOW, HIGH = -10, 10
RANGE = (LOW, HIGH)

_ALT = "A number line from -10 to 10 with a tick at every whole number, for marking the solution of an inequality."


def _figure():
    return figures.number_line(
        range_=RANGE, alt=_ALT, tick=1, canvas=_canvas("number_line_set", ["number_line_marks"], 1)
    )


def P(lo, lc, hi, hc):
    return (None if lo is None else Fraction(lo), lc, None if hi is None else Fraction(hi), hc)


def ray_left(a, closed):
    return P(None, False, a, closed)


def ray_right(a, closed):
    return P(a, closed, None, False)


def _fields(pieces) -> dict:
    text = ia.pieces_text(_merged(pieces))
    return dict(
        answer=ia.number_line_answer(pieces, range_=RANGE, snap=1),
        answer_text=text,
        answer_latex="plain:" + text,
        answer_kind="number_line_set",
    )


def _merged(pieces):
    return ia._expected_pieces(ia.number_line_answer(pieces, range_=RANGE, snap=1))


def _dist(fields, candidates, n=4):
    """`candidates` is ``[(pieces, tag), ...]``. Pieces off the line or empty are skipped."""
    cands = []
    for pieces, tag in candidates:
        if any(abs(v) > HIGH for p in pieces for v in (p[0], p[2]) if v is not None):
            continue
        try:
            text = ia.pieces_text(_merged(pieces))
            value = ia.number_line_answer(pieces, range_=RANGE, snap=1)
        except ValueError:
            continue
        cands.append((text, value, tag))
    return dist("number_line_set", fields["answer"], fields["answer_text"], cands, n)


def _holds(pieces, x: Fraction) -> bool:
    for lo, lc, hi, hc in pieces:
        above = lo is None or x > lo or (lc and x == lo)
        below = hi is None or x < hi or (hc and x == hi)
        if above and below:
            return True
    return False


_OPS = {"<": (False, "left"), "≤": (True, "left"), ">": (False, "right"), "≥": (True, "right")}


def _single(op: str, a):
    closed, side = _OPS[op]
    return [ray_left(a, closed) if side == "left" else ray_right(a, closed)]


def _flip(op: str) -> str:
    return {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}[op]


def _simple_wrong(op: str, a: int):
    """Wrong pictures for the single inequality x `op` a."""
    closed, side = _OPS[op]
    other = "right" if side == "left" else "left"
    mk = lambda s, c, v: [ray_left(v, c) if s == "left" else ray_right(v, c)]  # noqa: E731
    out = [
        (mk(other, closed, a), "w9-ray-wrong-direction"),
        (mk(side, not closed, a), "w9-circle-open-closed-swapped"),
        (mk(other, not closed, a), "w9-ray-and-circle-wrong"),
        (mk(side, closed, -a), "w9-bound-sign-ignored"),
        ([P(a, True, a, True)], "w9-bound-point-only"),
    ]
    return out


# ---------------------------------------------------------------------------------------------- easy


def _easy(rng):
    a = rng.choice([v for v in range(-8, 9) if v])
    op = rng.choice(list(_OPS))
    pieces = _single(op, a)
    style = rng.choice(("symbol", "symbol", "words"))
    if style == "symbol":
        text = phrase(
            rng,
            (
                "Show x {op} {a} on the number line.",
                "Draw the inequality x {op} {a} on the number line.",
                "Mark the solutions of x {op} {a} on a number line.",
            ),
            op=op,
            a=a,
        )
    else:
        words = {
            "<": "less than",
            "≤": "less than or equal to",
            ">": "greater than",
            "≥": "greater than or equal to",
        }[op]
        text = phrase(
            rng,
            (
                "Show all the numbers x that are {w} {a} on the number line.",
                "Mark every number that is {w} {a} on the number line.",
            ),
            w=words,
            a=a,
        )
    check(
        all(_holds(pieces, Fraction(x)) == _satisfies(op, x, a) for x in range(-10, 11)),
        "the picture disagrees with the inequality",
    )
    fields = _fields(pieces)
    d = _dist(fields, _simple_wrong(op, a))
    enough(d, 3)
    closed = op in "≤≥"
    working = [
        f"The bound is {a}. The sign {op} {'includes' if closed else 'does not include'} {a}, so the circle at {a} is {'closed (filled)' if closed else 'open (hollow)'}.",
        f"Shade {'to the left' if op in '<≤' else 'to the right'}, towards the numbers that are {'smaller' if op in '<≤' else 'larger'} than {a}.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


def _satisfies(op: str, x, a) -> bool:
    return {"<": x < a, "≤": x <= a, ">": x > a, "≥": x >= a}[op]


# ---------------------------------------------------------------------------------------------- medium


def _lin(k: int, d: int, op: str, rhs: int) -> str:
    left = f"{k}x" + (f" + {d}" if d > 0 else f" - {-d}" if d < 0 else "")
    return f"{left} {op} {rhs}"


def _medium_solve(rng):
    for _ in attempts(300, "a two-step inequality with an integer bound"):
        a = rng.choice([v for v in range(-6, 7) if v])
        k = rng.randint(2, 5)
        d = rng.choice([v for v in range(-9, 10) if v])
        op = rng.choice(list(_OPS))
        rhs = k * a + d
        if abs(rhs) <= 30:
            break
    pieces = _single(op, a)
    check(
        all(_satisfies(op, Fraction(k * x + d), rhs) == _holds(pieces, Fraction(x)) for x in range(-10, 11)),
        "the solution set is wrong",
    )
    text = phrase(
        rng,
        (
            "Solve {ineq}. Show the solution on the number line.",
            "Solve the inequality {ineq} and show the solution on a number line.",
        ),
        ineq=_lin(k, d, op, rhs),
    )
    fields = _fields(pieces)
    cands = [
        (_single(op, rhs - d), "w9-ineq-forgets-to-divide"),
        (_single(op, Fraction(rhs + d, k)) if (rhs + d) % k == 0 else [], "w9-ineq-constant-sign"),
        (_single(op, rhs * k) if abs(rhs * k) <= HIGH else [], "w9-ineq-divides-wrongly"),
        (_single(_flip(op), a), "w9-ray-wrong-direction"),
        (_single({"<": "≤", "≤": "<", ">": "≥", "≥": ">"}[op], a), "w9-circle-open-closed-swapped"),
    ]
    d_ = _dist(fields, [(p, t) for p, t in cands if p])
    enough(d_, 3)
    working = [
        f"Subtract {d} from both sides: {k}x {op} {rhs - d}."
        if d > 0
        else f"Add {-d} to both sides: {k}x {op} {rhs - d}.",
        f"Divide both sides by {k}, which is positive, so the sign stays: x {op} {a}.",
        f"On the number line the circle at {a} is {'closed' if op in '≤≥' else 'open'} and the ray goes {'left' if op in '<≤' else 'right'}.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d_, **fields)


def _medium_compound(rng):
    for _ in attempts(100, "two different bounds"):
        a = rng.randint(-8, 3)
        b = a + rng.randint(2, 8)
        if b <= 8:
            break
    lc, hc = rng.random() < 0.5, rng.random() < 0.5
    pieces = [P(a, lc, b, hc)]
    low_op, high_op = ("≤" if lc else "<"), ("≤" if hc else "<")
    text = phrase(
        rng,
        ("Show {a} {lo} x {hi} {b} on the number line.", "Draw {a} {lo} x {hi} {b} on a number line."),
        a=a,
        b=b,
        lo=low_op,
        hi=high_op,
    )
    check(
        all(
            _holds(pieces, Fraction(x)) == ((a < x or (lc and x == a)) and (x < b or (hc and x == b)))
            for x in range(-10, 11)
        ),
        "compound mismatch",
    )
    fields = _fields(pieces)
    cands = [
        ([P(a, not lc, b, not hc)], "w9-compound-wrong-end-open"),
        ([ray_right(a, lc)], "w9-compound-one-ray"),
        ([ray_left(b, hc)], "w9-compound-one-ray"),
        ([ray_left(a, not lc), ray_right(b, not hc)], "w9-compound-outside"),
        ([P(-b, hc, -a, lc)], "w9-bound-sign-ignored"),
        ([P(a, lc, b, not hc)], "w9-circle-open-closed-swapped"),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    working = [
        f"x is between {a} and {b}, so shade the segment from {a} to {b}.",
        f"At {a} the sign is {low_op}, so the circle is {'closed' if lc else 'open'}. At {b} the sign is {high_op}, so the circle is {'closed' if hc else 'open'}.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


def _medium(rng):
    return _medium_solve(rng) if rng.random() < 0.6 else _medium_compound(rng)


# ---------------------------------------------------------------------------------------------- hard


def _hard_negative(rng):
    for _ in attempts(300, "a negative-coefficient inequality with an integer bound"):
        a = rng.choice([v for v in range(-6, 7) if v])
        k = rng.randint(2, 5)
        d = rng.choice([v for v in range(-8, 9) if v])
        op = rng.choice(list(_OPS))
        rhs = -k * a + d
        if abs(rhs) <= 30:
            break
    final = _flip(op)
    pieces = _single(final, a)
    check(
        all(_satisfies(op, Fraction(-k * x + d), rhs) == _holds(pieces, Fraction(x)) for x in range(-10, 11)),
        "the solution set is wrong",
    )
    left = f"-{k}x" + (f" + {d}" if d > 0 else f" - {-d}")
    text = phrase(
        rng,
        ("Solve {ineq} and show the solution on a number line.", "Solve {ineq}. Show the solution on the number line."),
        ineq=f"{left} {op} {rhs}",
    )
    fields = _fields(pieces)
    cands = [
        (_single(op, a), "w9-ineq-sign-not-reversed"),
        (_single(final, -a), "w9-ineq-negative-bound-sign"),
        (_single(op, -a), "w9-ineq-sign-not-reversed"),
        (_single(final, Fraction(rhs + d, -k)) if (rhs + d) % k == 0 else [], "w9-ineq-constant-sign"),
        (_single(final, rhs - d) if abs(rhs - d) <= HIGH else [], "w9-ineq-forgets-to-divide"),
    ]
    d_ = _dist(fields, [(p, t) for p, t in cands if p])
    enough(d_, 3)
    working = [
        f"{'Subtract' if d > 0 else 'Add'} {abs(d)} from both sides: -{k}x {op} {rhs - d}.",
        f"Divide both sides by -{k}. Dividing by a negative number reverses the sign: x {final} {a}.",
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d_, **fields)


def _hard_absolute(rng):
    for _ in attempts(300, "an absolute-value inequality that fits the line"):
        c = rng.randint(-4, 4)
        r = rng.randint(2, 5)
        if abs(c) + r <= 9:
            break
    strict_less = rng.random() < 0.5
    closed = rng.random() < 0.5
    sym = ("≤" if closed else "<") if strict_less else ("≥" if closed else ">")
    inner = "x" if c == 0 else (f"x - {c}" if c > 0 else f"x + {-c}")
    pieces = [P(c - r, closed, c + r, closed)] if strict_less else [ray_left(c - r, closed), ray_right(c + r, closed)]
    for x in range(-10, 11):
        want = (
            abs(x - c) < r
            if strict_less and not closed
            else abs(x - c) <= r
            if strict_less
            else abs(x - c) > r
            if not closed
            else abs(x - c) >= r
        )
        check(_holds(pieces, Fraction(x)) == want, "absolute-value solution is wrong")
    text = phrase(
        rng,
        (
            "Solve |{inner}| {sym} {r} and show the solution on a number line.",
            "Show the solution of |{inner}| {sym} {r} on a number line.",
        ),
        inner=inner,
        sym=sym,
        r=r,
    )
    fields = _fields(pieces)
    shifted_wrong = (
        [P(-c - r, closed, -c + r, closed)] if strict_less else [ray_left(-c - r, closed), ray_right(-c + r, closed)]
    )
    centred = [P(-r, closed, r, closed)] if strict_less else [ray_left(-r, closed), ray_right(r, closed)]
    cands = [
        (shifted_wrong if c else [], "w9-abs-centre-sign"),
        (centred if c else [], "w9-abs-centre-ignored"),
        (
            [ray_left(c - r, closed), ray_right(c + r, closed)] if strict_less else [P(c - r, closed, c + r, closed)],
            "w9-abs-union-for-intersection" if strict_less else "w9-abs-intersection-for-union",
        ),
        ([ray_left(c + r, closed)] if strict_less else [ray_right(c + r, closed)], "w9-abs-one-side-only"),
        (
            [P(c - r, not closed, c + r, not closed)]
            if strict_less
            else [ray_left(c - r, not closed), ray_right(c + r, not closed)],
            "w9-circle-open-closed-swapped",
        ),
    ]
    d = _dist(fields, [(p, t) for p, t in cands if p])
    enough(d, 3)
    working = [
        f"|{inner}| {sym} {r} means the distance of x from {c} is {'at most' if strict_less and closed else 'less than' if strict_less else 'at least' if closed else 'more than'} {r}.",
        f"So x is {'between' if strict_less else 'outside'} {c - r} and {c + r}: "
        + (
            f"{c - r} {'≤' if closed else '<'} x {'≤' if closed else '<'} {c + r}."
            if strict_less
            else f"x {'≤' if closed else '<'} {c - r} or x {'≥' if closed else '>'} {c + r}."
        ),
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


def _hard_compound_word(rng):
    """Two linear inequalities joined by 'or' or 'and'."""
    use_or = rng.random() < 0.5
    for _ in attempts(500, "two inequalities with integer bounds"):
        a = rng.randint(-7, 2)
        b = a + rng.randint(2, 8)
        k1, k2 = rng.randint(2, 4), rng.randint(2, 4)
        d1, d2 = rng.randint(-5, 5), rng.randint(-5, 5)
        if b <= 8:
            break
    if use_or:
        # x < a  or  x >= b
        first = (_lin(k1, d1, "<", k1 * a + d1), ray_left(a, False))
        second = (_lin(k2, d2, "≥", k2 * b + d2), ray_right(b, True))
        pieces = [first[1], second[1]]
        joiner = "or"
    else:
        # x > a  and  x <= b
        first = (_lin(k1, d1, ">", k1 * a + d1), ray_right(a, False))
        second = (_lin(k2, d2, "≤", k2 * b + d2), ray_left(b, True))
        pieces = [P(a, False, b, True)]
        joiner = "and"
    text = f"Solve {first[0]} {joiner} {second[0]}. Show the solution on a number line."
    for x in range(-10, 11):
        x = Fraction(x)
        r1, r2 = _holds([first[1]], x), _holds([second[1]], x)
        check(_holds(pieces, x) == (r1 or r2 if use_or else r1 and r2), "the combined solution is wrong")
    fields = _fields(pieces)
    if use_or:
        cands = [
            ([P(a, False, b, True)], "w9-or-drawn-as-and"),
            ([first[1]], "w9-or-drawn-as-and"),
            ([second[1]], "w9-or-drawn-as-and"),
            ([ray_left(a, True), ray_right(b, False)], "w9-circle-open-closed-swapped"),
            ([ray_right(a, False), ray_left(b, True)], "w9-ray-wrong-direction"),
        ]
    else:
        cands = [
            ([ray_right(a, False), ray_left(b, True)], "w9-and-drawn-as-or"),
            ([first[1]], "w9-and-drawn-as-or"),
            ([second[1]], "w9-and-drawn-as-or"),
            ([P(a, True, b, False)], "w9-circle-open-closed-swapped"),
            ([P(-b, True, -a, False)], "w9-bound-sign-ignored"),
        ]
    d = _dist(fields, cands)
    enough(d, 3)
    working = [
        f"Solve each part. {first[0]} gives x {'<' if use_or else '>'} {a}. {second[0]} gives x {'≥' if use_or else '≤'} {b}.",
        f"'{joiner.capitalize()}' means "
        + (
            "x can satisfy either part, so shade both rays."
            if use_or
            else "x must satisfy both parts, so shade only where the two rays overlap: the segment between the bounds."
        ),
    ]
    return dict(text=text, figure=_figure(), working=working, distractors=d, **fields)


_HARD = (_hard_negative, _hard_absolute, _hard_compound_word)


@generator(
    TOPIC,
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w9_numberline_inequality(rng, tier="easy"):
    """Show an inequality's solution on a number line; the tiers are described in the module docstring."""
    pool = {"easy": (_easy,), "medium": (_medium,), "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# ---------------------------------------------------------------------------------------------- exam


@generator(EXAM_TOPIC, level=10, presets={"exam": dict(tier="exam")})
def exd_x_w9_numberline_inequality(rng, tier="exam"):
    """Exam chain: solve; show it; show it with a second condition; count the whole numbers that satisfy both."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam)


def _exam(rng):
    for _ in attempts(500, "a solution, a second bound and a count that fit"):
        a = rng.randint(-2, 6)  # x < a after solving
        k = rng.randint(2, 4)
        d = rng.randint(-6, 6)
        lower = a - rng.randint(3, 8)
        lower_closed = rng.random() < 0.5
        if lower >= -9 and d != 0:
            break
    # d - k*x > d - k*a   <=>   x < a   (dividing by -k reverses the sign)
    rhs = d - k * a
    ineq = f"{d} - {k}x > {rhs}"
    solution = [ray_left(a, False)]
    for x in range(-10, 11):
        check(((d - k * x) > rhs) == _holds(solution, Fraction(x)), "the solution set is wrong")
    combined = [P(lower, lower_closed, a, False)]
    count = sum(1 for x in range(-10, 11) if _holds(combined, Fraction(x)))
    check(count == (a - 1) - (lower if lower_closed else lower + 1) + 1, "the count disagrees with the formula")
    low_op = "≥" if lower_closed else ">"

    f_a_text = f"x < {a}"
    fa = dict(answer=f_a_text, answer_text=f_a_text, answer_latex="plain:" + f_a_text)
    fb, fc = _fields(solution), _fields(combined)
    da = dist(
        "inequality",
        fa["answer"],
        fa["answer_text"],
        [
            (f"x > {a}", f"x > {a}", "w9-ineq-sign-not-reversed"),
            (f"x < {-a}" if a else None, f"x < {-a}" if a else None, "w9-ineq-negative-bound-sign"),
            (f"x > {-a}" if a else None, f"x > {-a}" if a else None, "w9-ineq-negative-bound-sign"),
            (
                f"x < {Fraction(rhs - d, k)}" if (rhs - d) % k == 0 and Fraction(rhs - d, k) != a else None,
                f"x < {Fraction(rhs - d, k)}" if (rhs - d) % k == 0 else None,
                "w9-ineq-constant-sign",
            ),
            (
                f"x < {rhs - d}" if rhs - d != a else None,
                f"x < {rhs - d}" if rhs - d != a else None,
                "w9-ineq-forgets-to-divide",
            ),
        ],
    )
    db = _dist(fb, _simple_wrong("<", a))
    dc = _dist(
        fc,
        [
            ([ray_left(a, False)], "w9-compound-one-ray"),
            ([ray_right(lower, lower_closed)], "w9-compound-one-ray"),
            ([P(lower, not lower_closed, a, False)], "w9-compound-wrong-end-open"),
            ([P(lower, lower_closed, a, True)], "w9-circle-open-closed-swapped"),
            ([ray_left(lower, not lower_closed), ray_right(a, True)], "w9-compound-outside"),
        ],
    )
    dd = num_dist(
        count,
        [
            (count + 1, "w9-integer-count-includes-open-bound"),
            (count - 1, "w9-integer-count-misses-closed-bound"),
            (a - lower, "w9-integer-count-fencepost"),
            (a - lower - 1 if lower_closed else a - lower + 1, "w9-integer-count-fencepost"),
        ],
    )
    for dl in (da, db, dc, dd):
        enough(dl)
    stem = f"Solve the inequality {ineq}."
    return dict(
        text=stem,
        figure=_figure(),
        parts=[
            part(
                "Write the solution as an inequality.",
                "inequality",
                fa,
                2,
                [f"Subtract {d} from both sides: -{k}x > {rhs - d}. Divide by -{k} and reverse the sign: x < {a}."],
                da,
            ),
            part(
                "Show the solution on the number line.",
                "number_line_set",
                fb,
                1,
                [f"An open circle at {a}, shaded to the left."],
                db,
            ),
            part(
                f"The solution must also satisfy x {low_op} {lower}. Show the combined solution on the number line.",
                "number_line_set",
                fc,
                2,
                [f"Both conditions hold between {lower} and {a}: {lower} {'≤' if lower_closed else '<'} x < {a}."],
                dc,
            ),
            part(
                "How many whole numbers satisfy the combined solution?",
                "number",
                num_fields(count),
                1,
                [f"The whole numbers from {lower if lower_closed else lower + 1} to {a - 1} are {count} numbers."],
                dd,
                sensible_range=(0, None),
            ),
        ],
    )
