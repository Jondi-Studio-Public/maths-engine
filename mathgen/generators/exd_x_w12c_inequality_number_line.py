"""Exam chain for inequalities on a number line: read a shaded interval, solve a second inequality, count integers.

A shaded interval ``a < x <= b`` (open or closed ends) is drawn on a number line. A second condition ``px + q op r``
cuts it down at one end. Set the way a Year 10 exam sets it:

(a) the inequality shown on the number line                       inequality, 2 marks
(b) the solution of the second inequality                         inequality, 2 marks
(c) how many integers satisfy both                                number,     1 mark
(d) the sum of those integers                                     number,     1 mark, follow-through (c)

## The algebra (checked on every question)

The second inequality is solved by dividing by p (reversing the sign when p is negative); the solved form is also
tested against the original statement at points either side of its bound. The integers satisfying both conditions
are found by brute force over -15..15 from the two original statements, never from the solved bounds, and their
count and sum must equal the keys. Because the second condition always cuts one end of the shaded interval, the
integers form a run that keeps its other end from the number line: (d) follows through from the learner's count with
that fixed end (``n*L + n(n-1)/2`` when the run keeps its lowest integer, ``n*U - n(n-1)/2`` when it keeps its highest).

## Distractors

* (a): an open end read as closed, a closed end read as open, both flipped, an end read one tick out.
* (b): the sign not reversed for a negative p, reversed for a positive p, strict and inclusive swapped, the constant
  moved with the wrong sign, the division left out.
* (c): the count for the number line alone, one too many, one too few.
* (d): an excluded neighbouring integer added, the end integer left out, count times (first + last) without halving,
  the sum for the number line alone.
"""

from __future__ import annotations

from fractions import Fraction as F

from mathgen import figures
from mathgen.core import attempts, check, generator, join_terms, phrase
from mathgen.generators._exd_x_w12c_kit import redraw, inequality_part, int_part

_ALT = "a number line"
_FLIP = {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}
_TOGGLE = {"<": "≤", "≤": "<", ">": "≥", "≥": ">"}
_HOLDS = {
    "<": lambda l, r: l < r,
    "≤": lambda l, r: l <= r,
    ">": lambda l, r: l > r,
    "≥": lambda l, r: l >= r,
}

_OPEN = (
    "The number line shows the solution set of an inequality in x.",
    "The shaded part of the number line is the solution set of an inequality in x.",
    "An inequality in x has the solution set shown on the number line.",
)


def _num(v: F) -> str | None:
    """A bound as a whole number or a short decimal ("3.5"), else None."""
    if v.denominator == 1:
        return str(v.numerator)
    if 100 % v.denominator:
        return None
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def _interval_text(a, b, open_s, open_e) -> str:
    return f"{a} {'<' if open_s else '≤'} x {'<' if open_e else '≤'} {b}"


def _integers(a, b, open_s, open_e):
    """Integers in the shaded interval, from the interval's own statement."""
    return [i for i in range(-15, 16) if (i > a if open_s else i >= a) and (i < b if open_e else i <= b)]


def _build(rng):
    """Write the inequality shown, solve a second one, then count and sum the integers that satisfy both."""
    for _ in attempts(5000, "a number line and a second inequality that cuts it to three to nine integers"):
        a = rng.randint(-8, 1)
        b = rng.randint(a + 5, 8)
        open_s, open_e = rng.random() < 0.5, rng.random() < 0.5
        upper = rng.random() < 0.5  # the second inequality bounds the upper end (x < u) or the lower end (x > u)
        p = rng.choice((-4, -3, -2, 2, 3, 4))
        q = rng.choice([v for v in range(-9, 10) if v != 0])
        u = F(rng.randint(a + 2, b - 2))
        if abs(p) == 2 and rng.random() < 0.3:
            u += F(1, 2)
        sol_op = rng.choice(("<", "≤") if upper else (">", "≥"))
        op = sol_op if p > 0 else _FLIP[sol_op]
        r = p * u + q
        if r.denominator != 1:
            continue
        r = int(r)
        shown = _integers(a, b, open_s, open_e)
        both = [i for i in shown if _HOLDS[op](p * i + q, r)]  # from the ORIGINAL statement, not the solved bound
        if not 3 <= len(both) <= 9 or len(both) >= len(shown) or sum(both) == 0:
            continue
        break
    n, total = len(both), sum(both)
    check(both == list(range(both[0], both[-1] + 1)), "the integers are not a run")

    # Second route: the solved bound agrees with the original statement just either side of it.
    for probe in (u - 1, u - F(1, 2), u, u + F(1, 2), u + 1):
        check(
            _HOLDS[op](p * probe + q, r) == _HOLDS[sol_op](probe, u),
            "the solved inequality disagrees with the original",
        )

    expr = f"{join_terms([(p, 'x'), (q, '')])} {op} {r}"
    key_a = _interval_text(a, b, open_s, open_e)
    solved = f"x {sol_op} {_num(u)}"
    fig = figures.number_line(
        range_=(-10, 10),
        alt=_ALT,
        tick=1,
        intervals=[{"from": a, "to": b, "open_start": open_s, "open_end": open_e}],
    )
    figures.assert_no_leak(fig, [key_a])

    cands_a = [
        (_interval_text(a, b, not open_s, open_e), "strict-nonstrict-confused"),
        (_interval_text(a, b, open_s, not open_e), "strict-nonstrict-confused"),
        (_interval_text(a, b, not open_s, not open_e), "strict-nonstrict-confused"),
        (_interval_text(a, b + 1, open_s, open_e), "reads-one-gridline-out"),
        (_interval_text(a - 1, b, open_s, open_e), "reads-one-gridline-out"),
    ]
    rng.shuffle(cands_a)
    part_a = inequality_part(
        "Write down the inequality shown on the number line.",
        key_a,
        2,
        [
            f"The shading runs from {a} to {b}.",
            f"The end at {a} is {'open, so x > ' if open_s else 'closed, so x ≥ '}{a}; "
            f"the end at {b} is {'open, so x < ' if open_e else 'closed, so x ≤ '}{b}.",
            f"The inequality is {key_a}.",
        ],
        cands_a,
    )

    def bound(value):
        return _num(F(value))

    wrong_bounds = {
        "sign": bound((r + q) / F(p)),
        "nodiv": bound(r - q),
    }
    cands_b = [
        (
            f"x {_FLIP[sol_op] if p > 0 else op} {_num(u)}",
            "ineq-reversed-without-need" if p > 0 else "ineq-direction-not-reversed",
        ),
        (f"x {_TOGGLE[sol_op]} {_num(u)}", "strict-nonstrict-confused"),
    ]
    if wrong_bounds["sign"]:
        cands_b.append((f"x {sol_op} {wrong_bounds['sign']}", "sign-error-moving-term"))
    if wrong_bounds["nodiv"]:
        cands_b.append((f"x {sol_op} {wrong_bounds['nodiv']}", "forgot-to-divide-by-coefficient"))
    part_b = inequality_part(
        f"The value of x must also satisfy {expr}. Solve this inequality. "
        "Give your answer as an inequality in x, writing any bound that is not a whole number as a decimal.",
        solved,
        2,
        [
            f"Subtract {q} from both sides, or add {-q}: {p}x {op} {r - q}.",
            f"Divide both sides by {p}{', reversing the inequality sign because the divisor is negative' if p < 0 else ''}: {solved}.",
        ],
        cands_b,
    )

    lo, hi = both[0], both[-1]
    count_cands = [
        (len(shown), "compound-bound-omitted"),
        (n + 1, "integer-solutions-endpoint-miscount"),
        (n - 1, "integer-solutions-endpoint-miscount"),
    ]
    part_c = int_part(
        f"How many integers satisfy both the inequality shown on the number line and {expr}?",
        n,
        1,
        [
            f"Both conditions together give x from {lo} to {hi} as whole numbers.",
            f"The integers are {', '.join(str(i) for i in both)}: {n} of them.",
        ],
        count_cands,
        lo=0,
        hi=30,
    )
    if upper:
        fixed, follow = shown[0], f"part_c*({shown[0]}) + part_c*(part_c - 1)/2"
        extra_neighbour = hi + 1
    else:
        fixed, follow = shown[-1], f"part_c*({shown[-1]}) - part_c*(part_c - 1)/2"
        extra_neighbour = lo - 1
    sum_cands = [
        (total + extra_neighbour, "integer-solutions-endpoint-miscount"),
        (total - (hi if upper else lo), "integer-solutions-endpoint-miscount"),
        (n * (lo + hi), "sum-of-run-not-halved"),
        (sum(shown), "compound-bound-omitted"),
    ]
    part_d = int_part(
        "Find the sum of all the integers that satisfy both inequalities.",
        total,
        1,
        [f"{' + '.join(f'({i})' if i < 0 else str(i) for i in both)} = {total}."],
        sum_cands,
        lo=-200,
        hi=200,
        follow_through=follow,
    )
    return dict(text=phrase(rng, _OPEN), figure=fig, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_w12c_inequality_number_line", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w12c_inequality_number_line(rng, tier: str = "exam"):
    """Write the inequality shown, solve a second one, then count and sum the integers that satisfy both."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return redraw(rng, _build)
