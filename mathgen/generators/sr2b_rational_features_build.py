"""Generates `sr2b_rational_features` questions (easy, medium, hard) and the exam chain `sr2b_exam_rational_features`.

Skill ``rational-function-features`` (``functions-sequences``, answer ``pair``): the asymptotes, intercepts and holes of
a rational function that is a linear expression over a linear one, or a quadratic over a linear one. Sketching is not
asked; the learner reads the features off the algebra. Every answer is a labelled pair, so the order it is typed in
does not matter.

* easy: ``y = (ax + b)/(x + d)``. The vertical asymptote is where the denominator is zero and the horizontal
  asymptote is the ratio of the leading coefficients (here ``a``). The answer is ``x = -d, y = a``.
* medium: ``y = (ax + b)/(cx + d)`` with ``c`` of 2, 3, 4 or negative, so the asymptotes are fractions and the
  leading-coefficient ratio is not read off. Either the asymptotes ``x = -d/c, y = a/c``, or the intercepts
  ``x = -b/a, y = b/d``.
* hard: a quadratic over a linear expression, ``y = (px^2 + qx + r)/(sx + t)``. Either the oblique asymptote
  ``y = mx + k`` (long division, ``m`` and ``k`` asked), or a numerator that shares the factor ``x - h`` with the
  denominator, so there is a hole at ``(h, y)`` and no vertical asymptote there.
* exam (``sr2b_exam_rational_features``): for ``f(x) = (x^2 + bx + c)/(x - h)``, (a) the vertical asymptote, (b) the
  constant ``k`` in the oblique asymptote ``y = x + k``, (c) the height where the two asymptotes meet (follows through
  from both), (d) the y-intercept.

Each answer is built first (the asymptote, quotient or hole is drawn), then confirmed a second way by evaluating the
function exactly with `Fraction` arithmetic far out on the axis and at the excluded value.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, fmt_signed, generator, join_terms, phrase
from ._sr2b_kit import Pair, int_dist, pair_candidates, pick, retry_short, sup

_FORM = " Give your answer as {l1} = ..., {l2} = ...."


def _lin(a, b, constant_first=False) -> str:
    return join_terms([(b, ""), (a, "x")] if constant_first else [(a, "x"), (b, "")])


def _quad(p, q, r) -> str:
    return join_terms([(p, f"x{sup(2)}"), (q, "x"), (r, "")])


def _eval(num, den, x) -> Fraction:
    return Fraction(num(x)) / Fraction(den(x))


def _check_mobius(a, b, c, d, v_asym: Fraction, h_asym: Fraction) -> None:
    """Second route to the asymptotes: the denominator vanishes (and the numerator does not) at the vertical
    asymptote, and the function is within 1/1000 of the horizontal one far out on both sides."""
    num, den = (lambda x: a * x + b), (lambda x: c * x + d)
    check(den(v_asym) == 0 and num(v_asym) != 0, "the vertical asymptote differs from the denominator's zero")
    for far in (10**7, -(10**7)):
        check(abs(_eval(num, den, far) - h_asym) < Fraction(1, 1000), "the horizontal asymptote differs from the limit")


def _draw_mobius(rng, c_choices, a_choices):
    for _ in attempts(500, "a Möbius function that does not simplify"):
        a, c = rng.choice(a_choices), rng.choice(c_choices)
        b, d = rng.randint(-9, 9), rng.randint(-8, 8)
        if 0 in (a, b, d) or a * d == b * c:
            continue
        return a, b, c, d
    raise AssertionError  # pragma: no cover


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    a, b, c, d = _draw_mobius(rng, (1,), (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5))
    cf = rng.random() < 0.3
    f = f"({_lin(a, b, cf)})/({_lin(1, d)})"
    v, h = Fraction(-d), Fraction(a)
    _check_mobius(a, b, c, d, v, h)
    ans = Pair("x", v, "y", h)
    wrong = [
        ("sr2b-asymptotes-swapped", h, v),
        ("sr2b-vertical-asymptote-sign", d, h),
        ("sr2b-horizontal-asymptote-constant-term", v, b),
        ("sr2b-horizontal-asymptote-negated", v, -h),
    ]
    stem = phrase(
        rng,
        (
            "Find the equations of the vertical and horizontal asymptotes of the graph of y = {f}.",
            "The graph of y = {f} has one vertical and one horizontal asymptote. State their equations.",
        ),
        f=f,
    ) + _FORM.format(l1="x", l2="y")
    return dict(
        text=stem,
        working=[
            f"The denominator is zero when x = {-d}, and the numerator is not zero there, so x = {-d} is the vertical asymptote.",
            f"As x grows large the function approaches {fmt_signed(a, 'x')}/x = {a}, so y = {a} is the horizontal asymptote.",
        ],
        distractors=pick(ans.text, pair_candidates(ans, wrong)),
        **ans.fields(),
    )


# -- medium -------------------------------------------------------------------------------------------------------


def _medium(rng):
    a, b, c, d = _draw_mobius(rng, (-4, -3, -2, 2, 3, 4), (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6))
    cf = rng.random() < 0.3
    f = f"({_lin(a, b, cf)})/({_lin(c, d, cf)})"
    if rng.random() < 0.6:
        v, h = Fraction(-d, c), Fraction(a, c)
        _check_mobius(a, b, c, d, v, h)
        ans = Pair("x", v, "y", h)
        wrong = [
            ("sr2b-coefficient-of-x-ignored", -d, h),
            ("sr2b-coefficient-of-x-ignored", v, a),
            ("sr2b-asymptotes-swapped", h, v),
            ("sr2b-vertical-asymptote-sign", Fraction(d, c), h),
            ("sr2b-ratio-of-coefficients-inverted", v, Fraction(c, a)),
        ]
        stem = f"Find the equations of the vertical and horizontal asymptotes of the graph of y = {f}." + _FORM.format(
            l1="x", l2="y"
        )
        working = [
            f"The denominator is zero when {_lin(c, d)} = 0, so x = {ans.values[0]} is the vertical asymptote.",
            f"For large x the function approaches the ratio of the leading coefficients, {a}/{c}, so y = {ans.values[1]} is the horizontal asymptote.",
        ]
    else:
        xi, yi = Fraction(-b, a), Fraction(b, d)
        num, den = (lambda x: a * x + b), (lambda x: c * x + d)
        check(_eval(num, den, xi) == 0 and _eval(num, den, 0) == yi, "intercepts differ from direct evaluation")
        ans = Pair("x", xi, "y", yi)
        wrong = [
            ("sr2b-intercepts-swapped", yi, xi),
            ("sr2b-x-intercept-sign", Fraction(b, a), yi),
            ("sr2b-intercept-from-leading-coefficients", xi, Fraction(a, c)),
            ("sr2b-x-intercept-from-denominator", Fraction(-d, c), yi),
            ("sr2b-y-intercept-from-numerator", xi, b),
        ]
        stem = (
            f"Find the x-intercept and the y-intercept of the graph of y = {f}. "
            "Give your answer as x = ... for the x-intercept and y = ... for the y-intercept."
        )
        working = [
            f"At the x-intercept the numerator is zero: {_lin(a, b)} = 0, so x = {xi}.",
            f"At the y-intercept x = 0, so y = {b}/{d} = {yi}.",
        ]
    return dict(text=stem, working=working, distractors=pick(ans.text, pair_candidates(ans, wrong)), **ans.fields())


# -- hard ---------------------------------------------------------------------------------------------------------


def _check_oblique(p, q, r, s, t, m: Fraction, k: Fraction) -> None:
    """The function minus the claimed line is tiny far out on both sides (the remainder over the denominator)."""
    num = lambda x: p * x * x + q * x + r  # noqa: E731
    den = lambda x: s * x + t  # noqa: E731
    for far in (10**6, -(10**6)):
        check(abs(_eval(num, den, far) - (m * far + k)) < Fraction(1, 100), "the oblique asymptote differs")


def _hard_oblique(rng):
    for _ in attempts(500, "a quadratic over a linear expression with an oblique asymptote"):
        s = rng.choice((1, 1, 2))
        t = rng.choice([v for v in range(-6, 7) if v != 0 and (s == 1 or v % 2 == 0)])
        m = Fraction(rng.choice((-3, -2, -1, 1, 2, 3)) if s == 1 else rng.choice((-3, -1, 1, 3)), 1 if s == 1 else 2)
        k = rng.randint(-5, 5)
        rem = rng.choice([v for v in range(-9, 10) if v != 0])
        p, q, r = s * m, s * k + t * m, t * k + rem
        if any(Fraction(v).denominator != 1 for v in (p, q, r)):
            continue
        p, q, r = int(p), int(q), int(r)
        if k == 0 or q == 0 or r == 0:
            continue
        break
    _check_oblique(p, q, r, s, t, m, Fraction(k))
    ans = Pair("m", m, "k", k)
    wrong = [
        ("sr2b-leading-coefficient-not-divided", p, k),
        ("sr2b-constant-term-read-as-intercept", m, q),
        ("sr2b-division-sign-slip", m, Fraction(q + m * t, s)),
        ("sr2b-remainder-for-intercept", m, rem),
        ("sr2b-constants-divided", m, Fraction(r, t)),
    ]
    f = f"({_quad(p, q, r)})/({_lin(s, t)})"
    stem = f"The graph of y = {f} has an oblique asymptote with equation y = mx + k. Find m and k." + _FORM.format(
        l1="m", l2="k"
    )
    return dict(
        text=stem,
        working=[
            f"Divide {_quad(p, q, r)} by {_lin(s, t)}: the quotient is {_lin(m, k) if m.denominator == 1 else f'({m})x {"+" if k >= 0 else "-"} {abs(k)}'} with remainder {rem}.",
            "The remainder over the denominator tends to zero for large x, so the graph approaches y = quotient.",
        ],
        distractors=pick(ans.text, pair_candidates(ans, wrong)),
        **ans.fields(),
    )


def _hard_hole(rng):
    for _ in attempts(500, "a numerator sharing the denominator's factor"):
        h = rng.choice([v for v in range(-5, 6) if v != 0])
        p = rng.choice((-3, -2, -1, 1, 2, 3))
        w = rng.randint(-6, 6)
        y = p * h + w
        q, r = w - p * h, -h * w
        if w == 0 or q == 0 or y == 0:
            continue
        break
    num = lambda x: p * x * x + q * x + r  # noqa: E731
    den = lambda x: x - h  # noqa: E731
    for xv in (-7, -2, 3, 11):
        check(xv == h or _eval(num, den, xv) == p * xv + w, "the numerator does not divide out")
    check(num(h) == 0 and y == p * h + w, "the numerator is not zero at the hole")
    ans = Pair("x", h, "y", y)
    wrong = [
        ("sr2b-hole-height-from-constant", h, w),
        ("sr2b-hole-x-sign", -h, p * (-h) + w),
        ("sr2b-hole-height-from-coefficient", h, q),
        ("sr2b-hole-height-quotient-slope-only", h, p * h),
    ]
    f = f"({_quad(p, q, r)})/({_lin(1, -h)})"
    stem = (
        f"The graph of y = {f} has a hole, a single point where the function is undefined but the graph "
        "has no vertical asymptote. Find the coordinates of the hole." + _FORM.format(l1="x", l2="y")
    )
    return dict(
        text=stem,
        working=[
            f"The numerator factorises as (x {'-' if h > 0 else '+'} {abs(h)})({_lin(p, w)}), so the common factor (x {'-' if h > 0 else '+'} {abs(h)}) cancels.",
            f"The graph is y = {_lin(p, w)} with a hole at x = {h}, where y = {p}({h}) + ({w}) = {y}.",
        ],
        distractors=pick(ans.text, pair_candidates(ans, wrong)),
        **ans.fields(),
    )


@generator(
    "sr2b_rational_features",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def sr2b_rational_features(rng, tier="easy"):
    """One feature of a rational function; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_oblique, _hard_oblique, _hard_hole])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _exam_candidates(h, b, c, e, k):
    """The tagged wrong values for parts a to d of the exam chain."""
    part_a = [
        int_dist(-h, "sr2b-vertical-asymptote-sign"),
        int_dist(c, "sr2b-vertical-asymptote-from-numerator"),
        int_dist(b, "sr2b-vertical-asymptote-from-numerator"),
        int_dist(-c, "sr2b-vertical-asymptote-from-numerator"),
        int_dist(-b, "sr2b-vertical-asymptote-from-numerator"),
    ]
    part_b = [
        int_dist(b, "sr2b-division-sign-slip"),
        int_dist(b - h, "sr2b-division-sign-slip"),
        int_dist(c + h * h + b * h, "sr2b-remainder-for-intercept"),
        int_dist(-k, "sr2b-quotient-constant-negated"),
    ]
    part_c = [
        int_dist(k, "sr2b-asymptote-meeting-height-from-k"),
        int_dist(h * k, "sr2b-asymptote-meeting-height-product"),
        int_dist(k - h, "sr2b-asymptote-meeting-height-sign"),
        int_dist(h, "sr2b-asymptote-meeting-height-from-x"),
    ]
    part_d = [
        int_dist(-e, "sr2b-y-intercept-sign"),
        int_dist(c, "sr2b-y-intercept-from-numerator"),
        int_dist(h * e, "sr2b-y-intercept-from-numerator"),
        int_dist(c - h, "sr2b-y-intercept-from-numerator"),
    ]
    return part_a, part_b, part_c, part_d


@generator("sr2b_exam_rational_features", level=12, presets={"exam": dict(tier="exam")})
@retry_short
def sr2b_exam_rational_features(rng, tier="exam"):
    """Exam chain: vertical asymptote, oblique asymptote constant, where the asymptotes meet, y-intercept."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "an exam rational function"):
        h = rng.choice([v for v in range(-4, 5) if v != 0])
        e = rng.choice([v for v in range(-6, 7) if v not in (0,)])
        b = rng.randint(-6, 6)
        c = -h * e
        if h * h + b * h + c == 0 or b == 0 or abs(b + h) == 0:
            continue
        k = b + h
        answers = (h, k, h + k, e)
        pools = _exam_candidates(h, b, c, e, k)
        if all(len({d.text for d in pool if d.text != str(ans)}) >= 3 for pool, ans in zip(pools, answers)):
            break
    num = lambda x: x * x + b * x + c  # noqa: E731
    den = lambda x: x - h  # noqa: E731
    check(num(h) != 0, "the function simplifies")
    for far in (10**6, -(10**6)):
        check(abs(_eval(num, den, far) - (far + k)) < Fraction(1, 100), "oblique asymptote differs")
    check(_eval(num, den, 0) == e, "y-intercept differs")
    meet = h + k
    f = f"({_quad(1, b, c)})/({_lin(1, -h)})"
    part_a, part_b, part_c, part_d = _exam_candidates(h, b, c, e, k)
    return dict(
        text=f"Consider the function f(x) = {f}.",
        parts=[
            dict(
                prompt="Find the value of x at which the graph of y = f(x) has a vertical asymptote.",
                answer_kind="number",
                answer=h,
                answer_text=str(h),
                answer_latex=str(h),
                marks=1,
                sensible_range=(-100, 100),
                working=[
                    f"The denominator x {'-' if h > 0 else '+'} {abs(h)} is zero at x = {h}, and the numerator is not."
                ],
                distractors=pick(str(h), part_a),
            ),
            dict(
                prompt="The graph also has an oblique asymptote with equation y = x + k. Find the value of k.",
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                answer_latex=str(k),
                marks=2,
                sensible_range=(-100, 100),
                working=[
                    f"Dividing {_quad(1, b, c)} by {_lin(1, -h)} gives a quotient of x + {k} and remainder {num(h)}.".replace(
                        "+ -", "- "
                    ),
                ],
                distractors=pick(str(k), part_b),
            ),
            dict(
                prompt="The vertical asymptote and the oblique asymptote cross at a point. Find the y-coordinate of this point.",
                answer_kind="number",
                answer=meet,
                answer_text=str(meet),
                answer_latex=str(meet),
                marks=1,
                sensible_range=(-100, 100),
                follow_through="part_a+part_b",
                working=[
                    f"The point has x = {h} and lies on y = x {'+' if k >= 0 else '-'} {abs(k)}, so y = {h} {'+' if k >= 0 else '-'} {abs(k)} = {meet}."
                ],
                distractors=pick(str(meet), part_c),
            ),
            dict(
                prompt="Find the y-intercept of the graph of y = f(x).",
                answer_kind="number",
                answer=e,
                answer_text=str(e),
                answer_latex=str(e),
                marks=2,
                sensible_range=(-100, 100),
                working=[f"At x = 0, f(0) = {c}/({-h}) = {e}."],
                distractors=pick(str(e), part_d),
            ),
        ],
    )
