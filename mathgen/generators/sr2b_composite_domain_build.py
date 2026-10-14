"""Generates `sr2b_composite_domain` questions (easy, medium, hard) and the exam chain `sr2b_exam_composite_domain`.

Skill ``composite-function-domain`` (``functions-sequences``, answer ``inequality``): the largest domain of a composite
function ``g(f(x))`` as an inequality (or a union of them), where the functions involve a square root, a reciprocal or
a logarithm. The existing recognition skill is multiple choice; here the learner writes the domain.

* easy: ``f`` linear and ``g`` one of the square root, the natural logarithm or the reciprocal, so the single condition
  on ``f(x)`` is solved for x (a linear inequality, with the direction reversed when the coefficient is negative).
* medium: ``f`` quadratic, or ``f`` a root or reciprocal, so the condition on ``f(x)`` is a quadratic inequality, a
  union of two rays, a missing pair of points, or a strict end-point where ``f(x)`` is zero.
* hard: both stages restrict x: ``f`` has its own domain and ``f(x)`` must also lie in ``g``'s domain, giving a closed
  interval, a ray with a hole, a bound that depends on a sign, or a bound from a logarithm.
* exam (``sr2b_exam_composite_domain``): for ``f(x) = sqrt(x - p)`` and ``g(x) = 1/(x - q)``: (a) the least x for which
  f is defined, (b) the value of x for which g(f(x)) is undefined (follows through from (a)), (c) the domain of g(f(x)).

Each answer is built by solving the conditions as sets, then confirmed by testing the composite pointwise: a grid of
rational x values, and both sides of every end-point, is checked for being defined against the answer set.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..answers import Distractor
from ..core import check, generator, phrase, rational_text
from ..grading.inequality import read_solution_set, set_text
from ._sr2b_kit import int_dist, pick, retry_short

R = sp.S.Reals
OO = sp.oo
_I = sp.Interval


def _r(x) -> sp.Rational:
    return sp.Rational(x.numerator, x.denominator) if isinstance(x, Fraction) else sp.Rational(x)


def _text(s) -> str:
    """The set as inequalities; a line with one point removed is written x ≠ r."""
    if isinstance(s, sp.Union) and len(s.args) == 2:
        left, right = sorted(s.args, key=lambda i: i.start if i.start != -OO else -(10**9))
        if left.start == -OO and right.end == OO and left.end == right.start and left.right_open and right.left_open:
            return f"x ≠ {rational_text(left.end)}"
    return set_text(s, "x")


def _dist(s, tag: str) -> Distractor:
    t = _text(s)
    return Distractor(t, tag, t)


def _lin(p: int, q: int) -> str:
    if p == 1:
        head = "x"
    elif p == -1:
        head = "-x"
    else:
        head = f"{p}x"
    return head if q == 0 else f"{head} {'+' if q > 0 else '-'} {abs(q)}"


def _shift(a: int, root: bool = False) -> str:
    return f"x {'-' if a > 0 else '+'} {abs(a)}"


def _spans(ans: sp.Set) -> list:
    """The intervals of `ans` as plain tuples (low, low_open, high, high_open), with None for an infinite end."""
    parts = ans.args if isinstance(ans, sp.Union) else (ans,)
    out = []
    for part in parts:
        check(isinstance(part, sp.Interval), "the answer set is not a union of intervals")
        lo = None if part.start == -OO else Fraction(int(part.start.p), int(part.start.q))
        hi = None if part.end == OO else Fraction(int(part.end.p), int(part.end.q))
        out.append((lo, part.left_open, hi, part.right_open))
    return out


def _inside(spans: list, x: Fraction) -> bool:
    for lo, lo_open, hi, hi_open in spans:
        if lo is not None and (x < lo or (lo_open and x == lo)):
            continue
        if hi is not None and (x > hi or (hi_open and x == hi)):
            continue
        return True
    return False


def _verify(ans: sp.Set, defined) -> None:
    """Pointwise second route: `defined(x)` is true exactly on `ans` for a grid of rational x and around each
    end-point (the boundary values themselves included)."""
    pts = {Fraction(k, 4) for k in range(-80, 121)}
    for b in ans.boundary if hasattr(ans, "boundary") else []:
        if b.is_Rational:
            fb = Fraction(int(b.p), int(b.q))
            pts.update({fb, fb + Fraction(1, 64), fb - Fraction(1, 64)})
    spans = _spans(ans)
    for x in pts:
        check(defined(x) == _inside(spans, x), f"the answer set disagrees with the composite at x = {x}")


def _ok(fn) -> bool:
    try:
        fn()
    except (ValueError, ZeroDivisionError):
        return False
    return True


# -- the problem families ---------------------------------------------------------------------------------------


def _e_sqrt(rng):
    p, q = rng.choice((-3, -2, -1, 1, 2, 3, 4)), rng.choice([v for v in range(-9, 10) if v])
    r = sp.Rational(-q, p)
    ans = _I(r, OO) if p > 0 else _I(-OO, r)
    wrong = [
        (_I(r, OO, True, True) if p > 0 else _I(-OO, r, True, True), "sr2b-boundary-wrongly-excluded"),
        (_I(-OO, r) if p > 0 else _I(r, OO), "sr2b-inequality-direction-wrong"),
        (_I(sp.Rational(q, p), OO) if p > 0 else _I(-OO, sp.Rational(q, p)), "sr2b-boundary-sign-slip"),
        (_I(0, OO), "sr2b-outer-domain-used-for-x"),
    ]
    return dict(
        f=_lin(p, q),
        g="√x",
        ans=ans,
        wrong=wrong,
        defined=lambda x: p * x + q >= 0,
        working=[f"√ needs a non-negative input, so {_lin(p, q)} ≥ 0.", f"Solving: {_text(ans)}."],
    )


def _e_ln(rng):
    p, q = rng.choice((-3, -2, -1, 1, 2, 3, 4)), rng.choice([v for v in range(-9, 10) if v])
    r = sp.Rational(-q, p)
    ans = _I(r, OO, True, True) if p > 0 else _I(-OO, r, True, True)
    wrong = [
        (_I(r, OO) if p > 0 else _I(-OO, r), "sr2b-boundary-wrongly-included"),
        (_I(-OO, r, True, True) if p > 0 else _I(r, OO, True, True), "sr2b-inequality-direction-wrong"),
        (
            _I(sp.Rational(q, p), OO, True, True) if p > 0 else _I(-OO, sp.Rational(q, p), True, True),
            "sr2b-boundary-sign-slip",
        ),
        (_I(0, OO, True, True), "sr2b-outer-domain-used-for-x"),
    ]
    return dict(
        f=_lin(p, q),
        g="ln(x)",
        ans=ans,
        wrong=wrong,
        defined=lambda x: p * x + q > 0,
        working=[f"ln needs a positive input, so {_lin(p, q)} > 0.", f"Solving: {_text(ans)}."],
    )


def _e_recip(rng):
    p, q = rng.choice((-3, -2, 2, 3, 4)), rng.choice([v for v in range(-9, 10) if v])
    r = sp.Rational(-q, p)
    ans = sp.Union(_I(-OO, r, True, True), _I(r, OO, True, True))
    wrong = [
        (sp.Union(_I(-OO, -r, True, True), _I(-r, OO, True, True)), "sr2b-boundary-sign-slip"),
        (sp.Union(_I(-OO, 0, True, True), _I(0, OO, True, True)), "sr2b-outer-domain-used-for-x"),
        (R, "sr2b-no-restriction-found"),
        (_I(r, OO, True, True), "sr2b-one-side-only"),
    ]
    return dict(
        f=_lin(p, q),
        g="1/x",
        ans=ans,
        wrong=wrong,
        defined=lambda x: p * x + q != 0,
        working=[f"1/x is undefined when its input is 0, so {_lin(p, q)} ≠ 0.", f"Solving: {_text(ans)}."],
    )


def _m_sqrt_quad(rng):
    k = rng.choice((2, 3, 4, 5, 6))
    ans = sp.Union(_I(-OO, -k), _I(k, OO))
    wrong = [
        (_I(-k, k), "sr2b-quadratic-inequality-wrong-way"),
        (_I(k, OO), "sr2b-negative-branch-dropped"),
        (sp.Union(_I(-OO, -k, True, True), _I(k, OO, True, True)), "sr2b-boundary-wrongly-excluded"),
        (sp.Union(_I(-OO, -k * k), _I(k * k, OO)), "sr2b-square-root-not-taken"),
    ]
    return dict(
        f=f"x² - {k * k}",
        g="√x",
        ans=ans,
        wrong=wrong,
        defined=lambda x: x * x - k * k >= 0,
        working=[
            f"√ needs x² - {k * k} ≥ 0, so (x - {k})(x + {k}) ≥ 0.",
            f"This holds outside the roots: {_text(ans)}.",
        ],
    )


def _m_ln_quad(rng):
    k = rng.choice((2, 3, 4, 5, 6))
    ans = _I(-k, k, True, True)
    wrong = [
        (_I(-k, k), "sr2b-boundary-wrongly-included"),
        (sp.Union(_I(-OO, -k, True, True), _I(k, OO, True, True)), "sr2b-quadratic-inequality-wrong-way"),
        (_I(0, k, True, True), "sr2b-negative-branch-dropped"),
        (_I(-k * k, k * k, True, True), "sr2b-square-root-not-taken"),
    ]
    return dict(
        f=f"{k * k} - x²",
        g="ln(x)",
        ans=ans,
        wrong=wrong,
        defined=lambda x: k * k - x * x > 0,
        working=[f"ln needs {k * k} - x² > 0, so x² < {k * k}.", f"This holds between the roots: {_text(ans)}."],
    )


def _m_recip_root(rng):
    a = rng.choice([v for v in range(-6, 7) if v])
    ans = _I(a, OO, True, True)
    wrong = [
        (_I(a, OO), "sr2b-boundary-wrongly-included"),
        (_I(-OO, a), "sr2b-inequality-direction-wrong"),
        (_I(-a, OO, True, True), "sr2b-boundary-sign-slip"),
        (_I(0, OO, True, True), "sr2b-outer-domain-used-for-x"),
    ]
    return dict(
        f=f"√({_shift(a)})",
        g="1/x",
        ans=ans,
        wrong=wrong,
        defined=lambda x: x - a > 0,
        working=[f"f needs x ≥ {a}, and 1/f(x) also needs f(x) ≠ 0, which rules out x = {a}.", f"So {_text(ans)}."],
    )


def _m_root_recip(rng):
    a = rng.choice([v for v in range(-6, 7) if v])
    ans = _I(a, OO, True, True)
    wrong = [
        (_I(a, OO), "sr2b-boundary-wrongly-included"),
        (_I(-OO, a, True, True), "sr2b-inequality-direction-wrong"),
        (sp.Union(_I(-OO, a, True, True), _I(a, OO, True, True)), "sr2b-sign-condition-ignored"),
        (_I(-a, OO, True, True), "sr2b-boundary-sign-slip"),
    ]
    return dict(
        f=f"1/({_shift(a)})",
        g="√x",
        ans=ans,
        wrong=wrong,
        defined=lambda x: x != a and Fraction(1, 1) / (x - a) >= 0,
        working=[
            f"√ needs 1/({_shift(a)}) ≥ 0, and a reciprocal is never zero, so it must be positive.",
            f"{_shift(a)} > 0 gives {_text(ans)}.",
        ],
    )


def _m_recip_square(rng):
    k = rng.choice((2, 3, 4, 5))
    ans = sp.Union(_I(-OO, -k, True, True), _I(-k, k, True, True), _I(k, OO, True, True))
    wrong = [
        (sp.Union(_I(-OO, k, True, True), _I(k, OO, True, True)), "sr2b-one-root-only"),
        (sp.Union(_I(-OO, k * k, True, True), _I(k * k, OO, True, True)), "sr2b-square-root-not-taken"),
        (
            sp.Union(_I(-OO, -k * k, True, True), _I(-k * k, k * k, True, True), _I(k * k, OO, True, True)),
            "sr2b-square-root-not-taken",
        ),
        (R, "sr2b-no-restriction-found"),
    ]
    return dict(
        f="x²",
        g=f"1/(x - {k * k})",
        ans=ans,
        wrong=wrong,
        defined=lambda x: x * x != k * k,
        working=[
            f"g is undefined when its input is {k * k}, so f(x) = x² ≠ {k * k}.",
            f"That excludes x = {k} and x = -{k}: {_text(ans)}.",
        ],
    )


def _h_root_root(rng):
    a, c = rng.choice((1, 2, 3, 4, 5)), rng.choice((2, 3, 4))
    ans = _I(-a, c * c - a)
    wrong = [
        (_I(-a, OO), "sr2b-outer-condition-missed"),
        (_I(-OO, c), "sr2b-inner-domain-missed"),
        (_I(-a, c - a), "sr2b-inner-not-squared"),
        (_I(-a, c * c + a), "sr2b-boundary-sign-slip"),
        (_I(-a, c * c - a, False, True), "sr2b-boundary-wrongly-excluded"),
    ]
    return dict(
        f=f"√(x + {a})",
        g=f"√({c} - x)",
        ans=ans,
        wrong=wrong,
        defined=lambda x: x + a >= 0 and c - math.sqrt(x + a) >= 0,
        working=[
            f"f needs x ≥ -{a}. g needs {c} - f(x) ≥ 0, so √(x + {a}) ≤ {c}, so x + {a} ≤ {c * c}, so x ≤ {c * c - a}.",
            f"Both together: {_text(ans)}.",
        ],
    )


def _h_ln_root(rng):
    a = rng.choice([v for v in range(-4, 5) if v])
    b = rng.choice((-3, -2, -1, 1, 2, 3))
    if b > 0:
        ans = _I(a + b * b, OO, True, True)
        wrong = [
            (_I(a + b * b, OO), "sr2b-boundary-wrongly-included"),
            (_I(a + b, OO, True, True), "sr2b-inner-not-squared"),
            (_I(a, OO), "sr2b-outer-condition-missed"),
            (_I(a - b * b, OO, True, True), "sr2b-boundary-sign-slip"),
        ]
        why = f"√({_shift(a)}) > {b}, so {_shift(a)} > {b * b}"
    else:
        ans = _I(a, OO)
        wrong = [
            (_I(a, OO, True, True), "sr2b-boundary-wrongly-excluded"),
            (_I(a + b * b, OO, True, True), "sr2b-squared-negative-bound"),
            (_I(a + b * b, OO), "sr2b-squared-negative-bound"),
            (_I(-OO, a), "sr2b-inequality-direction-wrong"),
        ]
        why = f"√({_shift(a)}) > {b}, which is true whenever the root exists, since a root is never negative"
    f = f"√({_shift(a)}) {'-' if b > 0 else '+'} {abs(b)}"
    return dict(
        f=f,
        g="ln(x)",
        ans=ans,
        wrong=wrong,
        defined=lambda x: x - a >= 0 and math.sqrt(x - a) - b > 0,
        working=[f"f needs x ≥ {a}. ln needs f(x) > 0: {why}.", f"Both together: {_text(ans)}."],
    )


def _h_recip_root(rng):
    a = rng.choice([v for v in range(-4, 5) if v])
    c = rng.choice((-3, -2, -1, 1, 2, 3))
    if c > 0:
        hole = a + c * c
        ans = sp.Union(_I(a, hole, False, True), _I(hole, OO, True, True))
        wrong = [
            (_I(a, OO), "sr2b-excluded-value-missed"),
            (sp.Union(_I(a, c, False, True), _I(c, OO, True, True)), "sr2b-excluded-value-not-converted"),
            (sp.Union(_I(a, a + c, False, True), _I(a + c, OO, True, True)), "sr2b-inner-not-squared"),
            (sp.Union(_I(a, hole, True, True), _I(hole, OO, True, True)), "sr2b-boundary-wrongly-excluded"),
        ]
        why = f"√({_shift(a)}) = {c} when {_shift(a)} = {c * c}, i.e. x = {hole}"
    else:
        ans = _I(a, OO)
        wrong = [
            (sp.Union(_I(a, a + c * c, False, True), _I(a + c * c, OO, True, True)), "sr2b-squared-negative-bound"),
            (_I(a, OO, True, True), "sr2b-boundary-wrongly-excluded"),
            (sp.Union(_I(a, c, False, True), _I(c, OO, True, True)), "sr2b-excluded-value-not-converted"),
            (_I(-OO, a), "sr2b-inequality-direction-wrong"),
        ]
        why = f"√({_shift(a)}) = {c} is impossible, since a root is never negative, so nothing is excluded"
    return dict(
        f=f"√({_shift(a)})",
        g=f"1/(x - {c})" if c > 0 else f"1/(x + {-c})",
        ans=ans,
        wrong=wrong,
        defined=lambda x: x - a >= 0 and math.sqrt(x - a) != c,
        working=[f"f needs x ≥ {a}. g needs f(x) ≠ {c}: {why}.", f"Both together: {_text(ans)}."],
    )


def _h_root_ln(rng):
    a = rng.choice([v for v in range(-5, 6) if v])
    ans = _I(a + 1, OO)
    wrong = [
        (_I(a, OO, True, True), "sr2b-outer-condition-missed"),
        (_I(a + 1, OO, True, True), "sr2b-boundary-wrongly-excluded"),
        (_I(1, OO), "sr2b-shift-ignored"),
        (_I(a, OO), "sr2b-outer-condition-missed"),
    ]
    return dict(
        f=f"ln({_shift(a)})",
        g="√x",
        ans=ans,
        wrong=wrong,
        defined=lambda x: x - a > 0 and math.log(x - a) >= 0,
        working=[
            f"f needs {_shift(a)} > 0. √ needs ln({_shift(a)}) ≥ 0, so {_shift(a)} ≥ 1, so x ≥ {a + 1}.",
            f"Both together: {_text(ans)}.",
        ],
    )


_EASY = (_e_sqrt, _e_ln, _e_recip)
_MEDIUM = (_m_sqrt_quad, _m_ln_quad, _m_recip_root, _m_root_recip, _m_recip_square)
_HARD = (_h_root_root, _h_ln_root, _h_recip_root, _h_root_ln)


def _finish(rng, prob):
    ans = prob["ans"]
    _verify(ans, prob["defined"])
    text = _text(ans)
    read = read_solution_set(text)
    check(read is not None and read[0] == ans, "the answer text does not read back as the answer set")
    cands = []
    seen = {ans}
    for s, tag in prob["wrong"]:
        if s in seen:
            continue
        seen.add(s)
        cands.append(_dist(s, tag))
    stem = phrase(
        rng,
        (
            "Let f(x) = {f} and g(x) = {g}. Find the largest domain of the composite function g(f(x)), as an inequality.",
            "The functions f and g are given by f(x) = {f} and g(x) = {g}. What is the maximal domain of g(f(x))?",
            "For f(x) = {f} and g(x) = {g}, find the largest set of real x for which g(f(x)) is defined.",
        ),
        f=prob["f"],
        g=prob["g"],
    )
    return dict(
        text=stem,
        answer_kind="inequality",
        answer=text,
        answer_text=text,
        answer_latex="plain:" + text,
        working=prob["working"],
        distractors=pick(text, cands, 3),
    )


@generator(
    "sr2b_composite_domain",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def sr2b_composite_domain(rng, tier="easy"):
    """The largest domain of g(f(x)); the tiers are described in the module docstring."""
    families = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}
    check(tier in families, f"unknown tier {tier!r}")
    return _finish(rng, rng.choice(families[tier])(rng))


# -- exam -------------------------------------------------------------------------------------------------------


@generator("sr2b_exam_composite_domain", level=13, presets={"exam": dict(tier="exam")})
@retry_short
def sr2b_exam_composite_domain(rng, tier="exam"):
    """Exam chain on f(x) = sqrt(x - p), g(x) = 1/(x - q): least x, the excluded x, the domain."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p = rng.choice([v for v in range(-5, 6) if v])
    q = rng.choice((2, 3, 4))
    hole = p + q * q
    ans = sp.Union(_I(p, hole, False, True), _I(hole, OO, True, True))
    _verify(ans, lambda x: x - p >= 0 and math.sqrt(x - p) != q)
    text = _text(ans)
    wrong = [
        (_I(p, OO), "sr2b-excluded-value-missed"),
        (sp.Union(_I(p, q, False, True), _I(q, OO, True, True)), "sr2b-excluded-value-not-converted"),
        (sp.Union(_I(p, p + q, False, True), _I(p + q, OO, True, True)), "sr2b-inner-not-squared"),
        (sp.Union(_I(p, hole, True, True), _I(hole, OO, True, True)), "sr2b-boundary-wrongly-excluded"),
    ]
    cands_c = [_dist(s, t) for s, t in wrong if s != ans]
    cands_a = [
        int_dist(-p, "sr2b-boundary-sign-slip"),
        int_dist(p * p, "sr2b-square-root-not-taken"),
        int_dist(0, "sr2b-outer-domain-used-for-x"),
    ]
    cands_b = [
        int_dist(p + q, "sr2b-inner-not-squared"),
        int_dist(q * q, "sr2b-shift-ignored"),
        int_dist(q, "sr2b-excluded-value-not-converted"),
        int_dist(p - q * q, "sr2b-boundary-sign-slip"),
    ]
    fx = f"√({_shift(p)})"
    gx = f"1/(x - {q})"
    return dict(
        text=phrase(
            rng,
            (
                "Let f(x) = {fx} and g(x) = {gx}.",
                "Two functions are given by f(x) = {fx} and g(x) = {gx}.",
                "Consider the functions f(x) = {fx} and g(x) = {gx}.",
            ),
            fx=fx,
            gx=gx,
        ),
        parts=[
            dict(
                prompt="Find the least value of x for which f(x) is defined.",
                answer_kind="number",
                answer=p,
                answer_text=str(p),
                answer_latex=str(p),
                marks=1,
                sensible_range=(-100, 100),
                working=[f"√({_shift(p)}) needs {_shift(p)} ≥ 0, so x ≥ {p}."],
                distractors=pick(str(p), [c for c in cands_a if c.text != str(p)]),
            ),
            dict(
                prompt=f"There is one value of x in the domain of f for which g(f(x)) is undefined. Find it.",
                answer_kind="number",
                answer=hole,
                answer_text=str(hole),
                answer_latex=str(hole),
                marks=2,
                sensible_range=(-100, 100),
                follow_through=f"part_a+{q * q}",
                working=[
                    f"g is undefined when f(x) = {q}: √({_shift(p)}) = {q}, so {_shift(p)} = {q * q}, so x = {p} + {q * q} = {hole}."
                ],
                distractors=pick(str(hole), [c for c in cands_b if c.text != str(hole)]),
            ),
            dict(
                prompt="Hence find the largest domain of g(f(x)), as an inequality.",
                answer_kind="inequality",
                answer=text,
                answer_text=text,
                answer_latex="plain:" + text,
                marks=2,
                working=[f"f needs x ≥ {p} and the value x = {hole} is excluded: {text}."],
                distractors=pick(text, cands_c),
            ),
        ],
    )
