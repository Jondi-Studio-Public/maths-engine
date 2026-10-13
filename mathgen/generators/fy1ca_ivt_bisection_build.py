"""Generates `fy1ca_ivt_bisection` and `fy1ca_ivt_bisection_exam` questions.

One first-year skill, ``ucv1-ivt-bisection``: use the intermediate value theorem to place a root of a continuous
function, then narrow it by bisection. Every answer is a number (an integer, or a dyadic decimal shown to its stated
number of places), graded as a number.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: place a root between two consecutive integers by finding where f changes sign (give the lower integer).
- ``medium``: start from a bracketing interval, evaluate f at the midpoint, decide which half keeps the root, and give
  the new left or right endpoint (one bisection step, one decision).
- ``hard``: either repeat the step three times and give the left endpoint of the interval reached (three decisions that
  each depend on the last), or decide how many halvings are needed to bring the interval below a tolerance (no
  function evaluation, only the arithmetic of halving).
- ``exam`` (topic ``fy1ca_ivt_bisection_exam``): a four-part chain on x³ + bx - c: (a) the integer interval holding the
  root, (b) f at that interval's midpoint, (c) the endpoint kept after one bisection step, (d) the number of steps to
  reach a tolerance.

Every key is re-derived by a second route: the root is found with a float bisection to 12 digits and the claimed
interval is checked to contain it, and the number of steps is checked against the width after that many halvings.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, generator, join_terms, latex_answer, numeric_answer, phrase
from ._fy1ca_kit import (
    Redraw,
    build_with_redraws,
    grid,
    need_distractors,
    number_distractors,
    number_part,
    stated_precision_string,
)

_SEED_NAMES = ("f", "g", "h")


class _Fn:
    """A monotone increasing function with one sign change, its printed form and an exact or float evaluator."""

    def __init__(self, text: str, value, exact: bool):
        self.text = text
        self.value = value
        self.exact = exact

    def sign(self, x) -> int:
        v = self.value(x)
        if self.exact:
            return (v > 0) - (v < 0)
        if abs(v) < 1e-6:
            raise Redraw
        return 1 if v > 0 else -1


def _cubic(rng: random.Random, name: str, n_lo: int = 0, n_hi: int = 4) -> tuple[_Fn, int]:
    """f(x) = x³ + bx - c with exactly one real root, which lies strictly inside (n, n + 1)."""
    n = rng.randint(n_lo, n_hi)
    b = rng.randint(1, 6)
    lo, hi = n**3 + b * n, (n + 1) ** 3 + b * (n + 1)
    c = rng.randint(lo + 1, hi - 1)
    text = f"{name}(x) = {join_terms([(1, 'x³'), (b, 'x'), (-c, '')])}"
    fn = _Fn(text, lambda x: x**3 + b * x - c, True)
    return fn, n


def _exponential(rng: random.Random, name: str) -> tuple[_Fn, int]:
    """f(x) = e^x + x - c with its root strictly inside (n, n + 1)."""
    c = rng.randint(3, 40)
    value = lambda x: math.exp(float(x)) + float(x) - c  # noqa: E731
    n = next(k for k in range(60) if value(k + 1) > 0)
    fn = _Fn(f"{name}(x) = e^x + x - {c}", value, False)
    check(fn.sign(n) < 0 < fn.sign(n + 1), "no sign change inside the bracket")
    return fn, n


def _function(rng: random.Random, name: str, exact_only: bool = False) -> tuple[_Fn, int]:
    if exact_only or rng.random() < 0.7:
        return _cubic(rng, name)
    return _exponential(rng, name)


def _bisect(fn: _Fn, lo, hi, steps: int, flip_at: int | None = None) -> tuple:
    """The bracketing interval after `steps` bisections; `flip_at` (1-based) makes that one decision the wrong way."""
    for step in range(1, steps + 1):
        mid = (lo + hi) / 2
        keep_right = fn.sign(mid) < 0  # f(lo) < 0 < f(hi): a negative midpoint means the root is to its right
        if step == flip_at:
            keep_right = not keep_right
        lo, hi = (mid, hi) if keep_right else (lo, mid)
    return lo, hi


def _root(fn: _Fn, lo: float, hi: float) -> float:
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn.value(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def _halvings(width: Fraction, tol: Fraction) -> int:
    """The least n with width / 2^n < tol."""
    n = 0
    while width / 2**n >= tol:
        n += 1
    return n


def _dec(value) -> Fraction:
    return Fraction(value)


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    name = rng.choice(_SEED_NAMES)
    fn, n = _function(rng, name)
    lo, hi = fn.sign(n), fn.sign(n + 1)
    check(lo < 0 < hi, "f does not change sign across (n, n + 1)")
    # second route: a float bisection over a wide bracket lands inside (n, n + 1)
    root = _root(fn, 0.0, 12.0) if not fn.exact else _root(fn, Fraction(0), Fraction(12))
    check(n < float(root) < n + 1, "the root is not inside the stated interval")
    wrong = [
        (n + 1, "fy1ca-ivt-wrong-end"),
        (n - 1, "fy1ca-ivt-one-step-away"),
        (n + 2, "fy1ca-ivt-one-step-away"),
        (n + 3, "fy1ca-ivt-one-step-away"),
    ]
    text = phrase(
        rng,
        (
            "The function {fn} is continuous for all x and has exactly one real root. Find the integer n such that the root lies between n and n + 1.",
            "Let {fn}. The equation {nm}(x) = 0 has one real solution. Between which consecutive integers does it lie? Give the smaller of the two integers.",
            "Consider {fn}. By evaluating {nm} at whole numbers, find the integer n for which {nm}(n) < 0 < {nm}(n + 1).",
        ),
        fn=fn.text,
        nm=name,
    )
    working = [
        f"Evaluate {name} at consecutive whole numbers until the sign changes.",
        f"{name}({n}) is negative and {name}({n + 1}) is positive, so by the intermediate value theorem there is a root in ({n}, {n + 1}); n = {n}.",
    ]
    return _number_question(text, Fraction(n), 0, wrong, working)


def _number_question(text: str, answer: Fraction, dp: int, wrong, working) -> dict:
    distractors = number_distractors(answer, wrong, dp)
    need_distractors(distractors)
    rounded = grid(answer, dp)
    answer_text = stated_precision_string(rounded, dp)
    return dict(
        text=text,
        answer=numeric_answer(rounded),
        answer_text=answer_text,
        answer_kind="number",
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


# -- medium --------------------------------------------------------------------------------------------------------------


def _bracket(rng: random.Random, n: int, widths=(1, 3, 5)) -> tuple[int, int]:
    width = rng.choice(widths)
    lo = rng.randint(max(n - width + 1, n - 4), n)
    return lo, lo + width


def _medium(rng: random.Random) -> dict:
    name = rng.choice(_SEED_NAMES)
    fn, n = _function(rng, name)
    lo, hi = _bracket(rng, n)
    check(fn.sign(lo) < 0 < fn.sign(hi), "the interval does not bracket a sign change")
    mid = Fraction(lo + hi, 2)
    new_lo, new_hi = _bisect(fn, Fraction(lo), Fraction(hi), 1)
    ask_left = rng.random() < 0.5
    answer = new_lo if ask_left else new_hi
    other = new_hi if ask_left else new_lo
    root = float(_root(fn, float(lo), float(hi))) if not fn.exact else float(_root(fn, Fraction(lo), Fraction(hi)))
    check(float(new_lo) < root < float(new_hi), "the kept half does not hold the root")
    wrong_half = _bisect(fn, Fraction(lo), Fraction(hi), 1, flip_at=1)
    wrong = [
        (wrong_half[0] if ask_left else wrong_half[1], "fy1ca-bisection-wrong-half"),
        (other, "fy1ca-bisection-wrong-end"),
        (mid, "fy1ca-bisection-midpoint-reported"),
        (Fraction(lo if ask_left else hi), "fy1ca-bisection-no-update"),
        ((new_lo + new_hi) / 2, "fy1ca-bisection-two-steps"),
    ]
    side = "left" if ask_left else "right"
    text = phrase(
        rng,
        (
            "The function {fn} is continuous and changes sign on [{lo}, {hi}]. Apply one step of the bisection method to this interval and give the {side} endpoint of the interval that results, as a decimal.",
            "{fn} has a root in [{lo}, {hi}]. Bisect the interval at its midpoint, keep the half that still contains the root, and state its {side} endpoint as a decimal.",
            "A root of {nm}, where {fn}, is known to lie in [{lo}, {hi}]. After one bisection step, what is the {side} endpoint of the new interval? Write it as a decimal.",
        ),
        fn=fn.text,
        nm=name,
        lo=lo,
        hi=hi,
        side=side,
    )
    working = [
        f"The midpoint of [{lo}, {hi}] is {mid}. Evaluate {name}({mid}): it is {'negative' if fn.sign(mid) < 0 else 'positive'}.",
        f"{name}({lo}) < 0 < {name}({hi}), so the sign change, and so the root, is in [{new_lo}, {new_hi}]. The {side} endpoint is {answer}.",
    ]
    return _number_question(text, answer, 2, wrong, working)


# -- hard ----------------------------------------------------------------------------------------------------------------


def _hard_three_steps(rng: random.Random) -> dict:
    name = rng.choice(_SEED_NAMES)
    fn, n = _function(rng, name)
    lo, hi = Fraction(n), Fraction(n + 1)
    steps = 3
    final_lo, final_hi = _bisect(fn, lo, hi, steps)
    root = float(_root(fn, float(n), float(n + 1)))
    check(float(final_lo) < root < float(final_hi), "the final interval does not hold the root")
    wrong_a = _bisect(fn, lo, hi, steps, flip_at=1)[0]
    wrong_b = _bisect(fn, lo, hi, steps, flip_at=2)[0]
    two_steps = _bisect(fn, lo, hi, steps - 1)[0]
    wrong = [
        (wrong_b, "fy1ca-bisection-wrong-half"),
        (final_hi, "fy1ca-bisection-wrong-end"),
        (two_steps, "fy1ca-bisection-stops-early"),
        (wrong_a, "fy1ca-bisection-wrong-half"),
        ((final_lo + final_hi) / 2, "fy1ca-bisection-midpoint-reported"),
    ]
    text = phrase(
        rng,
        (
            "The function {fn} has a root between {n} and {m}. Bisect the interval [{n}, {m}] three times in succession. Give the left endpoint of the final interval, correct to 3 decimal places.",
            "Starting from [{n}, {m}], which contains a root of {fn}, carry out three steps of the bisection method. What is the left endpoint of the interval you reach? Give your answer correct to 3 decimal places.",
        ),
        fn=fn.text,
        n=n,
        m=n + 1,
    )
    working = [
        f"Step {i + 1}: interval {_bisect(fn, lo, hi, i + 1)[0]} to {_bisect(fn, lo, hi, i + 1)[1]}."
        for i in range(steps)
    ]
    working.append(f"After three steps the left endpoint is {final_lo}.")
    return _number_question(text, final_lo, 3, wrong, working)


def _hard_step_count(rng: random.Random) -> dict:
    name = rng.choice(_SEED_NAMES)
    fn, n = _function(rng, name, exact_only=True)
    width = rng.choice([1, 2, 4, 5, 8, 10])
    lo = rng.randint(max(0, n - width + 1), n)
    hi = lo + width
    check(fn.sign(lo) < 0 < fn.sign(hi), "the interval does not bracket a sign change")
    tol = rng.choice(
        [Fraction(1, 10), Fraction(1, 20), Fraction(1, 100), Fraction(1, 200), Fraction(1, 1000), Fraction(1, 10000)]
    )
    count = _halvings(Fraction(width), tol)
    check(
        Fraction(width) / 2**count < tol <= Fraction(width) / 2 ** (count - 1), "step count disagrees with the widths"
    )
    tol_text = {
        Fraction(1, 10): "0.1",
        Fraction(1, 20): "0.05",
        Fraction(1, 100): "0.01",
        Fraction(1, 200): "0.005",
        Fraction(1, 1000): "0.001",
        Fraction(1, 10000): "0.0001",
    }[tol]
    wrong = [
        (count - 1, "fy1ca-bisection-off-by-one"),
        (count + 1, "fy1ca-bisection-off-by-one"),
        (math.ceil(width / tol), "fy1ca-bisection-count-divided"),
        (math.ceil(math.log10(width / float(tol))), "fy1ca-bisection-log-base-ten"),
        (math.ceil(width / tol / 2), "fy1ca-bisection-count-divided"),
    ]
    text = phrase(
        rng,
        (
            "{fn} has a root in [{lo}, {hi}], where it changes sign. How many bisection steps are needed before the bracketing interval is narrower than {tol}?",
            "The bisection method is applied to {fn} on [{lo}, {hi}]. Find the smallest number of steps after which the interval containing the root has width less than {tol}.",
        ),
        fn=fn.text,
        lo=lo,
        hi=hi,
        tol=tol_text,
    )
    working = [
        f"Each step halves the width: after k steps it is {width}/2^k.",
        f"We need {width}/2^k < {tol_text}, that is 2^k > {width / float(tol):g}, so k = {count}.",
    ]
    return _number_question(text, Fraction(count), 0, wrong, working)


def _hard(rng: random.Random) -> dict:
    return rng.choice([_hard_three_steps, _hard_step_count])(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fy1ca_ivt_bisection", level=19, presets={t: dict(tier=t) for t in _TIERS})
def fy1ca_ivt_bisection(rng: random.Random, tier: str = "easy") -> dict:
    """Place a root with the intermediate value theorem (easy), one bisection step (medium), several steps or a count (hard)."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return build_with_redraws(rng, _TIERS[tier], f"a {tier} bisection question")


# -- exam ----------------------------------------------------------------------------------------------------------------


@generator("fy1ca_ivt_bisection_exam", level=19, presets={"exam": {}})
def fy1ca_ivt_bisection_exam(rng: random.Random) -> dict:
    """f(x) = x³ + bx - c with a root in (n, n + 1): n, f at the midpoint, the endpoint kept, steps to a tolerance."""
    return build_with_redraws(rng, _exam_build, "a bisection exam chain")


def _exam_build(rng: random.Random) -> dict:
    name = rng.choice(_SEED_NAMES)
    n = rng.randint(1, 4)
    b = rng.randint(1, 6)
    lo_v, hi_v = n**3 + b * n, (n + 1) ** 3 + b * (n + 1)
    c = rng.randint(lo_v + 1, hi_v - 1)

    def f(x):
        return x**3 + b * x - c

    mid = Fraction(2 * n + 1, 2)
    f_mid = f(mid)
    if f_mid == 0:
        raise Redraw
    kept_left = mid if f_mid < 0 else Fraction(n)
    # second route: float bisection finds the root inside the claimed first interval, and the kept half holds it
    root = float(_root(_Fn("", f, True), Fraction(0), Fraction(12)))
    check(n < root < n + 1, "the root is not inside (n, n + 1)")
    check((float(mid) < root) == (f_mid < 0), "the kept half does not hold the root")
    tol = rng.choice([Fraction(1, 10), Fraction(1, 100), Fraction(1, 200), Fraction(1, 1000)])
    tol_text = {
        Fraction(1, 10): "0.1",
        Fraction(1, 100): "0.01",
        Fraction(1, 200): "0.005",
        Fraction(1, 1000): "0.001",
    }[tol]
    steps = _halvings(Fraction(1), tol)

    a_dist = number_distractors(
        Fraction(n),
        [(n + 1, "fy1ca-ivt-wrong-end"), (n - 1, "fy1ca-ivt-one-step-away"), (n + 2, "fy1ca-ivt-one-step-away")],
        0,
    )
    b_dist = number_distractors(
        f_mid,
        [
            (f(Fraction(n)), "fy1ca-ivt-value-at-wrong-point"),
            (f(Fraction(n + 1)), "fy1ca-ivt-value-at-wrong-point"),
            (mid**3 + b - c, "fy1ca-ivt-linear-term-slip"),
            (mid**3 + b * mid + c, "fy1ca-ivt-sign-slip"),
            (3 * mid + b * mid - c, "fy1ca-ivt-cube-as-triple"),
        ],
        3,
    )
    c_dist = number_distractors(
        kept_left,
        [
            (Fraction(n) if f_mid < 0 else mid, "fy1ca-bisection-wrong-half"),
            (Fraction(n + 1), "fy1ca-bisection-wrong-end"),
            ((mid + Fraction(n + 1)) / 2 if f_mid < 0 else (mid + Fraction(n)) / 2, "fy1ca-bisection-two-steps"),
            (mid if f_mid > 0 else Fraction(n + 1), "fy1ca-bisection-midpoint-reported"),
        ],
        2,
    )
    d_dist = number_distractors(
        Fraction(steps),
        [
            (steps - 1, "fy1ca-bisection-off-by-one"),
            (steps + 1, "fy1ca-bisection-off-by-one"),
            (math.ceil(1 / tol), "fy1ca-bisection-count-divided"),
            (math.ceil(math.log10(1 / float(tol))), "fy1ca-bisection-log-base-ten"),
        ],
        0,
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        need_distractors(dist)
    stem = phrase(
        rng,
        (
            "Let {nm}(x) = {fx}. This function is continuous and increasing, so {nm}(x) = 0 has exactly one real root.",
            "The function {nm}(x) = {fx} is continuous for all x and has a single real root.",
        ),
        nm=name,
        fx=join_terms([(1, "x³"), (b, "x"), (-c, "")]),
    )
    sign_word = "negative" if f_mid < 0 else "positive"
    parts = [
        number_part(
            f"Find the integer n such that the root lies between n and n + 1.",
            Fraction(n),
            0,
            2,
            [f"{name}({n}) = {f(n)} < 0 and {name}({n + 1}) = {f(n + 1)} > 0, so the root is between {n} and {n + 1}."],
            a_dist,
            sensible_range=(0, None),
        ),
        number_part(
            f"Hence evaluate {name} at the midpoint of the interval from n to n + 1, correct to 3 decimal places.",
            f_mid,
            3,
            2,
            [f"The midpoint is {n} + 0.5 = {float(mid)}. {name}({float(mid)}) = {float(f_mid):.3f}."],
            b_dist,
            follow_through=f"(part_a + 0.5)**3 + {b}*(part_a + 0.5) - {c}",
        ),
        number_part(
            f"After one bisection step on the interval from n to n + 1, what is the left endpoint of the interval that contains the root? Give your answer to 2 decimal places.",
            kept_left,
            2,
            1,
            [
                f"{name}({float(mid)}) is {sign_word}, so the root is in the {'right' if f_mid < 0 else 'left'} half; the left endpoint is {float(kept_left):.2f}."
            ],
            c_dist,
            follow_through="part_a + (1 - sign(part_b))/4",
            sensible_range=(0, None),
        ),
        number_part(
            f"Starting from the interval from n to n + 1, how many bisection steps are needed altogether before the interval containing the root is narrower than {tol_text}?",
            Fraction(steps),
            0,
            1,
            [f"After k steps the width is 1/2^k. We need 1/2^k < {tol_text}, so k = {steps}."],
            d_dist,
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
