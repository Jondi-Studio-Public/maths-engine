"""Exam-tier source for `bisection-interval-update`: two bisection steps and how many steps a tolerance needs.

The practice tiers update a bracket once for `x^2 - c`. The exam question brackets the positive root of a power
function `x^p - c` or `c - x^p` (p = 2 or 3), so the sign test that picks the half changes with the function, and asks
for three things that need the bracket idea rather than one update:

1. the value of `f` at the first midpoint,
2. the bracket after TWO bisection steps (the second decision depends on the first), and
3. the fewest steps after which the bracket is guaranteed narrower than a stated tolerance (width over `2^n`).

Built from an integer bracket around the root. The independent route recomputes every sign with exact fractions and
finds the step count by direct halving rather than by logarithm.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import check, generator, pick_distractors, rational_text

from ._u4_kit import fraction_distractors, fraction_payload

_TOLERANCES = [Fraction(1, 10), Fraction(1, 20), Fraction(1, 50), Fraction(1, 100), Fraction(1, 200), Fraction(1, 1000)]
_TOLERANCE_TEXT = {
    Fraction(1, 10): "0.1",
    Fraction(1, 20): "0.05",
    Fraction(1, 50): "0.02",
    Fraction(1, 100): "0.01",
    Fraction(1, 200): "0.005",
    Fraction(1, 1000): "0.001",
}


def _is_perfect_power(c: int, power: int) -> bool:
    return round(c ** (1 / power)) ** power == c


def _pair_value(left: Fraction, right: Fraction) -> dict:
    return {"left": left, "right": right}


def _pair_text(left: Fraction, right: Fraction) -> str:
    return f"left = {rational_text(left)}, right = {rational_text(right)}"


def _halve(f, left: Fraction, right: Fraction, increasing: bool, keep_wrong: bool = False) -> tuple[Fraction, Fraction]:
    mid = (left + right) / 2
    positive = f(mid) > 0
    root_on_left = positive if increasing else not positive
    if keep_wrong:
        root_on_left = not root_on_left
    return (left, mid) if root_on_left else (mid, right)


@generator("exd_x_u4_bisection_steps", level=16, presets={"exam": dict(tier="exam")})
def exd_x_u4_bisection_steps(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: midpoint value, the bracket after two bisection steps, and the steps needed for a tolerance."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(1000):
        power = rng.choice([2, 3])
        c = rng.randint(3, 60 if power == 2 else 120)
        if _is_perfect_power(c, power):
            continue
        root_floor = int(c ** (1 / power))
        while (root_floor + 1) ** power <= c:
            root_floor += 1
        while root_floor**power > c:
            root_floor -= 1
        left = max(0, root_floor - rng.randint(0, 3))
        right = root_floor + 1 + rng.randint(0, 3)
        width = right - left
        if width not in (4, 5, 6, 7, 8) or left**power >= c or right**power <= c:
            continue
        increasing = rng.random() < 0.6
        tol = rng.choice(_TOLERANCES)
        steps = 0
        w = Fraction(width)
        while w >= tol:  # by direct halving, not by a logarithm
            w /= 2
            steps += 1
        if w * 2 == tol or steps < 4:
            continue  # a width that lands exactly on the tolerance would make "less than" ambiguous
        break
    else:
        check(False, "no usable bracket in 1000 tries")

    def f(x: Fraction) -> Fraction:
        return x**power - c if increasing else c - x**power

    check(f(Fraction(left)) * f(Fraction(right)) < 0, "the starting bracket does not change sign")
    mid1 = Fraction(left + right, 2)
    value1 = f(mid1)
    step1 = _halve(f, Fraction(left), Fraction(right), increasing)
    step2 = _halve(f, *step1, increasing)
    # Independent route: each bracket still changes sign, and its width has halved.
    for lo, hi, expected in ((*step1, Fraction(width, 2)), (*step2, Fraction(width, 4))):
        check(f(lo) * f(hi) < 0 and hi - lo == expected, "a bracket lost the root or did not halve")
    check(Fraction(width, 2**steps) < tol <= Fraction(width, 2 ** (steps - 1)), "step count is not the least")

    fn_text = (
        f"f(x) = x{'²' if power == 2 else '³'} − {c}" if increasing else f"f(x) = {c} − x{'²' if power == 2 else '³'}"
    )
    stem = (
        f"The positive root of {fn_text} is located by the bisection method, starting from the bracket "
        f"[{left}, {right}], on which f changes sign. Each step keeps the half of the bracket on which f changes sign."
    )
    a_candidates = [
        (mid1**power, "bisection-unsquared-midpoint"),
        ((f(Fraction(left)) + f(Fraction(right))) / 2, "bisection-averages-endpoint-values"),
        (-value1, "gap-sign-reversed"),
    ]
    one_step = _pair_text(*step1)
    wrong_second = _halve(f, *step1, increasing, keep_wrong=True)
    wrong_first = _halve(f, Fraction(left), Fraction(right), increasing, keep_wrong=True)
    after_wrong = _halve(f, *wrong_first, increasing)
    b_candidates = [
        Distractor(one_step, "bisection-one-step-only", _pair_value(*step1)),
        Distractor(_pair_text(*wrong_second), "keeps-wrong-half", _pair_value(*wrong_second)),
        Distractor(_pair_text(*after_wrong), "keeps-wrong-half", _pair_value(*after_wrong)),
    ]
    n_text = str(steps)
    ignoring_width = 0
    w2 = Fraction(1)
    while w2 >= tol:
        w2 /= 2
        ignoring_width += 1
    linear = int(Fraction(width) / tol) + 1
    c_candidates = [
        Distractor(str(ignoring_width), "bisection-steps-ignore-width", ignoring_width),
        Distractor(str(linear), "bisection-width-shrinks-linearly", linear),
        Distractor(str(steps - 1), None, steps - 1),
        Distractor(str(steps + 1), None, steps + 1),
    ]
    answer_pair = _pair_value(*step2)
    pair_text = _pair_text(*step2)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find f at the midpoint of the starting bracket (an exact fraction or integer).",
                answer_kind="fraction",
                answer=fraction_payload(value1)[0],
                answer_text=rational_text(value1),
                marks=1,
                working=[
                    f"The midpoint is {rational_text(mid1)}, so f({rational_text(mid1)}) = {rational_text(value1)}."
                ],
                distractors=fraction_distractors(value1, a_candidates),
            ),
            dict(
                prompt="Perform two bisection steps from the starting bracket and give the new bracket as "
                "left = ..., right = ....",
                answer_kind="pair",
                answer=answer_pair,
                answer_text=pair_text,
                marks=2,
                working=[
                    f"Step 1 keeps {_pair_text(*step1)}.",
                    f"Step 2 uses the midpoint {rational_text(sum(step1) / 2)} and keeps {pair_text}.",
                ],
                distractors=pick_distractors(pair_text, b_candidates, 3),
            ),
            dict(
                prompt=f"What is the fewest number of bisection steps, starting from [{left}, {right}], that "
                f"guarantees the root is in a bracket of width less than {_TOLERANCE_TEXT[tol]}?",
                answer_kind="number",
                answer=steps,
                answer_text=n_text,
                marks=2,
                working=[
                    f"After n steps the width is {width}/2ⁿ; this first falls below {_TOLERANCE_TEXT[tol]} at n = {steps}."
                ],
                distractors=pick_distractors(n_text, c_candidates, 3),
            ),
        ],
    )
