"""Exam source of `ucv1-ftc-polynomial`: an unknown coefficient, a second integral, then an unknown upper limit.

The practice tiers recover ONE unknown (a coefficient or a limit) from one stated integral. An exam question for
p(x) = a x^n chains three unknowns through the fundamental theorem of calculus:

(a) the coefficient a, from the stated value of one definite integral;
(b) the value of a second definite integral of p, from (a);
(c) the upper limit k for which the integral of p from the same lower limit to k is a stated multiple of (b),
    which needs the (n + 1)th root.

Parts (b) and (c) follow through from the earlier parts.

## Construction

a is a multiple of (n + 1) so every integral is a whole number. The key of (c) is an (n + 1)th root, correct to two
decimal places, kept clear of a rounding boundary. An independent route integrates with SymPy and solves for k
numerically.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Symbol, integrate

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)
from ._exd_distinct_parts import distinct_part_answers

_X = Symbol("x", positive=True)
_A = Symbol("a", positive=True)
_K = Symbol("k", positive=True)


def _root(value: Fraction, index: int) -> Fraction:
    """The real `index`th root of a positive Fraction, as a Fraction accurate to 1e-12."""
    return Fraction(float(value) ** (1.0 / index)).limit_denominator(10**12)


def _build(rng):
    for _ in attempts(3000, "an integral chain clear of every rounding boundary"):
        n = rng.choice([1, 2, 2, 3])
        a_unit = rng.randint(1, 6)
        a = (n + 1) * a_unit
        low = rng.randint(1, 3)
        up = low + rng.randint(1, 3)
        low2 = rng.randint(1, 3)
        up2 = low2 + rng.randint(2, 4)
        m = rng.choice([2, 3])
        value = Fraction(a * (up ** (n + 1) - low ** (n + 1)), n + 1)
        second = Fraction(a * (up2 ** (n + 1) - low2 ** (n + 1)), n + 1)
        radicand = (n + 1) * m * second / a + low ** (n + 1)
        k_exact = _root(radicand, n + 1)
        k = round_half_up(k_exact, 2)
        if abs((k_exact * 100) % 1 - Fraction(1, 2)) < Fraction(1, 20):
            continue
        keys = {a, int(second), k}
        shown = {low, up, low2, up2, n, m, int(value)}
        if any(Fraction(x) in {Fraction(s) for s in shown} for x in keys) or int(second) in (a, int(value)):
            continue
        return n, a, low, up, low2, up2, m, int(value), int(second), radicand, k_exact, k
    raise RuntimeError("no draw")


@generator("exd_x_r3a_ftc_power_chain", level=14, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_ftc_power_chain(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "enough distractors in every part"):
        n, a, low, up, low2, up2, m, value, second, radicand, k_exact, k = _build(rng)
        # Independent route: SymPy integrates; the integrals must reproduce the stated value and the key of (c).
        p = _A * _X**n
        f = p.subs(_A, a)
        check(integrate(f, (_X, low, up)) == value, "SymPy finds a different value for the stated integral")
        check(integrate(f, (_X, low2, up2)) == second, "SymPy finds a different second integral")
        at_k = integrate(f, (_X, low, _K))
        check(abs(float(at_k.subs(_K, float(k_exact))) - m * second) < 1e-6, "SymPy finds a different k")

        e = n + 1
        k_text = stated_precision_string(k, 2)
        a_d = pick_distractors(
            str(a),
            [
                Distractor(str(-a), "swapped-limits", -a),
                Distractor(str(a // e), "dropped-power-rule-factor", a // e),
                Distractor(str(a * e), "power-rule-factor-multiplied", a * e),
                Distractor(str(a + e), "power-rule-exponent-not-raised", a + e),
            ],
            3,
        )
        b_d = pick_distractors(
            str(second),
            [
                Distractor(str(-second), "swapped-limits", -second),
                Distractor(str(a * (up2**e - low2**e)), "dropped-power-rule-factor", a * (up2**e - low2**e)),
                Distractor(str(a * up2**e // e), "forgot-lower-limit", a * up2**e // e),
                Distractor(str(a * (up2**n - low2**n)), "used-integrand-not-antiderivative", a * (up2**n - low2**n)),
            ],
            3,
        )

        def kdist(value: Fraction, tag: str):
            return Distractor(stated_precision_string(round_half_up(value, 2), 2), tag, round_half_up(value, 2))

        c_d = pick_distractors(
            k_text,
            [
                kdist(_root(e * m * Fraction(second) / a, e), "forgot-lower-limit"),
                kdist(_root(e * Fraction(second) / a + low**e, e), "forgot-stated-multiplier"),
                kdist(_root(m * Fraction(second) / a + low**e, e), "dropped-power-rule-factor"),
                kdist(_root(radicand, 2) if e != 2 else _root(radicand, 3), "wrong-root-index"),
            ],
            3,
            tolerance=GRADER_TOLERANCE,
        )
        if min(len(a_d), len(b_d), len(c_d)) >= 2:
            break

    mult = {2: "twice", 3: "three times"}[m]
    power = f"x^{n}" if n > 1 else "x"
    stem = (
        f"Let p(x) = a {power}, where a is a positive constant. The definite integral of p(x) from x = {low} to "
        f"x = {up} is {value}."
    )
    ft_b = f"part_a*({up2}**{e} - {low2}**{e})/{e}"
    ft_c = f"root({e}*{m}*part_b/part_a + {low}**{e}, {e})"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of a.",
                answer_kind="number",
                answer=a,
                answer_text=str(a),
                answer_latex=latex_answer(str(a)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The antiderivative of p is a x^{e}/{e}.",
                    f"a(({up})^{e} - ({low})^{e})/{e} = {value}, so a = {a}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"Using your value of a, find the definite integral of p(x) from x = {low2} to x = {up2}.",
                answer_kind="number",
                answer=second,
                answer_text=str(second),
                answer_latex=latex_answer(str(second)),
                marks=2,
                sensible_range=(0, None),
                follow_through=ft_b,
                working=[f"{a}(({up2})^{e} - ({low2})^{e})/{e} = {second}."],
                distractors=b_d,
            ),
            dict(
                prompt=f"Using your answer to (b), find the value of k, where k > {low}, such that the definite "
                f"integral of p(x) from x = {low} to x = k is {mult} the answer to (b). "
                "Give k correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(k),
                answer_text=k_text,
                answer_latex=latex_answer(k_text),
                marks=4,
                sensible_range=(0, None),
                follow_through=ft_c,
                working=[
                    f"{a}(k^{e} - {low}^{e})/{e} = {m} × {second}.",
                    f"k^{e} = {_fmt(radicand)}, so k = {k_text} (2 d.p.).",
                ],
                distractors=c_d,
            ),
        ],
    )


def _fmt(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{float(value):.4f}"
