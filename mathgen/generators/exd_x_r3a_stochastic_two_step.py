"""Exam source of `ucv1-stochastic-euler-step`: two Euler-Maruyama steps, then the effect of the noise.

The practice tiers take one Euler-Maruyama step. An exam question takes two with multiplicative noise and then asks
what the noise did:

(a) Y1 from Y0 (drift, step size and the noise value dW1 scaled by sigma(Y0) = k Y0);
(b) Y2 from Y1, re-evaluating both the drift and the diffusion coefficient at Y1;
(c) the difference between Y2 and the approximation the same two steps give with the noise switched off, that is the
    amount by which the stated noise values moved the approximation.

Parts (b) and (c) follow through from the earlier parts. Every noise value is a stated rational number.

## Construction

All arithmetic is exact (`Fraction`). An independent route recomputes the two steps with SymPy `Rational` in a
different association of terms. Distractors are the slips of the single-step skill carried into the second step, and
the slips particular to chaining: coefficients not re-evaluated at Y1, a noise value used twice.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Rational

from mathgen.answers import Distractor, exact_value
from mathgen.core import attempts, check, generator, join_terms, latex_answer, pick_distractors, rational_text
from ._exd_distinct_parts import distinct_part_answers


def _t(value) -> str:
    return rational_text(value)


def _fr(value: Fraction) -> str:
    """A Fraction as a follow-through literal."""
    return f"({value.numerator}/{value.denominator})"


def _draw_noise(rng) -> Fraction:
    num = rng.randint(1, 3)
    den = rng.choice([1, 2, 3])
    return Fraction(rng.choice([-1, 1]) * num, den)


def _part(prompt, value, marks, distractors, working, follow_through=None):
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=_t(value),
        answer_latex=latex_answer(_t(value)),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _tag(pairs, answer):
    out = []
    for value, mid in pairs:
        value = Fraction(value)
        if value != answer:
            out.append(Distractor(_t(value), mid, value))
    return pick_distractors(_t(answer), out, 3)


def _build(rng):
    for _ in attempts(3000, "a two-step path with distinct, non-trivial keys"):
        lam, forcing = rng.randint(-3, 3), rng.randint(-3, 3)
        k = rng.choice([-3, -2, -1, 1, 2, 3])
        y0 = rng.choice([v for v in range(-6, 7) if v not in (0, 1, -1)])
        h = Fraction(1, rng.randint(2, 6))
        dw1, dw2 = _draw_noise(rng), _draw_noise(rng)
        if dw1 == dw2 or (lam == 0 and forcing == 0):
            continue
        if 1 + h * lam + k * dw1 == 0 or 1 + h * lam + k * dw2 == 0:
            continue  # the next value would not depend on the last
        d1 = y0 + h * (lam * y0 + forcing)
        y1 = d1 + k * y0 * dw1
        d2 = d1 + h * (lam * d1 + forcing)
        y2 = y1 + h * (lam * y1 + forcing) + k * y1 * dw2
        diff = y2 - d2
        values = {y1, y2, diff}
        if len(values) < 3 or 0 in values or d1 == y0 or y1 == y0 or diff == y2:
            continue
        shown = {Fraction(v) for v in (lam, forcing, k, y0, h, dw1, dw2)}
        if values & shown:
            continue
        if max(abs(y1), abs(y2), abs(diff)) > 400 or max(v.denominator for v in values) > 5000:
            continue
        return lam, forcing, k, y0, h, dw1, dw2, d1, y1, d2, y2, diff
    raise RuntimeError("no draw")


@generator("exd_x_r3a_stochastic_two_step", level=15, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_stochastic_two_step(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "enough distractors in every part"):
        lam, forcing, k, y0, h, dw1, dw2, d1, y1, d2, y2, diff = _build(rng)
        # Independent route: SymPy rationals, drift and diffusion grouped differently.
        R = Rational
        s_y0, s_h = R(y0), R(h.numerator, h.denominator)
        s1 = s_y0 * (1 + s_h * lam + k * R(dw1.numerator, dw1.denominator)) + s_h * forcing
        s2 = s1 * (1 + s_h * lam + k * R(dw2.numerator, dw2.denominator)) + s_h * forcing
        check(s1 == R(y1.numerator, y1.denominator) and s2 == R(y2.numerator, y2.denominator), "SymPy disagrees")

        a_d = _tag(
            [
                (d1, "diffusion-term-omitted"),
                (d1 + k * dw1, "noise-coefficient-without-state"),
                (d1 + k * y0 * h * dw1, "noise-scaled-by-step-size"),
                (d1 - k * y0 * dw1, "noise-sign-flipped"),
                (y0 + dw1 * (lam * y0 + forcing) + k * y0 * dw1, "noise-value-used-as-step-size"),
            ],
            y1,
        )
        b_d = _tag(
            [
                (y1 + h * (lam * y1 + forcing) + k * y0 * dw2, "diffusion-not-re-evaluated"),
                (y1 + h * (lam * y0 + forcing) + k * y1 * dw2, "drift-not-re-evaluated"),
                (y1 + h * (lam * y1 + forcing), "diffusion-term-omitted"),
                (y1 + h * (lam * y1 + forcing) + k * y1 * dw1, "noise-value-reused"),
                (y1 + h * (lam * y1 + forcing) - k * y1 * dw2, "noise-sign-flipped"),
            ],
            y2,
        )
        c_d = _tag(
            [
                (d2, "deterministic-value-reported"),
                (d2 - y2, "reversed-subtraction"),
                (y2 - d1, "compared-with-wrong-step"),
                (y1 - d1, "first-step-effect-only"),
                (y2 - y0, "compared-with-start-value"),
            ],
            diff,
        )
        if min(len(a_d), len(b_d), len(c_d)) >= 2:
            break

    drift = join_terms([(lam, "Y"), (forcing, "")])
    stem = (
        f"A process satisfies dY = ({drift}) dt + σ(Y) dW, where σ(Y) = {join_terms([(k, 'Y')])} and Y(0) = {y0}. Two "
        f"Euler-Maruyama steps of size h = {_t(h)} are taken, with the drift and the diffusion coefficient "
        f"re-evaluated at the start of each step. The stated noise values are ΔW1 = {_t(dw1)} for the first step "
        f"and ΔW2 = {_t(dw2)} for the second."
    )
    ft_b = f"part_a + {_fr(h)}*({lam}*part_a + {forcing}) + {k}*part_a*{_fr(dw2)}"
    ft_c = f"part_b - {_fr(d2)}"
    return dict(
        text=stem,
        parts=[
            _part(
                "Find the approximation Y1 of Y(h).",
                y1,
                2,
                a_d,
                [f"Y1 = {y0} + {_t(h)}({lam}·{y0} + {forcing}) + {k}·{y0}·({_t(dw1)}) = {_t(y1)}."],
            ),
            _part(
                "Using your value of Y1, find the approximation Y2 of Y(2h).",
                y2,
                3,
                b_d,
                [f"Y2 = Y1 + h(λY1 + f) + kY1ΔW2 = {_t(y2)}."],
                ft_b,
            ),
            _part(
                "The same two steps are repeated with both noise values set to 0. Using your value of Y2, find "
                "the amount by which the stated noise changes the approximation of Y(2h), that is Y2 minus "
                "the value without noise.",
                diff,
                3,
                c_d,
                [
                    f"Without noise: D1 = {_t(d1)}, D2 = {_t(d2)}.",
                    f"Y2 - D2 = {_t(y2)} - ({_t(d2)}) = {_t(diff)}.",
                ],
                ft_c,
            ),
        ],
    )
