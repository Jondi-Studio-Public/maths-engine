"""Exam-tier source for ``senior-define-the-natural-logarithm-ln-x-log-ex``.

The practice tiers evaluate ``ln`` of a stated value, or invert ``ln x = k``. Here the learner meets
an exponential equation that is a QUADRATIC IN ``e^x`` (or reduces to one after multiplying through
by ``e^x``), must recognise that structure, solve for the two values of ``e^x``, and only then apply
the natural logarithm to each. They are then asked for one solution or for the sum of both.

    e^(2x) - (p + q) e^x + p q = 0      or      e^x + p q e^(-x) = p + q

Both have ``e^x`` equal to ``p`` or ``q``, so the solutions are ``ln p`` and ``ln q``. Answers are
given to 3 decimal places.
"""

from __future__ import annotations

import math
import random

from ..core import check, generator, phrase, pick_distractors, rounds_unambiguously

_WORDINGS = {
    "larger": "Solve {eq} for x. Give the larger solution to 3 decimal places.",
    "smaller": "Solve {eq} for x. Give the smaller solution to 3 decimal places.",
    "sum": "The equation {eq} has two real solutions. Find their sum, to 3 decimal places.",
}
_MARGIN = 0.006  # distractors stay clear of the 3 d.p. grading tolerance (0.0005)


def _equation(rng: random.Random, p: int, q: int) -> tuple[str, str]:
    if rng.random() < 0.5:
        s, prod = p + q, p * q
        return f"e^(2x) - {s}e^x + {prod} = 0", rf"e^{{2x}} - {s}e^{{x}} + {prod} = 0"
    return f"e^x + {p * q}e^(-x) = {p + q}", rf"e^{{x}} + {p * q}e^{{-x}} = {p + q}"


def _build(rng: random.Random) -> dict:
    for _ in range(200):
        p, q = sorted(rng.sample(range(2, 13), 2))
        which = rng.choice(["larger", "smaller", "sum"])
        value = {"larger": math.log(q), "smaller": math.log(p), "sum": math.log(p * q)}[which]
        if not rounds_unambiguously(value, 3, margin=0.1):
            continue
        # Independent route: the two values of e^x are the roots of t^2 - (p+q)t + pq.
        disc = (p + q) ** 2 - 4 * p * q
        t_hi, t_lo = ((p + q) + math.isqrt(disc)) / 2, ((p + q) - math.isqrt(disc)) / 2
        check(math.isqrt(disc) ** 2 == disc and (t_hi, t_lo) == (q, p), "roots of the quadratic in e^x")
        pool = {
            # wrong root; forgot ln (stated e^x); common log; ln of the sum of the e^x values
            "larger": [math.log(p), float(q), math.log10(q), math.log(p + q)],
            "smaller": [math.log(q), float(p), math.log10(p), math.log(p + q)],
            "sum": [math.log(p + q), float(p + q), math.log10(p * q), math.log(q)],
        }[which]
        answer = round(value, 3)
        wrong: list[float] = []
        for w in pool:
            w = round(w, 3)
            if abs(w - answer) > _MARGIN and all(abs(w - o) > _MARGIN for o in wrong):
                wrong.append(w)
        if len(wrong) < 3:
            continue
        eq_text, eq_latex = _equation(rng, p, q)
        answer_text = f"{answer:.3f}"
        return dict(
            text=phrase(rng, [_WORDINGS[which]], eq=eq_text),
            latex=rf"{eq_latex}\ \text{{Give your answer to 3 decimal places}}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"With t = e^x the equation factors as (t - {p})(t - {q}) = 0, so e^x = {p} or {q}."],
            distractors=pick_distractors(answer_text, [f"{w:.3f}" for w in wrong[:3]]),
        )
    check(False, "could not build a natural logarithm question")


@generator(
    "exd_x_f1_ln_exponential_equation",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f1_ln_exponential_equation(rng: random.Random, tier: str = "exam") -> dict:
    """Exponential equation that is quadratic in e^x, solved with the natural logarithm."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _build(rng)
