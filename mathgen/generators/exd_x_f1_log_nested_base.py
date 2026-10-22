"""Exam-tier source for ``senior-define-logarithms-as-indices-a-x-b-is``.

The practice tiers apply the definition once: ``a^x = b  <=>  x = log_a(b)``, reading off the
exponent, the value, or a negative exponent. Here the definition has to be applied in a setting the
learner has not been given a procedure for. Two forms:

* ``nested``: ``log_a(log_b(x)) = c``. The definition is applied twice, from the outside in:
  ``log_b(x) = a^c`` and then ``x = b^(a^c)``.
* ``base``: ``log_x(N) = c`` with the BASE unknown, ``N = x^c`` (so ``N`` is a unit fraction when
  ``c`` is negative). The learner must turn ``x^c = N`` into a value of ``x`` with ``x > 0``,
  ``x != 1``.

All answers are positive integers, built backwards from the chosen answer.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import integer_log

from ..core import check, generator, phrase, pick_distractors

_NESTED_WORDINGS = (
    "Find x if log_{a}(log_{b}(x)) = {c}.",
    "Solve log_{a}(log_{b}(x)) = {c} for x.",
    "The number x satisfies log_{a}(log_{b}(x)) = {c}. What is x?",
)
_BASE_WORDINGS = (
    "Find the base x (with x > 0 and x not equal to 1) if log_x({n}) = {c}.",
    "Solve log_x({n}) = {c} for x, where x > 0 and x is not 1.",
    "The positive number x, not equal to 1, satisfies log_x({n}) = {c}. What is x?",
)


def _nested(rng: random.Random) -> dict:
    options = [(a, b, c) for a in (2, 3) for b in (2, 3, 4, 5, 10) for c in (1, 2, 3) if b ** (a**c) <= 1_000_000]
    a, b, c = rng.choice(options)
    inner = a**c
    answer = b**inner
    # Independent route: undo each logarithm with exact integer logarithms.
    check(integer_log(answer, b) == (inner, True), "x is not b to the stated power")
    check(integer_log(inner, a) == (c, True), "inner value is not a to the stated power")
    wrong = [
        inner,  # stopped after the inner logarithm: stated log_b(x)
        b * inner,  # read the power as a multiplier: b * a^c
        b ** (a * c),  # exponents combined as (b^a)^c
        a ** (b**c),  # tower built in the wrong order
    ]
    text = phrase(rng, _NESTED_WORDINGS, a=a, b=b, c=c)
    latex = rf"\log_{{{a}}}\left(\log_{{{b}}}(x)\right) = {c}"
    working = [f"log_{b}(x) = {a if c == 1 else f'{a}^{c}'} = {inner}, so x = {b}^{inner} = {answer}."]
    return dict(text=text, latex=latex, answer=answer, wrong=wrong, working=working)


def _base(rng: random.Random) -> dict:
    x = rng.choice([2, 3, 4, 5, 10])
    c = rng.choice([-4, -3, -2, 2, 3, 4])
    power = Fraction(x) ** c
    n_text = f"1/{x ** (-c)}" if c < 0 else str(x**c)
    # Independent route: exhaustive search over candidate bases for x^c = N.
    solutions = [b for b in range(2, 200) if Fraction(b) ** c == power]
    check(solutions == [x], "base is not unique")
    wrong = [
        x ** abs(c),  # stated the number inside the logarithm
        abs(c),  # stated the index
        x * abs(c),  # multiplied the base by the index
        x + 1,  # off by one
    ]
    text = phrase(rng, _BASE_WORDINGS, n=n_text, c=c)
    latex = rf"\log_{{x}}\left({n_text if c > 0 else rf'\frac{{1}}{{{x ** (-c)}}}'}\right) = {c}"
    working = [f"x^({c}) = {n_text}, so x = {x}."]
    return dict(text=text, latex=latex, answer=x, wrong=wrong, working=working)


def _build(rng: random.Random) -> dict:
    for _ in range(100):
        d = _nested(rng) if rng.random() < 0.5 else _base(rng)
        wrong = []
        for w in d["wrong"]:
            if w != d["answer"] and w > 0 and w not in wrong:
                wrong.append(w)
        if len(wrong) < 3:
            continue
        wrong = wrong[:3]
        check(all(w != d["answer"] for w in wrong))
        answer_text = str(d["answer"])
        return dict(
            text=d["text"],
            latex=d["latex"],
            answer=d["answer"],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=d["working"],
            distractors=pick_distractors(answer_text, [str(w) for w in wrong]),
        )
    check(False, "could not build a logarithm question")


@generator(
    "exd_x_f1_log_nested_base",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f1_log_nested_base(rng: random.Random, tier: str = "exam") -> dict:
    """Logarithm definition applied to a nested logarithm or to an unknown base."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _build(rng)
