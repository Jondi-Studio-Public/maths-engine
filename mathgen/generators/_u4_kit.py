"""Shared helpers for the numerical-methods and optimisation exam-tier generators.

Not a generator module (the leading underscore keeps it out of the registry scan). It builds exact-fraction
answers and tagged distractors the same way for each generator, so the per-skill modules only say what the errors
are. The misconception ids they tag are in `mathgen.misconceptions`.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import pick_distractors, rational_text


def fraction_payload(value) -> tuple[dict, str]:
    """``({"p", "q"}, "p/q")`` for an exact value, the way a `fraction`-graded answer is stored."""
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


def fraction_distractors(key, candidates, n: int = 3) -> list[Distractor]:
    """Up to `n` tagged distractors for an exact fraction answer.

    `candidates` is an ordered list of ``(value, misconception id or None)``; order is likelihood. A candidate equal
    to the key, or to an earlier candidate, is dropped (the grader compares exact fractions).
    """
    key = Fraction(key)
    options: list[Distractor] = []
    for value, misconception in candidates:
        value = Fraction(value)
        options.append(Distractor(rational_text(value), misconception, value))
    picked = pick_distractors(rational_text(key), options, n)
    return [d for d in picked if isinstance(d, Distractor) and d.value != key]


def survives_nsimplify(*values) -> bool:
    """Whether each exact value is returned unchanged by `sympy.nsimplify`.

    The follow-through marker restates a derived value through `nsimplify`, which can rewrite a rational with a large
    numerator and denominator as an unrelated expression. A generator draws again when its canonical answers would be
    restated that way, so every part's follow-through can be worked out.
    """
    import sympy

    for value in values:
        exact = sympy.Rational(Fraction(value).numerator, Fraction(value).denominator)
        if sympy.nsimplify(exact) != exact:
            return False
    return True
