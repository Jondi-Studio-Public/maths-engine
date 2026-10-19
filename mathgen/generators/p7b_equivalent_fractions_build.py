"""Equivalent fractions with related denominators (``f10-equivalent-fractions-related-denominators``, AC9M4N03,
AC9M4N04, AC9M5N03).

Scope: "Generate an equivalent fraction with a related denominator for
a supplied fraction." Only that clause is built here. The dropped clause
("connect each representation to a decimal or visual model") is the future
``f10-fraction-equivalence-numberline`` skill's territory (it lists this
skill as its own prerequisite) -- this generator never produces a decimal
connection or a number-line representation, and never asks the learner to
simplify or reduce (``exclusions: general rational simplification`` --
this skill only ever scales a fraction UP to a related, larger
denominator).

Three tiers, escalating the reasoning *shape*, not the number range --
easy and hard draw the original denominator ``d`` and scale factor ``k``
from the same small pools; medium's (d, k) pool is a restricted subset of
that same range (never a wider one), narrowed only so its target
denominator always has an ordinary word (see ``_NAMED_PAIRS``):

- ``direct`` (easy): the target denominator ``D`` is stated as a bare
  number ("Write n/d as an equivalent fraction with denominator D").
- ``named`` (medium): the target denominator is not given as a digit at
  all -- it is named with an ordinal-fraction word ("...in twelfths").
  The learner must first decode the word to a number before scaling,
  which is a genuine extra reasoning step, not a bigger number.
- ``backward`` (hard): the equivalence is stated as an equation with the
  new fraction's value missing ("n/d = ?/D. What is the missing
  equivalent fraction?"). The learner must recover the scale factor from
  an abstract equivalence statement rather than following an instruction
  to "write" a fraction -- solving for an unknown component of the
  equivalence, not producing it forward.

**Grading discipline (read before touching a distractor):** this skill's
grader is ``grade_fraction``, which compares the typed answer to the
canonical by exact ``Fraction`` VALUE, not display form (``mathgen/grading/typed.py``
-- a typed "2/4" is accepted for a canonical "1/2"). Every equivalent
fraction of the same original fraction has, by definition, the same value
as every other equivalent fraction of it -- so a distractor can never be
"the right value, a different denominator" (``grade_fraction`` would
correctly accept it, which is not a distractor at all). Every misconception
candidate below is instead a genuinely different fraction VALUE, proved
distinct from the correct answer by construction (see the docstring on
``_misconception_fractions``) and re-checked with ``check_distinct`` on
every seed, unconditionally.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors

# Original denominator and scale-factor pools, shared by all three shapes -- the escalation between
# tiers is the question's *shape*, never a bigger range of numbers.
_DENOMS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
_SCALE_FACTORS = (2, 3, 4)

# Easy only: the easy tier is restricted to "a small integer
# multiplier (2 or 3)" specifically.
_EASY_SCALE_FACTORS = (2, 3)

# Named tier only ("named" shape): an explicit (d, k) pool, rather than the
# general _DENOMS x _SCALE_FACTORS product, restricted to target
# denominators D = d*k a Year 4/5 learner has an ordinary word for --
# excludes d=7 (14/21/28ths are not words a learner meets) and the
# d=8,k=4 / d=9,k=3 / d=9,k=4 combinations (32nds, 27ths, 36ths). Every D
# this pool can produce is in {6, 8, 10, 12, 15, 16, 18, 20, 24, 30, 40}.
_NAMED_PAIRS = (
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 4),
    (4, 2),
    (4, 3),
    (5, 2),
    (5, 3),
    (5, 4),
    (6, 2),
    (6, 3),
    (6, 4),
    (8, 2),
    (8, 3),
    (9, 2),
    (10, 2),
    (10, 3),
    (10, 4),
)

# One-off helper (used only by this skill's "named" tier -- not put in
# mathgen/core.py because no other skill currently needs
# an English ordinal-fraction word). Covers every denominator this
# generator can ever produce (max 10 * 4 = 40).
_ONES = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
}
_TEENS = {
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
}
_TENS_STEM = {2: "twent", 3: "thirt", 4: "fort", 5: "fift", 6: "sixt", 7: "sevent", 8: "eight", 9: "ninet"}


def _ordinal_word(n: int) -> str:
    if n in _ONES:
        return _ONES[n]
    if n in _TEENS:
        return _TEENS[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return f"{_TENS_STEM[tens]}ieth"
    return f"{_TENS_STEM[tens]}y-{_ONES[ones]}"


def _denominator_word(n: int) -> str:
    """Plural ordinal-fraction word for a denominator, e.g. 12 -> 'twelfths'.
    Only ever called with n >= 4 by this module (the smallest possible
    target denominator, 2 * 2), so the irregular 'halves' plural for 2 is
    never reached."""
    return _ordinal_word(n) + "s"


def _draw_scale_factor(rng, d: int, pool: tuple[int, ...]) -> int:
    """A scale factor k from pool, excluding k == d.

    k == d is excluded structurally: it is the one case where the
    'used the original denominator as the scale factor' misconception
    (see _misconception_fractions) collapses onto the correct answer's own
    value, which would make that candidate a second correct option rather
    than a distractor.
    """
    choices = [k for k in pool if k != d]
    return rng.choice(choices)


def _verify_equivalent_fraction(n: int, d: int, numerator: int, denominator: int) -> None:
    """Independent check, two routes that must agree:

    Route A (scaling): the constructed fraction has the same VALUE as the
    original -- scaling a fraction can never change what it names.
    Route B (cross-multiplication): n * denominator == numerator * d,
    checked directly on the integers actually presented to the learner,
    independent of Fraction's own internal reduction.
    """
    check_equivalent(
        Fraction(numerator, denominator),
        Fraction(n, d),
        f"{numerator}/{denominator} is not equivalent to {n}/{d}",
    )
    check(
        n * denominator == numerator * d,
        f"cross-multiplication failed: {n} * {denominator} != {numerator} * {d}",
    )


def _misconception_fractions(n: int, d: int, k: int, D: int) -> list[Fraction]:
    """Four distractor candidates, every one a genuinely different fraction
    VALUE from the correct answer n*k/D (proved below, not merely assumed --
    grade_fraction compares by value, so 'right value, wrong denominator' is
    never a valid distractor here):

    - scaled the numerator but left the denominator alone: n*k/d.
      Equal to the answer iff k == 1; k is always >= 2 here, so never equal
      (cross-multiply: n*k*D == n*k*d*k <=> D == d*k <=> k == 1).
    - scaled the denominator but left the numerator alone: n/D.
      Equal to the answer iff k == 1, same reasoning, never equal.
    - added the scale factor instead of multiplying: (n+k)/(d+k).
      Cross-multiplying against n*k/D reduces to k^2 * (d - n) == 0; since
      n is drawn from 1..d-1 (a proper fraction), d - n >= 1, so this is
      never equal.
    - used the wrong scale factor -- the original denominator used as if it
      were the scale factor: n*d/D. This reduces to n/k; the answer reduces
      to n/d; equal iff k == d, which _draw_scale_factor excludes by
      construction.
    """
    scaled_numerator_only = Fraction(n * k, d)
    scaled_denominator_only = Fraction(n, D)
    added_scale_factor = Fraction(n + k, d + k)
    wrong_scale_factor = Fraction(n * d, D)
    return [scaled_numerator_only, scaled_denominator_only, added_scale_factor, wrong_scale_factor]


@generator(
    "equivalent_fraction_related_denominator",
    level=5,
    presets={
        "easy": dict(shape="direct"),
        "medium": dict(shape="named"),
        "hard": dict(shape="backward"),
    },
)
def equivalent_fraction_related_denominator(rng, shape="direct"):
    """Generate an equivalent fraction with a related denominator for a
    supplied fraction -- never a decimal connection, never a number line,
    never a simplification (see module docstring)."""
    if shape == "named":
        d, k = rng.choice(_NAMED_PAIRS)
    else:
        d = rng.choice(_DENOMS)
        pool = _EASY_SCALE_FACTORS if shape == "direct" else _SCALE_FACTORS
        k = _draw_scale_factor(rng, d, pool)
    n = rng.randint(1, d - 1)  # proper fraction: n < d, needed for the
    # added-scale-factor distinctness proof above
    D = d * k
    numerator = n * k

    _verify_equivalent_fraction(n, d, numerator, D)
    answer = Fraction(numerator, D)
    answer_text = f"{numerator}/{D}"

    if shape == "direct":
        text = f"Write {n}/{d} as an equivalent fraction with denominator {D}."
    elif shape == "named":
        text = f"Write {n}/{d} as an equivalent fraction in {_denominator_word(D)}."
    elif shape == "backward":
        text = f"{n}/{d} = ?/{D}. What is the missing equivalent fraction?"
    else:
        raise ValueError(f"unknown shape {shape!r}")

    # check_distinct proves the four candidate VALUES (Fraction, auto-reduced)
    # are genuinely different from the answer's value -- see
    # _misconception_fractions' docstring for the by-construction proof.
    # Display text is built separately, in the *unreduced* raw form the
    # misconception actually produces (n*k/d, not its reduced equivalent) --
    # otherwise every distractor would print reduced while only the answer
    # keeps the asked-for denominator D, making the answer the one
    # identifiably different-shaped option (the midpoint_coordinate_scalar
    # defect the conventions warn about).
    candidates = _misconception_fractions(n, d, k, D)
    check_distinct(answer, *candidates, msg="a distractor equals the correct equivalent fraction's value")
    distractor_texts = [
        f"{n * k}/{d}",  # scaled numerator only
        f"{n}/{D}",  # scaled denominator only
        f"{n + k}/{d + k}",  # added the scale factor instead of multiplying
        f"{n * d}/{D}",  # used the original denominator as the scale factor
    ]
    # Two misconceptions can land on the same value ("3/9" and "4/12" are both 1/3): show it once.
    seen_values = {answer}
    unique_texts = []
    for value, text_option in zip(candidates, distractor_texts):
        if value not in seen_values:
            seen_values.add(value)
            unique_texts.append(text_option)
    distractors = pick_distractors(answer_text, unique_texts)

    working = [
        f"Scale factor: {D} ÷ {d} = {k}.",
        f"Multiply the numerator by the same factor: {n} × {k} = {numerator}.",
        f"The equivalent fraction is {answer_text}.",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": numerator, "q": D},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
