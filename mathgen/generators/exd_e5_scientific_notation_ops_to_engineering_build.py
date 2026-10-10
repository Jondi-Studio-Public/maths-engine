"""Generates `exd_e5_scientific_notation_ops_to_engineering` questions as a chain: a scientific-notation product
or quotient, then conversion of the result to engineering notation; `final` re-derives the engineering answer
a third, independent way, straight from the two original given values, never from either chain part's already-
processed intermediate.

This module is a real shipped skill, unlike the
`exd_compose_example.py` demonstration module. This file deliberately does
NOT import from `exd_compose_example.py` (a plumbing/demo module, not a
shipped-skill precedent): the
re-normalise-to-engineering-form algorithm (`shift = exponent % 3;
eng_exponent = exponent - shift; eng_mantissa = coefficient * 10**shift`) is
public mathematics, reimplemented independently below using only
`mathgen.core` public helpers. Also does not import from
`mathgen/generators/p8a_notation_build.py` (the pinned `engineering_notation`
topic's own module) -- its `_build_engineering` is read-only house-style
precedent for the `mcq` distractor-shuffle discipline, not an import source
(its own helpers are private, underscore-prefixed).

## Grader kind -- pinned, `answer: mcq`

A value-comparing grader (`number`, `expression`) is wrong here: the
unconverted scientific-notation form (chain part 1's own rendered answer)
is the SAME real number as the correctly-converted engineering form, so any
grader that compares by VALUE would wrongly accept it. `mathgen.grading.grade_mcq`
compares the selected OPTION STRING against `answer_text` by exact string
equality (`mathgen.grading.mcq_options`, `[answer_text] + distractors`, shuffled
deterministically by seed) -- it never re-parses or re-evaluates the value,
so the unconverted form is correctly rejected as a distinct string without
needing any new `answer_form`. Matches the sibling
`skills/secondary-number/engineering-notation.yaml` (`answer: mcq`) already
shipped precedent.

## Coefficients: 4 significant figures, `Fraction(rng.randint(1000, 9999), 1000)`

NOT the demo's 1-decimal-place `[1.0, 9.9]` coefficients. Both `c1` and `c2`
are UNSIGNED magnitudes drawn this way for a MULTIPLY question. For a DIVIDE
question the divisor magnitude is instead drawn from
`_TERMINATING_DIVISOR_MAGNITUDES` -- a curated set of "5-smooth" values
(`2**x * 5**y` for integers x, y, so every value's prime factorisation has
only 2 and 5) -- because a freely-drawn divisor can make the quotient
non-terminating (e.g. dividing by a coefficient with a factor of 3), which
`exact_decimal_string` would raise on for SOME seed: `mathgen.make` must
succeed for every seed in 1..2**31, not only the ones a test sweep happens
to draw. Because every dividend coefficient here is itself `a / 1000`
(1000 = 2**3 * 5**3, so its own reduced denominator already divides 1000),
dividing by a 5-smooth divisor keeps the result's reduced denominator
5-smooth too, by the standard "product/quotient of 5-smooth denominators is
5-smooth" fact -- this is what guarantees termination for every draw, not
sampling.
decimal-places arithmetic): the normalised scientific-form coefficient needs
at most 7 decimal places to terminate exactly for a MULTIPLY question (worst
case: `9999 * 9999 / 10**6`, shifted right once during [1,10) normalisation,
needs 7 places) and at most 5 for a DIVIDE question -- `_MAX_PLACES = 7`
below covers both with margin, and every engineering-form mantissa (a
further multiply by `10**shift`, `shift` in `{0, 1, 2}`, which only ever
REDUCES the decimal-place count) stays within that same bound.

## Negatives: signs are drawn independently on hard/exam only

`c1`/`c2` are each `sign * magnitude`; `sign in {1, -1}` is drawn
independently for each value on the hard and exam tiers only (easy/medium
are always positive). The combined sign of a product OR a quotient is the
same rule (`sign1 * sign2`) -- a negative divided by a positive is negative,
two negatives multiplied or divided are positive -- so the sign is computed
once, uniformly, regardless of `op`.

## Tiers: template WORDING escalates, not just magnitude/sign

`scripts/measure_tiers.py`/`mathgen.text_quality.prompt_shape` masks every
digit run INCLUDING its leading sign (`_DIGIT_RUN_RE = r"-?\\d+(?:\\.\\d+)?"`)
and every `mathgen.slots` bank word -- so a tier that differs only in
coefficient sign, exponent magnitude, or which operation was drawn from an
otherwise-shared template bank is invisible to the instrument. The one thing
that DOES survive masking, confirmed against the demo's own two templates: a
literal `×`/`÷` character is not a digit and not a slot-bank word.
This module's design:

- **easy** -- multiplication-only template bank, its own wording, no mention
  of a sign. `op` is always "×".
- **medium** -- division-only template bank, its own wording, using
  "÷". The TOP-LEVEL CONNECTING operator between the two given values is
  always "×" on easy and always "÷" on medium -- note this is the
  connector only, not a claim that "×" is absent from medium's rendered
  text altogether: "×" is also the scientific-notation grouping symbol in
  every given value's own "c × 10^e" rendering, on every tier, regardless
  of which operator connects them. The CONNECTOR alone is still enough to
  make the two tiers' masked shapes differ (a "÷" character present in
  medium's text and wholly absent from easy's), on top of the differently-
  worded sentences.
- **hard** -- its own template bank whose wording explicitly states "One of
  the two given values may be negative"; `op` is drawn from both symbols.
- **exam** -- its own template bank with an explicit two-step instruction
  ("First combine ... then convert ...") matching an exam-style framing
  -- wording not present at any lower
  tier; `op` drawn from both symbols, negatives still possible.

Each tier's own bank carries >= 2 distinct template strings.
"""

from __future__ import annotations

import functools
from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    compose,
    exact_decimal_string,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

#: Both given values' exponents are drawn from this range (never 0 -- an
#: exponent of 0 in a scientific-notation prompt is technically valid but
#: not how this shape is normally posed). |e1|, |e2| <= 6, so the combined
#: exponent (multiply: e1+e2, divide: e1-e2) stays within the +-12 band
#: engineering notation conventionally works within.
_EXPONENTS = tuple(e for e in range(-6, 7) if e != 0)

#: The shortest terminating-decimal string within this many places covers
#: the worst case worked out in the module docstring (7 for a MULTIPLY
#: question's normalised scientific coefficient; every later multiply-by-
#: 10**shift step only ever reduces the decimal-place count).
_MAX_PLACES = 7

#: A curated divisor-magnitude set, each value `2**x * 5**y` for integers
#: x, y (a "5-smooth" rational) -- own version of the demo's
#: `_TERMINATING_DIVISORS` technique, NOT imported from it. Dividing any
#: dividend whose own reduced denominator is 5-smooth (every coefficient
#: this module draws is, since it is built as `k / 1000` and
#: 1000 = 2**3 * 5**3) by one of these values is GUARANTEED to terminate in
#: decimal, by construction -- not sampled. All lie in [1, 10), matching the
#: magnitude range every scientific-notation coefficient must satisfy.
_TERMINATING_DIVISOR_MAGNITUDES = (
    Fraction(1, 1),  # 1
    Fraction(2, 1),  # 2
    Fraction(4, 1),  # 4
    Fraction(5, 1),  # 5
    Fraction(8, 1),  # 8
    Fraction(5, 4),  # 1.25
    Fraction(8, 5),  # 1.6
    Fraction(5, 2),  # 2.5
    Fraction(25, 8),  # 3.125
    Fraction(25, 4),  # 6.25
)

_SIGNS = (1, -1)

_TIERS = ("easy", "medium", "hard", "exam")

#: Per-tier: which operation(s) this tier's questions draw, and whether a
#: negative sign is ever drawn for either given value. See the module
#: docstring's "Tiers" section for why this is the masking-safe escalation.
_TIER_OPS = {
    "easy": ("×",),
    "medium": ("÷",),
    "hard": ("×", "÷"),
    "exam": ("×", "÷"),
}
_TIER_ALLOWS_NEGATIVE = {
    "easy": False,
    "medium": False,
    "hard": True,
    "exam": True,
}

# Each tier's own template bank -- own wording per tier, not shared. Every template shares the SAME
# parseable core "(<c1> × 10^<e1>) <op> (<c2> × 10^<e2>)" so the independent oracle (this module's
# test file) can parse any tier's rendered text with one regex.
_EASY_TEMPLATES = (
    "Calculate ({c1_text} × 10^{e1}) × ({c2_text} × 10^{e2}), then write "
    "the result in engineering notation (coefficient between 1 and 1000, exponent a "
    "multiple of 3).",
    "Evaluate ({c1_text} × 10^{e1}) × ({c2_text} × 10^{e2}) and express your answer in engineering notation.",
    "({c1_text} × 10^{e1}) × ({c2_text} × 10^{e2}) -- write your answer in engineering notation.",
)
_MEDIUM_TEMPLATES = (
    "Calculate ({c1_text} × 10^{e1}) ÷ ({c2_text} × 10^{e2}), then write "
    "the result in engineering notation (coefficient between 1 and 1000, exponent a "
    "multiple of 3).",
    "Evaluate ({c1_text} × 10^{e1}) ÷ ({c2_text} × 10^{e2}) and express your answer in engineering notation.",
    "({c1_text} × 10^{e1}) ÷ ({c2_text} × 10^{e2}) -- write your answer in engineering notation.",
)
_HARD_TEMPLATES = (
    "Calculate ({c1_text} × 10^{e1}) {op} ({c2_text} × 10^{e2}). One of the two "
    "given values may be negative, so check the sign of your final answer. Write the "
    "result in engineering notation.",
    "Evaluate ({c1_text} × 10^{e1}) {op} ({c2_text} × 10^{e2}), bearing in mind "
    "that one of the two given values may be negative, and express your answer in "
    "engineering notation.",
)
_EXAM_TEMPLATES = (
    "First combine ({c1_text} × 10^{e1}) {op} ({c2_text} × 10^{e2}), then convert "
    "the result to engineering notation. One of the two given values may be negative.",
    "Step 1: work out ({c1_text} × 10^{e1}) {op} ({c2_text} × 10^{e2}). Step 2: "
    "convert that result to engineering notation. One of the two given values may be "
    "negative.",
)
_TEMPLATES = {
    "easy": _EASY_TEMPLATES,
    "medium": _MEDIUM_TEMPLATES,
    "hard": _HARD_TEMPLATES,
    "exam": _EXAM_TEMPLATES,
}


def _normalise_magnitude(magnitude: Fraction, exponent: int) -> tuple[Fraction, int]:
    """Renormalise an unsigned `magnitude * 10**exponent` so `magnitude`
    lands in [1, 10) -- the way an ordinary un-normalised product/quotient
    of two already-normalised scientific-notation magnitudes must be
    renormalised before it is itself valid scientific notation. Operates on
    the UNSIGNED magnitude only; the caller reattaches the sign."""
    while magnitude >= 10:
        magnitude /= 10
        exponent += 1
    while magnitude < 1:
        magnitude *= 10
        exponent -= 1
    return magnitude, exponent


def _part_product(rng, values, *, tier: str):
    """Chain part 1 (`values` empty -- this is the first part): draw two
    signed scientific-notation values and an operation for `tier`, compute
    the raw product/quotient, renormalise into scientific form ([1, 10) in
    magnitude). `value` is `(raw_coefficient, raw_exponent, op, e1, e2,
    c1, c2, c1_text, c2_text, tier)` -- every later part/`final` needs this,
    never a rendered string."""
    op = rng.choice(_TIER_OPS[tier])
    allow_negative = _TIER_ALLOWS_NEGATIVE[tier]
    sign1 = rng.choice(_SIGNS) if allow_negative else 1
    sign2 = rng.choice(_SIGNS) if allow_negative else 1

    e1 = rng.choice(_EXPONENTS)
    e2 = rng.choice(_EXPONENTS)
    m1 = Fraction(rng.randint(1000, 9999), 1000)  # 4 significant figures, [1.000, 9.999]
    if op == "×":
        m2 = Fraction(rng.randint(1000, 9999), 1000)
        raw_magnitude = m1 * m2
        raw_exponent = e1 + e2
    else:
        m2 = rng.choice(_TERMINATING_DIVISOR_MAGNITUDES)
        check(m2 != 0, "scientific_notation_ops_to_engineering: divisor magnitude is zero")
        raw_magnitude = m1 / m2
        raw_exponent = e1 - e2
    raw_magnitude, raw_exponent = _normalise_magnitude(raw_magnitude, raw_exponent)

    raw_sign = sign1 * sign2  # same rule for x and / -- see module docstring
    raw_coefficient = raw_sign * raw_magnitude
    c1 = sign1 * m1
    c2 = sign2 * m2

    c1_text = exact_decimal_string(c1, _MAX_PLACES)
    c2_text = exact_decimal_string(c2, _MAX_PLACES)
    result_text = exact_decimal_string(raw_coefficient, _MAX_PLACES)
    working = f"({c1_text} × 10^{e1}) {op} ({c2_text} × 10^{e2}) = {result_text} × 10^{raw_exponent}"
    value = (raw_coefficient, raw_exponent, op, e1, e2, c1, c2, c1_text, c2_text, tier)
    return value, working


def _part_engineering(rng, values):
    """Chain part 2: convert chain part 1's normalised scientific-notation
    value into engineering form (mantissa with 1 <= |mantissa| < 1000,
    exponent a multiple of 3). `values[0]` is `_part_product`'s own tuple.
    `value` is `(eng_mantissa, eng_exponent)`. Sign-independent: the shift
    is purely a function of the exponent, and multiplying the already-signed
    `raw_coefficient` by a positive `10**shift` preserves its sign."""
    raw_coefficient, raw_exponent = values[0][0], values[0][1]
    shift = raw_exponent % 3  # Python floor-mod: always in {0, 1, 2}
    eng_exponent = raw_exponent - shift
    eng_mantissa = raw_coefficient * (Fraction(10) ** shift)
    mantissa_text = exact_decimal_string(eng_mantissa, _MAX_PLACES)
    if shift == 0:
        working = (
            f"{raw_exponent} is already a multiple of 3, so the engineering form keeps "
            f"the same coefficient: {mantissa_text} × 10^{eng_exponent}."
        )
    else:
        working = (
            f"The nearest lower multiple of 3 below {raw_exponent} is {eng_exponent}, "
            f"so shift the decimal point {shift} place{'s' if shift != 1 else ''}: "
            f"{mantissa_text} × 10^{eng_exponent}."
        )
    return (eng_mantissa, eng_exponent), working


def _final(values, rng):
    (raw_coefficient, raw_exponent, op, e1, e2, c1, c2, c1_text, c2_text, tier) = values[0]
    eng_mantissa, eng_exponent = values[1]

    # Independent in-generator verification (distinct from the test file's
    # oracle): recompute the engineering mantissa a THIRD way, straight from
    # the two ORIGINAL given values c1, e1, c2, e2, op -- never from
    # raw_coefficient/raw_exponent (part 1's already-normalised intermediate)
    # or from eng_mantissa/eng_exponent (part 2's own result). A slip in
    # either chain part is caught here rather than confirmed by reuse.
    value1 = c1 * (Fraction(10) ** e1)
    value2 = c2 * (Fraction(10) ** e2)
    raw_value = value1 * value2 if op == "×" else value1 / value2
    recomputed = eng_mantissa * (Fraction(10) ** eng_exponent)
    check_equivalent(
        raw_value,
        recomputed,
        "scientific_notation_ops_to_engineering: engineering value disagrees with the original product/quotient",
    )
    check(
        Fraction(1) <= abs(eng_mantissa) < Fraction(1000),
        "scientific_notation_ops_to_engineering: mantissa outside [1, 1000) in magnitude",
    )
    check(eng_exponent % 3 == 0, "scientific_notation_ops_to_engineering: exponent not a multiple of 3")

    mantissa_text = exact_decimal_string(eng_mantissa, _MAX_PLACES)
    answer_text = f"{mantissa_text} × 10^{eng_exponent}"

    text = phrase(
        rng,
        _TEMPLATES[tier],
        c1_text=c1_text,
        e1=e1,
        c2_text=c2_text,
        e2=e2,
        op=op,
    )

    # Misconceptions (mcq distractor STRINGS, distinct text, not required to
    # differ in value -- mathgen.grading.grade_mcq grades by option identity):
    #  (a) the scientific-notation product/quotient itself, unconverted
    #      (chain part 1's own rendered form). Only offered when the
    #      scientific exponent is NOT already a multiple of 3 (shift != 0)
    #      -- when it already is, the "unconverted" form is textually
    #      IDENTICAL to the answer (a coincidence, not a genuine
    #      misconception), so it is honestly dropped for that seed rather
    #      than offered and rejected by `check(cand != answer_text, ...)`.
    #  (b) shifted one multiple of 3 in the wrong direction (both
    #      directions offered).
    #  (c) the sign dropped or flipped on the mantissa -- only exercisable
    #      once negatives exist (hard/exam), the sibling engineering-
    #      notation skill's own sign-error-on-exponent misconception, here
    #      for the mantissa's sign instead of the exponent's.
    result_text = exact_decimal_string(raw_coefficient, _MAX_PLACES)
    shift = raw_exponent % 3
    candidates = []
    if shift != 0:
        candidates.append(f"{result_text} × 10^{raw_exponent}")
    candidates.append(f"{mantissa_text} × 10^{eng_exponent + 3}")
    candidates.append(f"{mantissa_text} × 10^{eng_exponent - 3}")
    if tier in ("hard", "exam"):
        flipped_text = exact_decimal_string(-eng_mantissa, _MAX_PLACES)
        candidates.append(f"{flipped_text} × 10^{eng_exponent}")

    for cand in candidates:
        check(
            cand != answer_text,
            f"scientific_notation_ops_to_engineering: a distractor matches the answer text: {cand!r}",
        )

    # `pick_distractors` caps at 3 and takes candidates IN ORDER: with 4
    # candidates (hard/exam, shift != 0) and a fixed order, the 4th would
    # never survive and its exercisability would measure 0% forever
    # (p8a_notation_build.py's own `_build_engineering` comment, same
    # reasoning here). Shuffle so every misconception gets a genuine,
    # roughly-equal chance across seeds.
    rng.shuffle(candidates)

    return dict(
        text=text,
        latex=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "exd_e5_scientific_notation_ops_to_engineering",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e5_scientific_notation_ops_to_engineering(rng, tier: str = "medium"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    part1 = functools.partial(_part_product, tier=tier)
    return compose(rng, [part1, _part_engineering], final=_final)
