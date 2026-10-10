"""Extends the LIVE `scientific-notation` skill (`skills/secondary-number/scientific-notation.yaml`, `answer:
mcq`) with the four arithmetic operations on two numbers already given in scientific notation --
`scientific_notation_operate`. Deliberately its own module: it never imports from, and is never imported by,
`curriculum_expansion.py`'s existing `scientific_notation` (conversion-only) topic, so nothing pinned there
can move.

Every value in this module is an exact `fractions.Fraction` until the very
last step (formatting for display), so "is this the same number" is decided
by exact equality, never a float tolerance. The one place a float genuinely
appears is the independent verification route: a
*completely different* computation (`coefficient * 10 ** exponent` in plain
float arithmetic on the ORIGINAL two operands) is compared against the
Fraction-derived renormalised answer with `math.isclose`.

**Design decisions, forced by real contradictions in a literal reading of the requirements:**

1. The requirements are "the un-renormalised form is always a distractor" AND
   "all four [options] are pairwise distinct beyond a safe relative
   tolerance". Renormalisation is a value-preserving rescale (`19.8 x 10^4`
   IS `1.98 x 10^5`, not merely close to it), so the un-renormalised
   distractor is *always* value-equal to the answer by construction -- the
   two requirements cannot both hold under a literal "all four pairwise
   distinct by value" reading. This module reads "distinct" as: distinct as
   a displayed (coefficient, exponent) PAIR (matching how the real mcq
   grader and `pick_distractors`' own string-based dedupe treat options, and
   how an independent oracle would treat it -- asserting the un-renormalised
   distractor is numerically equal to the TRUE result, i.e. equal, not merely
   distinct-but-close); the un-
   renormalised distractor is asserted EQUAL in value to the answer (that is
   the whole point of it), and every OTHER pair (answer/D2, answer/D3,
   D1/D2, D1/D3, D2/D3) is asserted properly value-distinct.
2. A direct consequence: whenever the raw (pre-renormalisation) result is
   already in [1, 10), renormalisation is a no-op and the un-renormalised
   "distractor" would be pair-IDENTICAL to the answer (not just value-equal
   -- the same displayed string), which genuinely is an unfair/duplicate
   option. Every emitted question is therefore one where renormalisation was
   actually needed (100% by construction).
3. For add/subtract, drawing both exponents independently from the
   full 7-value pool has two problems: (a) renormalisation is only
   even POSSIBLE when the two exponents are exactly 1 apart (see
   `_renormalise`'s docstring for why), so an unrestricted draw would need
   50-100 reject-and-redraw cycles per accepted question under filter (2)
   above -- a real cost against `check_generator_speed.py`'s budget; (b) a
   gap of 6+ (crossing the negative/positive exponent clusters) makes the
   smaller operand invisible in the displayed result, which is a
   construction defect ("a subject teacher would
   read as nonsense") independently of the performance concern. This module
   draws the exponent PAIR from five fixed adjacent pairs instead
   (`_ADJACENT_EXPONENT_PAIRS`) and biases the larger-exponent operand's
   coefficient so an accepting draw is found in a small number of attempts.
   Multiply/divide keep the full independent-draw pool
   unchanged -- neither problem applies to them (renormalisation is common
   under an unbiased draw, and there is no "invisible operand" concern for a product/quotient).
4. `float(coeff):g` can round a coefficient just under 10 UP to the display
   string "10" (e.g. 9.9999996 -> "10"), which would be an invalid
   scientific-notation coefficient despite `coeff` itself being an exact
   Fraction in [1, 10). `_display_in_range` re-parses the DISPLAYED string
   and rejects-and-redraws any draw whose answer coefficient would render
   this way.
"""

import math
from fractions import Fraction

from ..core import check, check_distinct, generator, pick_distractors

_EXPONENTS = (-5, -4, -3, 3, 4, 5, 6)

# add/subtract only -- see module docstring, deviation 3. Each pair is
# (smaller exponent, larger exponent); renormalisation for +/- can only ever
# happen when the gap is exactly 1 (see `_renormalise`'s docstring).
_ADJACENT_EXPONENT_PAIRS = ((-5, -4), (-4, -3), (3, 4), (4, 5), (5, 6))

_OP_SYMBOL = {"add": "+", "subtract": "-", "multiply": "×", "divide": "÷"}


def _draw_coefficient(rng, lo=10, hi=99):
    """A coefficient in [1, 10), one decimal place, same draw shape as the
    existing `scientific_notation` topic's `rng.randint(10, 99) / 10` --
    kept exact here as a Fraction instead of a float."""
    return Fraction(rng.randint(lo, hi), 10)


def _renormalise(coeff, exponent):
    """Shift `coeff` into [1, 10) by exact powers of ten, adjusting
    `exponent` to match -- exact Fraction arithmetic throughout, so this is a
    value-preserving rescale, never a numeric approximation.

    For add/subtract at a common exponent E built from two operands each
    already in [1, 10): the rescaled smaller-exponent operand contributes
    less than 1 whenever the exponent gap is >= 2 (`c * 10**-gap < 10 *
    10**-2 = 0.1` for gap 2, shrinking further for bigger gaps), so the sum
    can only ever reach 10 (needing a shift here) when the gap is exactly 1.
    """
    while coeff >= 10:
        coeff, exponent = coeff / 10, exponent + 1
    while coeff < 1:
        coeff, exponent = coeff * 10, exponent - 1
    return coeff, exponent


def _fmt(coeff, exponent):
    """The one option-format string every option (answer and every
    distractor) uses, matching the existing `scientific_notation` topic's own
    `f"{coefficient:g} x 10^{exponent}"` exactly."""
    return f"{float(coeff):g} × 10^{exponent}"


def _display_in_range(coeff):
    """True iff the *displayed* (`:g`-rounded) coefficient still reads in
    [1, 10) -- guards against e.g. 9.9999996 rounding up to the string "10",
    which would show as an invalid scientific-notation coefficient even
    though the exact Fraction `coeff` itself is in range."""
    shown = Fraction(f"{float(coeff):g}")
    return 1 <= shown < 10


def _value(coeff, exponent):
    """The exact value `coeff * 10**exponent` a (coefficient, exponent) pair
    represents, as a Fraction."""
    return coeff * Fraction(10) ** exponent


def _pairwise_ok(answer_pair, d1, d2, d3):
    """Cheap, exact-Fraction hygiene checks run inside the reject-and-redraw
    loop (never sympy -- see module docstring for why): reject a draw unless
    all four (coefficient, exponent) option pairs are distinct AND every pair
    of option VALUES is distinct except (answer, d1), which is required to be
    value-EQUAL (the un-renormalised form) -- see module docstring deviation
    1. Returns True to accept the draw, False to redraw."""
    pairs = (answer_pair, d1, d2, d3)
    if len({tuple(p) for p in pairs}) != 4:
        return False
    values = [_value(*p) for p in pairs]
    if values[0] != values[1]:
        return False
    if len({values[0], values[2], values[3]}) != 3:
        return False
    return True


def _verify(answer_pair, d1, d2, d3, true_value_float):
    """The two named in-generator checks (`mathgen.core`), run ONCE on a
    draw that has already passed `_pairwise_ok` -- confirms via the
    sympy-backed helpers what the cheap Fraction pre-filter already
    established, and is the authoritative independent-route check for the
    answer itself."""
    answer_coeff, answer_exp = answer_pair
    values = [_value(answer_coeff, answer_exp), _value(*d1), _value(*d2), _value(*d3)]
    check(values[0] == values[1], "un-renormalised distractor must equal the true value")
    check_distinct(values[0], values[2], values[3])
    check_distinct(values[1], values[2], values[3])
    check_distinct(values[2], values[3])
    candidate_value = float(answer_coeff) * 10.0**answer_exp
    check(
        math.isclose(candidate_value, true_value_float, rel_tol=1e-9),
        "renormalised answer disagrees with the independent float route",
    )


def _build_add_subtract(rng, op):
    while True:
        e_small, e_big = rng.choice(_ADJACENT_EXPONENT_PAIRS)
        # Biased so an accepting (renormalisation-needed) draw is found in a
        # handful of attempts -- see module docstring deviation 3. Add needs
        # c_big + c_small/10 >= 10, so c_big must sit near the top of [1,10);
        # subtract needs c_big - c_small/10 < 1, so c_big must sit near the
        # bottom.
        if op == "add":
            c_big = _draw_coefficient(rng, 95, 99)
        else:
            c_big = _draw_coefficient(rng, 10, 14)
        c_small = _draw_coefficient(rng)

        # Question order is always (larger-exponent operand) then
        # (smaller-exponent operand): rescaling the smaller-exponent operand
        # down to the common exponent e_big always gives a value in (0, 1)
        # (e_small is always e_big - 1, so the rescale is a single division
        # by 10), so `raw` is always positive for subtract without needing
        # its own reject-on-sign check.
        rescaled_small = c_small / 10
        raw = c_big + rescaled_small if op == "add" else c_big - rescaled_small
        answer_coeff, answer_exp = _renormalise(raw, e_big)
        if answer_exp == e_big:
            continue  # no renormalisation happened -- see deviation 2
        if not _display_in_range(answer_coeff):
            continue

        d1 = (raw, e_big)  # un-renormalised sum/difference
        d2 = (answer_coeff, e_small + e_big)  # correct coefficient, added exponents (wrong rule for +/-)
        d3 = (raw, e_small)  # correct coefficient, wrong (smaller) exponent

        if not _pairwise_ok((answer_coeff, answer_exp), d1, d2, d3):
            continue

        ordinary_big = float(c_big) * 10.0**e_big
        ordinary_small = float(c_small) * 10.0**e_small
        true_value = ordinary_big + ordinary_small if op == "add" else ordinary_big - ordinary_small
        _verify((answer_coeff, answer_exp), d1, d2, d3, true_value)

        symbol = _OP_SYMBOL[op]
        text = (
            f"Calculate ({_fmt(c_big, e_big)}) {symbol} ({_fmt(c_small, e_small)}). "
            f"Give your answer in scientific notation."
        )
        answer = _fmt(answer_coeff, answer_exp)
        distractors = pick_distractors(answer, [_fmt(*d1), _fmt(*d2), _fmt(*d3)])
        if len(distractors) != 3:
            continue
        verb = "adding" if op == "add" else "subtracting"
        working = [
            f"Rewrite both terms with exponent {e_big}, then re-normalise after {verb} the coefficients: {answer}."
        ]
        return dict(
            text=text, answer=answer, answer_text=answer, answer_latex=answer, working=working, distractors=distractors
        )


def _build_multiply_divide(rng, op):
    while True:
        c1, e1 = _draw_coefficient(rng), rng.choice(_EXPONENTS)
        c2, e2 = _draw_coefficient(rng), rng.choice(_EXPONENTS)
        if op == "multiply":
            raw, raw_exp = c1 * c2, e1 + e2
        else:
            raw, raw_exp = c1 / c2, e1 - e2
        answer_coeff, answer_exp = _renormalise(raw, raw_exp)
        if answer_exp == raw_exp:
            continue  # already in [1, 10): see deviation 2
        if not _display_in_range(answer_coeff):
            continue

        d1 = (raw, raw_exp)  # un-renormalised product/quotient
        if op == "multiply":
            d2 = (c1 + c2, e1 + e2)  # added coefficients, but (correctly) added exponents
            d3 = (answer_coeff, e1 - e2)  # correct coefficient, subtracted instead of added
        else:
            d2 = (answer_coeff, e1 + e2)  # correct coefficient, added instead of subtracted
            d3 = (answer_coeff, e2 - e1)  # correct coefficient, subtracted the wrong way round

        if not _pairwise_ok((answer_coeff, answer_exp), d1, d2, d3):
            continue

        ordinary1 = float(c1) * 10.0**e1
        ordinary2 = float(c2) * 10.0**e2
        true_value = ordinary1 * ordinary2 if op == "multiply" else ordinary1 / ordinary2
        _verify((answer_coeff, answer_exp), d1, d2, d3, true_value)

        symbol = _OP_SYMBOL[op]
        text = f"Calculate ({_fmt(c1, e1)}) {symbol} ({_fmt(c2, e2)}). Give your answer in scientific notation."
        answer = _fmt(answer_coeff, answer_exp)
        distractors = pick_distractors(answer, [_fmt(*d1), _fmt(*d2), _fmt(*d3)])
        if len(distractors) != 3:
            continue
        rule = "add" if op == "multiply" else "subtract"
        verb = "Multiply" if op == "multiply" else "Divide"
        working = [f"{verb} the coefficients and {rule} the exponents, then re-normalise: {answer}."]
        return dict(
            text=text, answer=answer, answer_text=answer, answer_latex=answer, working=working, distractors=distractors
        )


@generator("scientific_notation_operate", level=9)
def scientific_notation_operate(rng, ops=("add", "subtract", "multiply", "divide")):
    op = rng.choice(list(ops))
    if op in ("add", "subtract"):
        return _build_add_subtract(rng, op)
    return _build_multiply_divide(rng, op)
