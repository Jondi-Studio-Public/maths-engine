"""Rationalising a binomial surd denominator, e.g. 1/(2+sqrt(3)) -> 2-sqrt(3).

Every ``Question`` this generator returns sets ``answer_form="rationalised"``
 -- plain sympy equivalence alone would accept the un-rationalised
fraction copied straight back (sympy does not rationalise a binomial
denominator on parse; see mathgen/grading/typed.py's own comment above
``_rationalised_denominator_flagged_parsed``). CONFINED TO BINOMIAL
denominators only: a monomial denominator such as ``sqrt(2)`` is auto-
rationalised by sympy on parse (``1/sqrt(2)`` -> ``sqrt(2)/2``), so the
denominator this generator builds is always a genuine two-term
``a + s*sqrt(b)``, never a bare ``sqrt(b)`` alone.

``answer_text`` is a HAND-BUILT string, never ``str()`` of a sympy value
built by ordinary division -- sympy auto-distributes a ``Rational``
coefficient across an ``Add`` on construction (``(5 - sqrt(3)) / 11``
becomes ``5/11 - sqrt(3)/11``, not one bracketed fraction), so building the
display this way is the only way to show a single fraction
``(a - s*sqrt(b))/D``. Every ``sqrt(...)`` is spelled ASCII, never unicode
``√`` -- same round-trip reason as ``simplify-surds``: self-
grading ``answer_text`` against itself must return ``correct=True``.
"""

from __future__ import annotations

import math

from sympy import Integer, Rational, sqrt

from ..core import (
    check_distinct,
    check_equivalent,
    compose,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

# Same list as simplify-surds.
SQUAREFREE = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23]


def _draw_a_b(rng):
    """a in [2..9], b from SQUAREFREE, rejection-sampled so a*a > b -- this
    guarantees D = a*a - b is always strictly positive."""
    while True:
        a = rng.randint(2, 9)
        b = rng.choice(SQUAREFREE)
        if a * a > b:
            return a, b


def _two_term_text(const: int, coef: int, b: int) -> str:
    """'{const} + {coef}sqrt(b)' / '{const} - {|coef|}sqrt(b)', never a
    literal '+ -', and never a unit 'coef' printed as '1sqrt(b)'."""
    if coef == 0:
        return str(const)
    magnitude = abs(coef)
    sign = "+" if coef > 0 else "-"
    radical = f"sqrt({b})" if magnitude == 1 else f"{magnitude}sqrt({b})"
    return f"{const} {sign} {radical}"


def _fraction_text(numerator_text: str, denom: int) -> str:
    """Hand-built '(numerator)/denom', or the bare numerator when denom == 1
    (no fraction line needed at all)."""
    if denom == 1:
        return numerator_text
    return f"({numerator_text})/{denom}"


def _denom_text(a: int, s: int, b: int) -> str:
    """Human-facing PROMPT spelling only -- unicode '√', matching the sibling
    `t1_simplify_surds` skill's "Simplify √{n}" convention. `answer_text`
    stays ASCII `sqrt(...)` throughout (see module docstring); the grader's
    own `_normalize_surd_notation` rewrites '√' to 'sqrt(' before comparing,
    so this never changes what a learner's typed answer is graded against."""
    op = "+" if s > 0 else "-"
    return f"{a} {op} √{b}"


def _value(const: int, coef: int, b: int, denom: int):
    """sympy value of (const + coef*sqrt(b)) / denom."""
    return (Integer(const) + Integer(coef) * sqrt(b)) / Integer(denom)


@generator(
    "t1_rationalise_binomial",
    level=10,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def rationalise_binomial(rng, tier: str = "easy"):
    # Escalation across tiers (genuine reasoning, not span):
    #  easy   -- c = 1: only the conjugate-multiply-and-simplify-the-
    #            denominator step; no reduction step exists to teach yet.
    #  medium -- c in [2..6], gcd(c, D) == 1: the learner must additionally
    #            distribute c across BOTH terms of the conjugate before
    #            dividing, not just carry a 1 through.
    #  hard   -- c in [2..9], gcd(c, D) > 1: after distributing c, the
    #            learner must additionally spot and cancel the common factor
    #            between the numerator and D -- a genuinely new reasoning
    #            step, not just bigger numbers.
    for _outer_attempt in range(500):
        a, b = _draw_a_b(rng)
        s = rng.choice([1, -1])
        D = a * a - b

        if tier == "easy":
            c = 1
        elif tier == "medium":
            c = None
            for _ in range(200):
                candidate = rng.randint(2, 6)
                if math.gcd(candidate, D) == 1:
                    c = candidate
                    break
            if c is None:
                continue  # redraw a, b too
        elif tier == "hard":
            c = None
            for _ in range(200):
                candidate = rng.randint(2, 9)
                g_candidate = math.gcd(candidate, D)
                # g_candidate < D as well as > 1, so D_prime (= D // g) can
                # never collapse to 1 -- see the D_prime == 1 note below.
                if g_candidate > 1 and g_candidate < D:
                    c = candidate
                    break
            if c is None:
                continue  # redraw a, b too
        else:
            raise ValueError(f"unknown tier {tier!r}")

        g = math.gcd(c, D)
        c_prime, D_prime = c // g, D // g
        if D_prime == 1:
            # Deliberately excluded, even though the construction below supports it (a bare two-term
            # answer, no fraction at all): `wrong-denominator-formula`'s own denominator is a + b,
            # which is never 1 (a, b >= 2), so it can never be honestly displayed bare the way a
            # D_prime == 1 answer would be -- redrawing here keeps every option in the query in the
            # same "(...)/D" shape.
            continue

        wdf_denom = a + b
        if wdf_denom == D or wdf_denom == D_prime:
            # wrong-denominator-formula would coincide with the real
            # denominator; redraw the whole question rather than drop the
            # misconception.
            continue
        break
    else:
        raise AssertionError(f"could not draw a valid {tier} question (a, b, c) after 500 attempts")

    denom_text = _denom_text(a, s, b)

    # -- Correct answer: c' * (a - s*sqrt(b)) / D'. ---------------------
    numerator_const = c_prime * a
    numerator_coef = -c_prime * s
    answer_numerator_text = _two_term_text(numerator_const, numerator_coef, b)
    answer_text = _fraction_text(answer_numerator_text, D_prime)
    answer_value = _value(numerator_const, numerator_coef, b, D_prime)

    # Independent check, second route: the literal un-rationalised fraction
    # sympy builds from the question's own numbers, never touched by the
    # g/c'/D' reduction arithmetic above.
    check_equivalent(
        answer_value,
        Integer(c) / (Integer(a) + s * sqrt(b)),
        "rationalised value disagrees with the original fraction",
    )

    # -- Misconceptions, built against the UNREDUCED c, a, s, b, D. -----
    # kept-same-sign: multiplies by (a + s*sqrt(b)) again instead of the
    # conjugate (a - s*sqrt(b)) -- a very common "forgot to flip the sign"
    # error. Same denominator D.
    kss_const, kss_coef = c * a, c * s
    kept_same_sign_text = _fraction_text(_two_term_text(kss_const, kss_coef, b), D)
    kept_same_sign_value = _value(kss_const, kss_coef, b, D)

    # forgot-numerator: multiplies the denominator by the conjugate
    # correctly but leaves the original numerator untouched -- a plain
    # rational number c/D, no radical at all. Displayed in LOWEST TERMS
    # (`rational_text`, same helper the rest of the suite's global
    # `test_fractions_are_in_lowest_terms` sweep requires of every fraction-
    # shaped option): the misconception is "forgot the numerator", not "left
    # the fraction unreduced", and the hard tier's own rejection sampling
    # guarantees gcd(c, D) > 1, so the literal unreduced text would fail that
    # sweep on every hard-tier question.
    forgot_numerator_value = Integer(c) / Integer(D)
    forgot_numerator_text = rational_text(Rational(c, D))

    # wrong-denominator-formula: uses a + b in place of a*a - b, numerator
    # otherwise correct (still the unreduced c*(a - s*sqrt(b))).
    wdf_const, wdf_coef = c * a, -c * s
    wrong_denominator_formula_text = _fraction_text(_two_term_text(wdf_const, wdf_coef, b), wdf_denom)
    wrong_denominator_formula_value = _value(wdf_const, wdf_coef, b, wdf_denom)

    check_distinct(
        answer_value,
        kept_same_sign_value,
        forgot_numerator_value,
        wrong_denominator_formula_value,
        msg="rationalise-binomial-denominator distractor equals the answer",
    )

    text = (
        f"Rationalise the denominator of {c}/({denom_text}). Give your answer as a single "
        "simplified expression with a rational denominator."
    )

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_value,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, evaluate=False),
        distractors=pick_distractors(
            answer_text,
            [kept_same_sign_text, forgot_numerator_text, wrong_denominator_formula_text],
        ),
        answer_form="rationalised",
    )


# The exam-tier escalation -- a SUM of two independent binomial-denominator fractions, each
# rationalised by its own conjugate, combined into ONE simplified expression. Genuine escalation
# over the three tiers above: two independent conjugate multiplications plus a common-denominator
# combination step, not bigger numbers on the same single-fraction shape.
#
# New topic (`t1_rationalise_binomial_sum`), not a new preset on `t1_rationalise_binomial` above --
# the function signature and the whole shape of the question (two drawn fractions, not one) are
# different, so there is nothing above this comment for it to collide with.
#
# Uses `mathgen.core.compose`: two INDEPENDENT parts (each rationalises its own fraction;
# `_part_fraction_two` reads `values[0]` only to reject-sample away from a degenerate pairing with
# part 1's own draw, never to compute its OWN fraction's value from it), then `_final` combines both
# into the single answer and re-derives it a SECOND, independent way -- straight from the two
# ORIGINAL given fractions via `sympy`, never by reusing the `D1`/`D2`/`const`/`coef1`/`coef2`
# integer arithmetic the construction route used -- before returning. `compose` was chosen over a
# hand-written variant because the two parts genuinely are independent sub-computations feeding one
# combining step -- the "bundle" shape its own docstring describes, `values` simply threaded through
# unused by anything except the reject-sampling above -- and the single "Simplify 1/(...) +
# 1/(...)..." prompt is built in `_final` from both parts' raw drawn values, which fits `compose`'s
# contract (`final` receives every part's own value, never any part's rendered text) without needing
# a part-by-part working SEQUENCE to drive the prompt itself; `working` carries the two
# rationalisation steps as supplementary detail, the same split
# `exd_compose_scientific_to_engineering`'s own `_part_product`/`_part_engineering` uses.
#
# `answer_form="rationalised"` DOES generalise to this two-fraction-combined shape -- read both
# functions and confirmed this before choosing it. The two
# intermediate rationalised fractions are combined over the common denominator D1*D2 into ONE
# fraction whose denominator is always a PLAIN INTEGER (D1, D2 are each `a*a - b` for integer a, b
# -- never irrational, by the same rejection-sampled construction `_draw_a_b` already guarantees for
# the three tiers above), so the combined `answer_text` has exactly one "/" and its denominator
# operand never contains "sqrt(" -- both the parsed check (`sympy.fraction(...)` has no
# non-integer-exponent `Pow` in its denominator) and the raw-text backstop correctly pass the true
# canonical, and both correctly reject a still-not-fully-combined answer (see the
# "half-rationalised" misconception below, which exploits exactly this: it is VALUE-equal to the
# canonical -- a rationalised fraction and its un-rationalised original are the same real number --
# and is caught ONLY by this form check, never by value comparison).


def _value3(const: int, coef1: int, b1: int, coef2: int, b2: int, denom: int):
    """sympy value of (const + coef1*sqrt(b1) + coef2*sqrt(b2)) / denom."""
    return (Integer(const) + Integer(coef1) * sqrt(b1) + Integer(coef2) * sqrt(b2)) / Integer(denom)


def _three_term_text(const: int, coef1: int, b1: int, coef2: int, b2: int) -> str:
    """'{const} +/- {coef1}sqrt(b1) +/- {coef2}sqrt(b2)', joining whichever of
    the three terms are non-zero with a correct sign (never a literal
    '+ -'), the same formatting discipline `_two_term_text` uses above,
    generalised to three slots. For every value this generator itself
    constructs, const, coef1 and coef2 are all provably non-zero (a1, a2 >=
    2 and D1, D2 > 0 by `_draw_a_b`'s own rejection sampling, and coef1/
    coef2 are each +/-D2 or +/-D1) -- the zero-term-dropped branch below is
    defensive, not exercised by any live draw."""
    terms = [(const, None), (coef1, b1), (coef2, b2)]
    parts: list[str] = []
    for value, b in terms:
        if value == 0:
            continue
        if b is None:
            token = str(abs(value))
        else:
            magnitude = abs(value)
            token = f"sqrt({b})" if magnitude == 1 else f"{magnitude}sqrt({b})"
        if not parts:
            parts.append(f"-{token}" if value < 0 else token)
        else:
            parts.append(f"{'+' if value > 0 else '-'} {token}")
    return " ".join(parts) if parts else "0"


def _draw_fraction(rng, *, avoid_b=None, avoid_D=None):
    """(a, b, s, D) for one binomial-denominator fraction 1/(a +/- sqrt(b)),
    rejection-sampled so D = a*a - b is never 1 (a bare-integer rationalised
    denominator for THIS fraction alone would render its own working step as
    an ugly '/1' -- the same D_prime == 1 exclusion `rationalise_binomial`
    above already applies to its own hard tier, for the same reason) and,
    when `avoid_b`/`avoid_D` are given (the second fraction, checked against
    the first), never sharing the other fraction's own radicand (b1 == b2
    would collapse the combined numerator to two terms instead of a genuine
    three, the escalation this exam source exists to teach) or denominator
    (D1 == D2 would make the 'mismatched-denominator' misconception below
    coincide with the real answer by coincidence rather than by the
    intended mistake)."""
    for _ in range(500):
        a, b = _draw_a_b(rng)
        D = a * a - b
        if D == 1:
            continue
        if avoid_b is not None and b == avoid_b:
            continue
        if avoid_D is not None and D == avoid_D:
            continue
        s = rng.choice([1, -1])
        return a, b, s, D
    raise AssertionError("could not draw a valid rationalise-binomial-sum fraction after 500 attempts")


def _fraction_working(a: int, s: int, b: int, D: int) -> str:
    """One worked rationalisation step, learner-facing (unicode '√', same
    convention `_denom_text` uses for the prompt -- `working` is read before
    the learner answers, same as `text`, never graded, so it carries no
    round-trip constraint the way `answer_text` does)."""
    denom_text = _denom_text(a, s, b)
    conjugate_text = _denom_text(a, -s, b)
    result_text = _two_term_text(a, -s, b)
    return f"1/({denom_text}) × ({conjugate_text})/({conjugate_text}) = ({result_text})/{D}"


def _part_fraction_one(rng, values):
    """Chain-compose part 1 (`values` unused -- this is the first part): draw and rationalise the FIRST fraction.
    Returns (a, b, s, D), the raw draw, never a formatted string."""
    a, b, s, D = _draw_fraction(rng)
    return (a, b, s, D), _fraction_working(a, s, b, D)


def _part_fraction_two(rng, values):
    """Chain-compose part 2: draw and rationalise the SECOND fraction,
    reject-sampled against part 1's own (b, D) (`values[0]`) -- see
    `_draw_fraction`'s own docstring for why. `values[0]` is read only to
    steer this draw away from a degenerate pairing, never to compute this
    fraction's own value from part 1's."""
    a1, b1, s1, D1 = values[0]
    a, b, s, D = _draw_fraction(rng, avoid_b=b1, avoid_D=D1)
    return (a, b, s, D), _fraction_working(a, s, b, D)


_SUM_TEMPLATES = (
    "Simplify 1/({d1}) + 1/({d2}) by rationalising each denominator. Give your answer "
    "as a single simplified expression with a rational denominator.",
    "Write 1/({d1}) + 1/({d2}) as a single fraction with a rational denominator.",
)


def _rationalise_binomial_sum_final(values, rng):
    (a1, b1, s1, D1), (a2, b2, s2, D2) = values

    # -- Correct answer: combine the two rationalised fractions over the
    # common denominator D1*D2. ------------------------------------------
    D = D1 * D2
    const = a1 * D2 + a2 * D1
    coef1 = -s1 * D2  # coefficient of sqrt(b1)
    coef2 = -s2 * D1  # coefficient of sqrt(b2)
    answer_text = _fraction_text(_three_term_text(const, coef1, b1, coef2, b2), D)
    answer_value = _value3(const, coef1, b1, coef2, b2, D)

    # Independent check, second route: sympy's own value of the RAW sum of the two ORIGINAL given
    # fractions -- never touches D1, D2, const, coef1 or coef2 above, an independent recomputation
    # route as required.
    check_equivalent(
        answer_value,
        Integer(1) / (Integer(a1) + s1 * sqrt(b1)) + Integer(1) / (Integer(a2) + s2 * sqrt(b2)),
        "rationalise-binomial-denominator (sum): combined value disagrees with the original sum of the two fractions",
    )

    # -- Misconceptions -------------------------------------------------
    # half-rationalised: applied the conjugate to only ONE of the two
    # fractions -- fraction 2 correctly rationalised, fraction 1 left as
    # the original un-rationalised 1/(a1 +/- sqrt(b1)). VALUE-EQUAL to the
    # true answer by construction (a rationalised fraction and its
    # un-rationalised original are literally the same real number) --
    # deliberately NOT run through check_distinct below (which would
    # legitimately raise: it is not a different value). Caught only by
    # answer_form="rationalised"'s raw-text/parsed check -- asserted
    # directly by the test file: a does-not-simplify-shaped candidate is kept
    # provided it is proven caught by construction.
    frac2_rationalised_text = _fraction_text(_two_term_text(a2, -s2, b2), D2)
    half_rationalised_text = f"{frac2_rationalised_text} + 1/({_two_term_text(a1, s1, b1)})"
    half_rationalised_value = _value(a2, -s2, b2, D2) + Integer(1) / (Integer(a1) + s1 * sqrt(b1))
    check_equivalent(
        half_rationalised_value,
        answer_value,
        "half-rationalised distractor should be value-equal to the answer by "
        "construction (only the FORM differs, caught by answer_form=rationalised)",
    )

    # wrong-conjugate-sign: multiplied fraction 1's numerator and
    # denominator by (a1 + s1*sqrt(b1)) again instead of the conjugate --
    # keeps D1 (the correctly-computed denominator formula) but flips only
    # the numerator's sign, the same "forgot to flip the sign" shape
    # `kept_same_sign` uses above for the single-fraction tiers, applied to
    # just one of the two terms before combining.
    wrong_sign_coef1 = s1 * D2  # sign NOT flipped -- the error
    wrong_sign_text = _fraction_text(_three_term_text(const, wrong_sign_coef1, b1, coef2, b2), D)
    wrong_sign_value = _value3(const, wrong_sign_coef1, b1, coef2, b2, D)

    # mismatched-denominator: rationalises both fractions correctly but
    # combines them by adding the numerators over D1 ALONE -- forgot the
    # second fraction needs multiplying up to the common denominator D1*D2
    # too (the "forgot a common denominator" shape).
    mismatched_const = a1 + a2
    mismatched_coef1, mismatched_coef2 = -s1, -s2
    mismatched_text = _fraction_text(_three_term_text(mismatched_const, mismatched_coef1, b1, mismatched_coef2, b2), D1)
    mismatched_value = _value3(mismatched_const, mismatched_coef1, b1, mismatched_coef2, b2, D1)

    check_distinct(
        answer_value,
        wrong_sign_value,
        mismatched_value,
        msg="rationalise-binomial-denominator (sum) distractor equals the answer",
    )

    d1_text = _denom_text(a1, s1, b1)
    d2_text = _denom_text(a2, s2, b2)
    text = phrase(rng, _SUM_TEMPLATES, d1=d1_text, d2=d2_text)

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_value,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, evaluate=False),
        distractors=pick_distractors(
            answer_text,
            [half_rationalised_text, wrong_sign_text, mismatched_text],
        ),
        answer_form="rationalised",
    )


@generator("t1_rationalise_binomial_sum", level=10)
def rationalise_binomial_sum(rng):
    """Exam shape: sum of two independent binomial-
    denominator fractions, each rationalised by its own conjugate, combined
    into one single simplified expression. Single tier (exam only, no
    presets) -- see the module comment above `_value3` for why `compose()`
    and `answer_form="rationalised"` both fit this shape."""
    return compose(rng, [_part_fraction_one, _part_fraction_two], final=_rationalise_binomial_sum_final)
