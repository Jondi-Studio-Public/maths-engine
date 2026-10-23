"""Generates `fr2b_expanded_notation` questions and the exam chain `exd_x_fr2b_expanded_notation`, for
`f10-expanded-notation-decompose` (AC9M7N03). `answer: ordered_list`.

Scope: a number written as a numeral, with its expanded notation using powers of ten, including decimals. The
learner supplies the exponents. The answer is the list of exponents and not the expansion itself, because a typed
expansion is graded by its value and "5206" would be accepted for "5 x 10^3 + 2 x 10^2 + 6". The neighbouring skill
`f10-expanded-powers-ten` goes the other way, from an expansion to a numeral.

Ladder (reasoning escalation, not size):
  easy:   a whole number with an empty place, so the exponent for each non-zero digit has to count the zeros:
          40 305 = 4 x 10^a + 3 x 10^b + 5 x 10^c.
  medium: a decimal, so the exponents after the point are negative: 5.206 = 5 x 10^a + 2 x 10^b + 6 x 10^c.
  hard:   an expansion is given, the number is multiplied or divided by a power of ten, and the learner gives the
          exponents of the new expansion without writing the new numeral first.
  exam:   a measurement with a digit on each side of the point: the exponent of each, how many places apart they
          are, and the exponent after the number is multiplied by a power of ten.

Exponents are read off the digit string by position, then re-derived from the numeral's value (each term
`digit x 10^exponent` must sum back to the numeral).
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from ._fr2b_kit import build, int_dist, list_text, tagged

_TIERS = ("easy", "medium", "hard")
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_LETTERS = "abcd"


def sup(exponent: int) -> str:
    return str(exponent).translate(_SUP)


def terms_of(numeral: str) -> list[tuple[int, int]]:
    """[(digit, exponent)] for each non-zero digit of a numeral such as "5093.07", left to right."""
    whole, _, frac = numeral.partition(".")
    out = []
    for i, c in enumerate(whole):
        if c != "0":
            out.append((int(c), len(whole) - 1 - i))
    for i, c in enumerate(frac):
        if c != "0":
            out.append((int(c), -(i + 1)))
    return out


def check_expansion(numeral: str, terms) -> None:
    """The terms digit x 10^exponent sum back to the numeral, in exact arithmetic."""
    total = sum(Fraction(d) * Fraction(10) ** e for d, e in terms)
    check(total == Fraction(numeral), "the expansion does not rebuild the numeral")


def expansion_text(terms) -> str:
    return " + ".join(f"{d} × 10{sup(e)}" for d, e in terms)


def blank_text(terms) -> str:
    return " + ".join(f"{d} × 10^{_LETTERS[i]}" for i, (d, _) in enumerate(terms))


def spaced(numeral: str) -> str:
    whole, dot, frac = numeral.partition(".")
    return f"{int(whole):,}".replace(",", " ") + dot + frac


def _skip_zero_exponents(terms) -> list[int]:
    """Exponents a learner gets by counting only the non-zero digits (skipping the zeros)."""
    out, above, below = [], len([e for _, e in terms if e >= 0]), 0
    for _, e in terms:
        if e >= 0:
            above -= 1
            out.append(above)
        else:
            below += 1
            out.append(-below)
    return out


def _list_dist(values, misconception) -> Distractor:
    return Distractor(list_text(values), misconception, list(values))


def _easy_numeral(rng) -> str:
    for _ in attempts(500, "a whole number with an inner zero"):
        length = rng.randint(4, 6)
        digits = [str(rng.randint(1, 9))] + [
            str(rng.choice((0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9))) for _ in range(length - 2)
        ]
        digits.append(str(rng.randint(1, 9)))
        numeral = "".join(digits)
        if "0" in numeral[1:-1] and 3 <= len(terms_of(numeral)) <= 4:
            return numeral
    raise AssertionError  # pragma: no cover


def _medium_numeral(rng) -> str:
    for _ in attempts(1000, "a decimal with an inner zero"):
        whole = str(rng.randint(1, 999)) if rng.random() < 0.6 else str(rng.randint(1, 9))
        frac_len = rng.randint(2, 3)
        frac = "".join(str(rng.choice((0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9))) for _ in range(frac_len - 1)) + str(
            rng.randint(1, 9)
        )
        numeral = f"{whole}.{frac}"
        t = terms_of(numeral)
        if 3 <= len(t) <= 4 and ("0" in frac[:-1] or "0" in whole[1:]):
            return numeral
    raise AssertionError  # pragma: no cover


def _exponent_question(rng, numeral, templates):
    terms = terms_of(numeral)
    check_expansion(numeral, terms)
    exps = [e for _, e in terms]
    names = ", ".join(_LETTERS[: len(terms)][:-1]) + f" and {_LETTERS[len(terms) - 1]}"
    text = phrase(rng, templates, n=spaced(numeral), blanks=blank_text(terms), names=names)
    return terms, exps, text


_EASY_TEMPLATES = [
    "Write {n} in expanded notation: {n} = {blanks}. Give {names} in that order, separated by commas.",
    "The number {n} is written in expanded notation as {blanks}. What are {names}? Give them in that order, separated by commas.",
]


def _easy(rng):
    numeral = _easy_numeral(rng)
    terms, exps, text = _exponent_question(rng, numeral, _EASY_TEMPLATES)
    cands = [
        _list_dist([e + 1 for e in exps], "fr2b-exponent-counts-digits-not-zeros"),
        _list_dist(_skip_zero_exponents(terms), "fr2b-exponent-skips-zero-digits"),
        _list_dist(list(reversed(exps)), "fr2b-exponents-listed-in-reverse-order"),
        _list_dist([e - 1 if e > 0 else e for e in exps], "fr2b-exponent-counts-digits-not-zeros"),
    ]
    working = [
        f"The digits of {spaced(numeral)} from the left are worth {', '.join(f'{d} × 10{sup(e)}' for d, e in terms)}."
    ]
    working.append("A zero digit has no term, but it still takes up a place, so the exponents skip over it.")
    return _ordered(text, exps, tagged(list_text(exps), cands), working)


def _medium(rng):
    numeral = _medium_numeral(rng)
    terms, exps, text = _exponent_question(rng, numeral, _EASY_TEMPLATES)
    cands = [
        _list_dist([abs(e) for e in exps], "fr2b-decimal-exponent-sign-dropped"),
        _list_dist(_skip_zero_exponents(terms), "fr2b-exponent-skips-zero-digits"),
        _list_dist([e + 1 if e < 0 else e for e in exps], "fr2b-first-decimal-place-counted-as-power-zero"),
        _list_dist(list(reversed(exps)), "fr2b-exponents-listed-in-reverse-order"),
    ]
    working = [
        f"Whole-number places count 0, 1, 2 upward from the point; places after the point count -1, -2, -3 downward.",
        f"{spaced(numeral)} = {expansion_text(terms)}.",
    ]
    return _ordered(text, exps, tagged(list_text(exps), cands), working)


_SCALE_WORDS = {
    1: "multiplied by 10",
    2: "multiplied by 100",
    3: "multiplied by 1 000",
    -1: "divided by 10",
    -2: "divided by 100",
    -3: "divided by 1 000",
}


def _hard(rng):
    for _ in attempts(1000, "a decimal and a scale"):
        numeral = _medium_numeral(rng) if rng.random() < 0.7 else _easy_numeral(rng)
        k = rng.choice((1, 2, 3, -1, -2, -3))
        terms = terms_of(numeral)
        if 3 <= len(terms) <= 4 and any(e > 0 for _, e in terms):
            break
    check_expansion(numeral, terms)
    exps = [e + k for _, e in terms]
    new_numeral_value = Fraction(numeral) * Fraction(10) ** k
    check(
        sum(Fraction(d) * Fraction(10) ** (e + k) for d, e in terms) == new_numeral_value,
        "the shifted terms do not rebuild the product",
    )
    names = ", ".join(_LETTERS[: len(terms)][:-1]) + f" and {_LETTERS[len(terms) - 1]}"
    text = (
        f"A number is {expansion_text(terms)}. It is {_SCALE_WORDS[k]}. The new number is {blank_text(terms)}. "
        f"Give {names} in that order, separated by commas."
    )
    digits_in_factor = abs(k) + 1
    cands = [
        _list_dist([e - k for _, e in terms], "fr2b-scaling-shifts-exponents-the-wrong-way"),
        _list_dist(
            [e + (digits_in_factor if k > 0 else -digits_in_factor) for _, e in terms],
            "fr2b-scaling-shift-counts-digits",
        ),
        _list_dist([e for _, e in terms], "fr2b-scaling-leaves-exponents-unchanged"),
        _list_dist([terms[0][1] + k] + [e for _, e in terms[1:]], "fr2b-scaling-shifts-only-the-first-term"),
    ]
    working = [
        f"Each digit moves {abs(k)} {'place' if abs(k) == 1 else 'places'} {'left' if k > 0 else 'right'}, "
        f"so each exponent {'goes up' if k > 0 else 'goes down'} by {abs(k)}.",
        f"{', '.join(str(e) for e in exps)}.",
    ]
    return _ordered(text, exps, tagged(list_text(exps), cands), working)


def _ordered(text, exps, distractors, working):
    key = list_text(exps)
    return dict(
        text=text,
        latex="",
        answer_kind="ordered_list",
        answer=list(exps),
        answer_text=key,
        answer_latex=key,
        working=working,
        distractors=distractors,
    )


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fr2b_expanded_notation", level=7, presets={tier: dict(tier=tier) for tier in _TIERS})
def fr2b_expanded_notation(rng, tier="easy"):
    """The exponents in the expanded notation of a numeral, and after a scaling; tiers in the module docstring."""
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    return build(_BUILDERS[tier], rng, tier)


# -- exam -------------------------------------------------------------------------------------------------------


def _exam_numeral(rng) -> str:
    """A measurement such as 5 093.07 or 640.8: three to five whole-number digits and one to three after the point,
    with at least one zero inside, the first and last digits non-zero."""
    for _ in attempts(1000, "an exam numeral"):
        whole = [rng.randint(1, 9)] + [rng.choice((0, 0, rng.randint(1, 9))) for _ in range(rng.randint(2, 4))]
        frac = [rng.choice((0, rng.randint(1, 9))) for _ in range(rng.randint(0, 2))] + [rng.randint(1, 9)]
        digits = whole + frac
        if 0 in digits[1:-1] and sum(1 for d in whole if d) >= 2:
            return "".join(map(str, whole)) + "." + "".join(map(str, frac))
    raise AssertionError  # pragma: no cover


@generator("exd_x_fr2b_expanded_notation", level=7, presets={"exam": dict(tier="exam")})
def exd_x_fr2b_expanded_notation(rng, tier="exam"):
    """Exam chain: the exponent of a digit before the point, of one after it, how far apart they are, and after scaling."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "two digits that each appear once"):
        numeral = _exam_numeral(rng)
        terms = terms_of(numeral)
        # A digit before the point (not the units digit) and the last digit after it, each as a term of the expansion.
        (d1, e1) = rng.choice([t for t in terms if t[1] >= 1])
        (d2, e2) = terms[-1]
        if numeral.count(str(d1)) == 1 and numeral.count(str(d2)) == 1:
            break
    check_expansion(numeral, terms)
    check(e2 < 0 < e1, "the two digits must be on opposite sides of the point")
    unit, quantity = rng.choice(
        (
            ("g", "A scientist weighs a sample at"),
            ("mL", "A lab measures a volume of"),
            ("m", "A surveyor records a distance of"),
        )
    )
    k = rng.choice((2, 3, -1, -2))
    verb, scale = {2: ("multiplied", "100"), 3: ("multiplied", "1 000"), -1: ("divided", "10"), -2: ("divided", "100")}[
        k
    ]
    skip = _skip_zero_exponents(terms)
    idx1, idx2 = terms.index((d1, e1)), len(terms) - 1
    check(e1 + k != e1 and k != 0, "the scale must move the digit")
    digits_between = e1 - e2 - 1
    stem = f"{quantity} {spaced(numeral)} {unit}."
    p1 = f"The digit {d1} is written {d1} × 10^n in the expanded notation of {spaced(numeral)}. Find n."
    p2 = f"The last digit, {d2}, is written {d2} × 10^m. Find m."
    p3 = f"How many places apart are the digit {d1} and the digit {d2}?"
    p4 = (
        f"The measurement is {verb} by {scale}. In the expanded notation of the new number the digit {d1} is "
        f"written {d1} × 10^p. Find p."
    )
    a_cands = [
        int_dist(e1 + 1, "fr2b-exponent-counts-digits-not-zeros"),
        int_dist(skip[idx1], "fr2b-exponent-skips-zero-digits"),
        int_dist(10**e1, "fr2b-exponent-given-as-the-place-value"),
        int_dist(e1 - 1, "fr2b-exponent-off-by-one"),
    ]
    b_cands = [
        int_dist(abs(e2), "fr2b-decimal-exponent-sign-dropped"),
        int_dist(e2 + 1, "fr2b-first-decimal-place-counted-as-power-zero"),
        int_dist(skip[idx2], "fr2b-exponent-skips-zero-digits"),
        int_dist(e2 - 1, "fr2b-exponent-off-by-one"),
    ]
    c_cands = [
        int_dist(e1 + e2, "fr2b-places-apart-adds-the-exponents"),
        int_dist(digits_between, "fr2b-places-apart-counts-digits-between"),
        int_dist(e1 - e2 + 1, "fr2b-places-apart-counts-both-ends"),
        int_dist(e1 - abs(e2), "fr2b-places-apart-adds-the-exponents"),
    ]
    d_cands = [
        int_dist(e1 - k, "fr2b-scaling-shifts-exponents-the-wrong-way"),
        int_dist(e1, "fr2b-scaling-leaves-exponents-unchanged"),
        int_dist(e1 * k, "fr2b-scaling-multiplies-the-exponent"),
        int_dist(e1 + (abs(k) + 1) * (1 if k > 0 else -1), "fr2b-scaling-shift-counts-digits"),
    ]
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p1,
                answer_kind="number",
                answer=e1,
                answer_text=str(e1),
                marks=1,
                working=[
                    f"Counting from the units digit (exponent 0), the {d1} is {e1} {'place' if e1 == 1 else 'places'} to the left, so n = {e1}."
                ],
                distractors=tagged(str(e1), a_cands),
            ),
            dict(
                prompt=p2,
                answer_kind="number",
                answer=e2,
                answer_text=str(e2),
                marks=2,
                working=[f"Places after the point count -1, -2, so the {d2} has exponent {e2}."],
                distractors=tagged(str(e2), b_cands),
            ),
            dict(
                prompt=p3,
                answer_kind="number",
                answer=e1 - e2,
                answer_text=str(e1 - e2),
                marks=1,
                follow_through="part_a - part_b",
                sensible_range=(0, None),
                working=[
                    f"The exponents differ by {e1} - ({e2}) = {e1 - e2}, so the digits are {e1 - e2} {'place' if e1 - e2 == 1 else 'places'} apart."
                ],
                distractors=tagged(str(e1 - e2), c_cands),
            ),
            dict(
                prompt=p4,
                answer_kind="number",
                answer=e1 + k,
                answer_text=str(e1 + k),
                marks=2,
                follow_through=f"part_a {'+' if k > 0 else '-'} {abs(k)}",
                working=[
                    f"{verb.capitalize()} by {scale} moves each digit {abs(k)} {'place' if abs(k) == 1 else 'places'} {'left' if k > 0 else 'right'}, so p = {e1} {'+' if k > 0 else '-'} {abs(k)} = {e1 + k}."
                ],
                distractors=tagged(str(e1 + k), d_cands),
            ),
        ],
    )
