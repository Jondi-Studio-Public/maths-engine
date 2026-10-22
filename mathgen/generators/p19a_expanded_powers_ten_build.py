"""Generates `p19a_expanded_powers_ten` questions for `f10-expanded-powers-ten` (AC9M7N03): reconstruct a whole
number from its expanded place-value form using powers of ten.

Scope: "Given a natural number written in expanded place-value
notation using powers of ten (for example 4*10^3 + 3*10^2 + 5*10 + 6),
reconstruct the single numeral it represents."

## Answer shape

`answer: number`, a plain whole number, graded EXACTLY (`answer_text` carries
no decimal point). `_strip_comma_groups` means "4,356" is accepted alongside
"4356"; a space-grouped "4 356" is rejected because `extract_values` reads
the space as a SEPARATE number and neither "4" nor "356" equals the target
4356 (see the build test's direct probe against
`mathgen.grading.GRADERS["number"]`). The prompt therefore always states "write
the number without spaces."

## The four tiers (genuine reasoning escalation, never a bigger number)

- **easy**: every place is present (no missing powers), digits drawn 1-9, and
  the terms are always listed in descending power order. DECISION: read off
  each coefficient against its own power and place it directly -- no place
  is missing, no reordering needed.
- **medium**: some internal digit is exactly 0, so its term is OMITTED from
  the expansion entirely -- the learner must recognise a MISSING POWER means
  a 0 digit belongs there, a decision easy never raises (every place is
  always present at easy).
- **hard**: the terms are always listed OUT OF ORDER (never descending), and
  sometimes the lowest two powers are written explicitly (`10^1`, `10^0`
  rather than bare `10` / a bare digit) -- the learner must sort by power
  before placing any digit, a decision neither easy nor medium requires
  (both are always already in descending order).
- **exam**: the expansion is NON-STANDARD -- one place's stated coefficient
  is 10 or more (e.g. `3 x 10^3 + 14 x 10^2 + 6 x 10`) -- so the learner must
  regroup the oversized coefficient into the place above it before the
  numeral can be written at all, a step no lower tier ever requires (every
  lower-tier coefficient is a single digit, 0-9).

## Independent check

`_check_reconstruct` sums `coeff * 10**power` over the DISPLAYED terms (the
learner's own route) and confirms it reconstructs the target numeral built
from placing DIGITS directly (`d3*1000 + d2*100 + d1*10 + d0`) -- construction
and check never share an arithmetic step. `check_pairwise_distinct` runs over
the plain integer values of the answer and all three distractors.
"""

from __future__ import annotations

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    phrase,
)

TOPIC = "p19a_expanded_powers_ten"
_TIERS = ("easy", "medium", "hard", "exam")
_RETRY_LIMIT = 500

_FORMAT_NOTE = " Write the number without spaces."

_EASY_PATTERNS = (
    "What number is {expr}?" + _FORMAT_NOTE,
    "Write {expr} as a single number." + _FORMAT_NOTE,
    "{expr} is the expanded form of what number?" + _FORMAT_NOTE,
)
_MEDIUM_PATTERNS = (
    "What number is {expr}?" + _FORMAT_NOTE,
    "Write {expr} as a single number." + _FORMAT_NOTE,
    "{expr} is the expanded form of what number? (Some places are missing.)" + _FORMAT_NOTE,
)
_HARD_PATTERNS = (
    "The terms below are not in order. What number do they add up to: {expr}?" + _FORMAT_NOTE,
    "Sort the powers of ten below, then write the number they represent: {expr}." + _FORMAT_NOTE,
    "{expr} is the expanded form of what number? (The terms are listed out of order.)" + _FORMAT_NOTE,
)
_EXAM_PATTERNS = (
    "{expr} is a non-standard expanded form: one coefficient is 10 or more. "
    "Regroup it and write the number it represents." + _FORMAT_NOTE,
    "What number is {expr}? (One term needs to be regrouped before you can write the numeral.)" + _FORMAT_NOTE,
    "Regroup {expr} into a single number." + _FORMAT_NOTE,
)


def _term_text(coeff: int, power: int, explicit_low: bool = False) -> str:
    if power == 0:
        return f"{coeff} × 10^0" if explicit_low else f"{coeff}"
    if power == 1:
        return f"{coeff} × 10^1" if explicit_low else f"{coeff} × 10"
    return f"{coeff} × 10^{power}"


def _expr_text(terms: list[tuple[int, int]], explicit_low: bool = False) -> str:
    return " + ".join(_term_text(c, p, explicit_low) for c, p in terms)


def _reconstruct(terms: list[tuple[int, int]]) -> int:
    return sum(c * 10**p for c, p in terms)


def _check_reconstruct(terms: list[tuple[int, int]], N: int) -> None:
    """Independent route: sum coeff*10**power over the DISPLAYED terms and
    confirm it reconstructs N, which was built by placing digits directly."""
    check(_reconstruct(terms) == N, "p19a_expanded_powers_ten: reconstruction mismatch")


def _concat_distractor(terms: list[tuple[int, int]]) -> int:
    """Misconception: concatenates the coefficients as if they were adjacent
    digits, ignoring which powers are actually present (and, at exam,
    ignoring that an oversized coefficient needs regrouping first)."""
    return int("".join(str(c) for c, _ in terms))


def _wrong_place_distractor(terms: list[tuple[int, int]]) -> int:
    """Misconception: swaps the two lowest-power present terms' coefficients
    -- a digit lands in the wrong place."""
    if len(terms) < 2:
        return _reconstruct(terms) + 1
    swapped = list(terms)
    (c1, p1), (c2, p2) = swapped[-2], swapped[-1]
    swapped[-2], swapped[-1] = (c2, p1), (c1, p2)
    return _reconstruct(swapped)


def _exponent_off_by_one_distractor(terms: list[tuple[int, int]]) -> int:
    """Misconception: reads the leading term's power of ten as one fewer
    zero than it actually has."""
    c_lead, p_lead = terms[0]
    if p_lead >= 1:
        return _reconstruct(terms) - c_lead * 10**p_lead + c_lead * 10 ** (p_lead - 1)
    return _reconstruct(terms) + c_lead * 9


def _dropped_term_distractor(terms: list[tuple[int, int]]) -> int:
    """Misconception: forgets the last (lowest-power) term entirely -- used
    in place of the concatenation misconception at the easy tier, where
    every place is present and in order, so concatenating the coefficients
    would trivially reproduce the correct numeral rather than a genuine
    wrong answer."""
    c_last, p_last = terms[-1]
    return _reconstruct(terms) - c_last * 10**p_last


def _distractor_candidates(terms: list[tuple[int, int]], use_concat: bool) -> list[int]:
    candidates = [_wrong_place_distractor(terms), _exponent_off_by_one_distractor(terms)]
    if use_concat:
        candidates.insert(0, _concat_distractor(terms))
    else:
        candidates.insert(0, _dropped_term_distractor(terms))
    return candidates


def _finalise(text: str, N: int, terms: list[tuple[int, int]], candidates: list[int]) -> dict:
    check_pairwise_distinct(
        N, *candidates, msg="p19a_expanded_powers_ten: a distractor collides with the answer or another option"
    )
    seen = {str(N)}
    distractors = []
    for cand in candidates:
        s = str(cand)
        if s in seen:
            continue
        seen.add(s)
        distractors.append(s)
    check(len(distractors) >= 2, "p19a_expanded_powers_ten: not enough surviving distractors")
    return dict(
        text=text,
        answer=N,
        answer_text=str(N),
        distractors=distractors,
        working=[f"{' + '.join(f'{c} × 10^{p}' for c, p in terms)} = {N}."],
    )


def _draw_digits(rng, digit_count: int, allow_internal_zero: bool):
    """`digit_count` digits, most significant first, most-significant digit
    always nonzero. If `allow_internal_zero` is False every digit is drawn
    1-9 (no missing powers); if True at least one non-leading digit is
    forced to 0."""
    if not allow_internal_zero:
        return [rng.randint(1, 9) for _ in range(digit_count)]
    digits = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(digit_count - 1)]
    if all(d != 0 for d in digits[1:]):
        zero_pos = rng.randint(1, digit_count - 1)
        digits[zero_pos] = 0
    return digits


def _terms_from_digits(digits: list[int]) -> list[tuple[int, int]]:
    n = len(digits)
    return [(d, n - 1 - i) for i, d in enumerate(digits) if d != 0]


# --- easy: every place present, descending order --------------------------


def _build_easy(rng):
    for _ in range(_RETRY_LIMIT):
        digit_count = rng.choice((3, 4))
        digits = _draw_digits(rng, digit_count, allow_internal_zero=False)
        terms = _terms_from_digits(digits)
        N = int("".join(str(d) for d in digits))
        _check_reconstruct(terms, N)
        candidates = _distractor_candidates(terms, use_concat=False)
        try:
            text = phrase(rng, _EASY_PATTERNS, expr=_expr_text(terms))
            return _finalise(text, N, terms, candidates)
        except Exception:
            continue
    check(False, "p19a_expanded_powers_ten easy: could not draw distinct distractors")


# --- medium: a missing power, three- to five-digit result -----------------


def _build_medium(rng):
    for _ in range(_RETRY_LIMIT):
        digit_count = rng.choice((3, 4, 5))
        digits = _draw_digits(rng, digit_count, allow_internal_zero=True)
        terms = _terms_from_digits(digits)
        N = int("".join(str(d) for d in digits))
        _check_reconstruct(terms, N)
        candidates = _distractor_candidates(terms, use_concat=True)
        try:
            text = phrase(rng, _MEDIUM_PATTERNS, expr=_expr_text(terms))
            return _finalise(text, N, terms, candidates)
        except Exception:
            continue
    check(False, "p19a_expanded_powers_ten medium: could not draw distinct distractors")


# --- hard: shuffled order, sometimes explicit 10^1/10^0 --------------------


def _build_hard(rng):
    for _ in range(_RETRY_LIMIT):
        digit_count = rng.choice((3, 4, 5))
        allow_zero = rng.random() < 0.5
        digits = _draw_digits(rng, digit_count, allow_internal_zero=allow_zero)
        terms = _terms_from_digits(digits)
        N = int("".join(str(d) for d in digits))
        _check_reconstruct(terms, N)
        explicit_low = rng.random() < 0.5
        display_terms = list(terms)
        rng.shuffle(display_terms)
        # Guard against a shuffle landing on the already-descending order by
        # construction alone -- with >= 3 terms this is astronomically
        # unlikely, but a hard tier degenerating to easy's ordering is
        # exactly the defect this skill exists to avoid, so force one
        # adjacent swap if it happens.
        if display_terms == terms and len(display_terms) >= 2:
            display_terms[0], display_terms[1] = display_terms[1], display_terms[0]
        candidates = _distractor_candidates(display_terms, use_concat=True)
        try:
            text = phrase(rng, _HARD_PATTERNS, expr=_expr_text(display_terms, explicit_low))
            return _finalise(text, N, display_terms, candidates)
        except Exception:
            continue
    check(False, "p19a_expanded_powers_ten hard: could not draw distinct distractors")


# --- exam: a non-standard (>= 10) coefficient needs regrouping first -------


def _build_exam(rng):
    for _ in range(_RETRY_LIMIT):
        d3 = rng.randint(2, 9)
        d2 = rng.randint(0, 9)
        d1 = rng.randint(0, 9)
        d0 = rng.randint(0, 9)
        digits = {3: d3, 2: d2, 1: d1, 0: d0}
        N = d3 * 1000 + d2 * 100 + d1 * 10 + d0

        boundary = rng.choice(("3-2", "2-1", "1-0"))
        hi, lo = (int(boundary[0]), int(boundary[2]))
        min_hi = 1 if hi == 3 else 0  # the leading digit (hi == 3) must stay nonzero
        if digits[hi] - 1 < min_hi:
            continue
        disp = dict(digits)
        disp[hi] = digits[hi] - 1
        disp[lo] = digits[lo] + 10

        terms = [(disp[p], p) for p in (3, 2, 1, 0) if disp[p] != 0]
        if len(terms) < 2:
            continue
        check_equivalent(_reconstruct(terms), N, "p19a_expanded_powers_ten exam: regroup mismatch")
        candidates = _distractor_candidates(terms, use_concat=True)
        try:
            text = phrase(rng, _EXAM_PATTERNS, expr=_expr_text(terms))
            return _finalise(text, N, terms, candidates)
        except Exception:
            continue
    check(False, "p19a_expanded_powers_ten exam: could not draw a valid case")


_BUILDERS = {
    "easy": _build_easy,
    "medium": _build_medium,
    "hard": _build_hard,
    "exam": _build_exam,
}


@generator(
    TOPIC,
    level=5,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19a_expanded_powers_ten(rng, tier: str = "easy"):
    """Reconstruct the single numeral a natural number's expanded
    place-value notation (using powers of ten) represents. `tier` selects
    one of the four genuinely different decisions (see module docstring)."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {_TIERS}")
    return _BUILDERS[tier](rng)
