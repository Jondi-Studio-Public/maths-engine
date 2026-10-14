"""Independent oracle for the `p13a_direct_proportion_scale` questions: it re-derives each answer from the
rendered question text.

Parses the RENDERED question text and mcq options only
(`question.text`/`mathgen.grading.mcq_options`/`question.answer_text`) and
imports NOTHING from
`mathgen.generators.p13a_direct_proportion_scale_build`. Every numeric route
below is re-derived from the stated text with `Fraction` arithmetic, never by
reusing the generator's own construction values, and the classification is
independently re-derived from what the text itself states -- a fixed-fee
clause present in the text means NOT proportional, absent means proportional
(for exam, whether the two stated data points' implied per-unit rates agree).

The skill's whole grader kind changed from `number` to `mcq` -- every tier's mcq now bundles a classification
label with a computed value into ONE option string, and the oracle below re-derives BOTH independently, then
asserts the option encoding both matches `question.answer_text` (the keyed-correct option), exactly the way
`tests/test_p10d_similarity_scale_factor_oracle.py` hardcodes its own `OPTION_*` literals rather than
importing them -- these two label strings and the two option-format templates are part of the skill's public
contract, copied here as literals.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills
from mathgen import make

SKILL = "f10-direct-proportion-scale-model"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS_SMALL = range(60)
SEEDS_1000 = range(1000)

# -- Public-contract literals (copied, never imported) -----------------------
_LABEL_YES = "Yes, directly proportional"
_LABEL_NO = "No, not directly proportional (a fixed fee applies)"


def _money_opt(label: str, money_text: str) -> str:
    return f"{label} -- the total cost is ${money_text}."


def _qty_opt(label: str, qty_text: str, unit_plural: str) -> str:
    return f"{label} -- you can buy {qty_text} {unit_plural}."


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL]
    if difficulty == "exam":
        # The skill's exam tier is now the multi-part chain `exd_x_r2x_direct_proportion_table`; this module
        # tests this topic's own single-step exam preset, which the generator still provides.
        return make("p13a_direct_proportion_scale", seed=seed, difficulty="exam")
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# =============================================================================
# Easy: pure_rate only (disclosed constant). "{Item} costs $rate per unit ...
# Is this relationship directly proportional, and what is the total cost for
# N units?"
# =============================================================================

_RATE_RE = re.compile(r"\$(\d+\.\d{2}) per \w+")
_QTY_FOR_RE = re.compile(r"for (\d+) [a-zA-Z]+")


def _easy_oracle_value(text: str) -> Fraction:
    rate = Fraction(_RATE_RE.search(text).group(1))
    quantity = int(_QTY_FOR_RE.search(text).group(1))
    return rate * quantity


def _easy_oracle_key(text: str) -> str:
    value = _easy_oracle_value(text)
    return _money_opt(_LABEL_YES, f"{value:.2f}")


# =============================================================================
# Medium: two variants (extract-forward, inverse), each drawing pure_rate or
# rate_plus_fixed.
# =============================================================================

_GROUP_STATEMENT_RE = re.compile(
    r"^(\d+) [a-zA-Z]+ of [a-z ]+ (?:cost|together cost|costs) \$(\d+\.\d{2})", re.IGNORECASE
)
_BUYING_STATEMENT_RE = re.compile(r"Buying (\d+) [a-zA-Z]+ of [a-z ]+ costs \$(\d+\.\d{2})", re.IGNORECASE)
_MEDIUM_QTY2_RE = re.compile(r"for (\d+) [a-zA-Z]+(?:, including delivery)?\?$")
_DELIVERY_FEE_RE = re.compile(r"one-off \$(\d+\.\d{2}) delivery fee")


def _is_medium_inverse(text: str) -> bool:
    return "how many" in text.lower()


def _medium_extract_oracle_key(text: str) -> str:
    m = _GROUP_STATEMENT_RE.search(text) or _BUYING_STATEMENT_RE.search(text)
    n, x = int(m.group(1)), Fraction(m.group(2))
    q2 = int(_MEDIUM_QTY2_RE.search(text).group(1))
    rate = x / n
    variable = rate * q2
    fee_match = _DELIVERY_FEE_RE.search(text)
    if fee_match:
        fixed = Fraction(fee_match.group(1))
        return _money_opt(_LABEL_NO, f"{fixed + variable:.2f}")
    return _money_opt(_LABEL_YES, f"{variable:.2f}")


def _medium_inverse_oracle_key(text: str) -> str:
    """The rate is the FIRST money figure (`$rate per unit`); a delivery fee
    (if stated) is the money figure inside the `one-off $... delivery fee`
    clause; the remaining stated money figure is the total. Extracted this
    way (rather than by matching each template's own surrounding words) so
    the oracle does not depend on which of the wordings the generator drew."""
    rate = Fraction(_RATE_RE.search(text).group(1))
    fee_match = _DELIVERY_FEE_RE.search(text)
    fixed = Fraction(fee_match.group(1)) if fee_match else Fraction(0)
    all_money = [Fraction(m) for m in re.findall(r"\$(\d+\.\d{2})", text)]
    excluded = {rate, fixed} if fee_match else {rate}
    totals = [m for m in all_money if m not in excluded]
    assert len(totals) == 1, (text, all_money, rate, fixed)
    total = totals[0]
    quantity = (total - fixed) / rate
    assert quantity.denominator == 1, (text, quantity)
    label = _LABEL_NO if fee_match else _LABEL_YES
    # the unit noun is whatever word follows "how many " in the tail clause
    unit_plural_m = re.search(r"how many ([a-zA-Z]+) can you buy", text.lower())
    unit_plural_text = unit_plural_m.group(1)
    return _qty_opt(label, str(int(quantity)), unit_plural_text)


def _medium_oracle_key(q) -> str:
    if _is_medium_inverse(q.text):
        return _medium_inverse_oracle_key(q.text)
    return _medium_extract_oracle_key(q.text)


# =============================================================================
# Hard: both pure_rate and rate_plus_fixed, same larger rate pool/contexts.
# =============================================================================

_FIXED_RE = re.compile(r"\$(\d+\.\d{2}) (?:call-out fee|flat booking fee|a month)")
_HARD_RATE_RE = re.compile(r"\$(\d+\.\d{2}) per (?:hour|GB|day)")
_HARD_QTY_RE = re.compile(r"for (\d+) (?:hours|GB|days)")


def _hard_oracle_key(text: str) -> str:
    rate = Fraction(_HARD_RATE_RE.search(text).group(1))
    quantity = int(_HARD_QTY_RE.search(text).group(1))
    fixed_match = _FIXED_RE.search(text)
    if fixed_match:
        fixed = Fraction(fixed_match.group(1))
        return _money_opt(_LABEL_NO, f"{fixed + rate * quantity:.2f}")
    return _money_opt(_LABEL_YES, f"{rate * quantity:.2f}")


# =============================================================================
# Exam: two data points, no formula named.
# =============================================================================

_EXAM_DATA_RE = re.compile(r"for (\d+) \S+ the total is \$(\d+\.\d{2}); for (\d+) \S+ the total is \$(\d+\.\d{2})")
_EXAM_Q3_RE = re.compile(r"Find the total for (\d+) \S+\?$|what is the total for (\d+) \S+\?$")


def _exam_oracle_key(text: str) -> str:
    m = _EXAM_DATA_RE.search(text)
    q1, t1, q2, t2 = int(m.group(1)), Fraction(m.group(2)), int(m.group(3)), Fraction(m.group(4))
    q3_m = _EXAM_Q3_RE.search(text)
    q3 = int(q3_m.group(1) or q3_m.group(2))
    rate = (t2 - t1) / (q2 - q1)
    fixed = t1 - rate * q1
    total3 = fixed + rate * q3
    label = _LABEL_YES if fixed == 0 else _LABEL_NO
    return _money_opt(label, f"{total3:.2f}")


def _exam_kind(text: str) -> str:
    m = _EXAM_DATA_RE.search(text)
    q1, t1, q2, t2 = int(m.group(1)), Fraction(m.group(2)), int(m.group(3)), Fraction(m.group(4))
    rate = (t2 - t1) / (q2 - q1)
    fixed = t1 - rate * q1
    return "pure_rate" if fixed == 0 else "rate_plus_fixed"


_ORACLE_KEYS = {
    "easy": lambda q: _easy_oracle_key(q.text),
    "medium": lambda q: _medium_oracle_key(q),
    "hard": lambda q: _hard_oracle_key(q.text),
    "exam": lambda q: _exam_oracle_key(q.text),
}


# =============================================================================
# requires: independently determine the classification AND the value from
# the rendered text alone, find the option in `mcq_options(q)` that encodes
# BOTH, and assert it equals `question.answer_text` (the keyed-correct
# option).
# =============================================================================


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_mcq_key_rederived_from_rendered_text_over_1000_seeds(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        expected_key = _ORACLE_KEYS[difficulty](q)
        opts = mcq_options(q)
        assert expected_key in opts, (
            f"seed {seed} {difficulty}: independently-derived key {expected_key!r} is not even "
            f"one of the rendered options {opts!r} -- {q.text!r}"
        )
        assert expected_key == q.answer_text, (
            f"seed {seed} {difficulty}: expected key {expected_key!r}, but the keyed-correct option "
            f"was {q.answer_text!r} -- {q.text!r}"
        )
        checked += 1
    assert checked == 1000


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_oracle_never_matches_a_distractor(difficulty):
    for seed in SEEDS_SMALL:
        q = _make_from_sources(difficulty, seed)
        expected_key = _ORACLE_KEYS[difficulty](q)
        for d in q.distractors:
            assert d != expected_key, (difficulty, seed, q.text, d)


# =============================================================================
# Hand-worked examples
# =============================================================================


def test_oracle_hand_worked_example_easy():
    text = "Fabric costs $2.50 per metre. Is this relationship directly proportional, and what is the total cost for 4 metres?"
    assert _easy_oracle_key(text) == _money_opt(_LABEL_YES, "10.00")


def test_oracle_hand_worked_example_medium_extract_pure():
    text = (
        "5 metres of fabric cost $12.50 in total. Is this relationship directly proportional, and "
        "what is the total cost for 14 metres?"
    )
    assert _medium_extract_oracle_key(text) == _money_opt(_LABEL_YES, "35.00")


def test_oracle_hand_worked_example_medium_extract_fixed():
    text = (
        "5 metres of fabric cost $12.50 in total. There is also a one-off $3.00 delivery fee for the whole order. "
        "Is this relationship directly proportional, and what is the total cost for 14 metres, including delivery?"
    )
    assert _medium_extract_oracle_key(text) == _money_opt(_LABEL_NO, "38.00")


def test_oracle_hand_worked_example_medium_inverse_pure():
    text = (
        "Flour costs $3.50 per kg. Is this relationship directly proportional, and how many kg can you buy for $42.00?"
    )
    assert _medium_inverse_oracle_key(text) == _qty_opt(_LABEL_YES, "12", "kg")


def test_oracle_hand_worked_example_medium_inverse_fixed():
    text = (
        "Flour costs $3.50 per kg, plus a one-off $7.00 delivery fee for the whole order. Is this relationship "
        "directly proportional, and how many kg can you buy if the total spent, including delivery, is $49.00?"
    )
    assert _medium_inverse_oracle_key(text) == _qty_opt(_LABEL_NO, "12", "kg")


def test_oracle_hand_worked_example_hard_fixed():
    text = (
        "A plumber charges a $50.00 call-out fee plus $30.00 per hour. Is this relationship directly proportional, "
        "and what is the total cost for 3 hours of work?"
    )
    assert _hard_oracle_key(text) == _money_opt(_LABEL_NO, "140.00")


def test_oracle_hand_worked_example_hard_pure():
    text = (
        "A car rental costs $40.00 per day of hire. Is this relationship directly proportional, and what is the "
        "total cost for 11 days of hire?"
    )
    assert _hard_oracle_key(text) == _money_opt(_LABEL_YES, "440.00")


def test_oracle_hand_worked_example_exam_fixed():
    text = (
        "A tradesperson's total charge depends on the number of hours worked: for 7 hours the total is $147.50; "
        "for 10 hours the total is $207.50. Is this relationship directly proportional, and what is the total for "
        "13 hours?"
    )
    assert _exam_oracle_key(text) == _money_opt(_LABEL_NO, "267.50")
    assert _exam_kind(text) == "rate_plus_fixed"


def test_oracle_hand_worked_example_exam_pure():
    text = (
        "A subscription service's monthly bill depends on the GB used: for 4 GB the total is $40.00; "
        "for 8 GB the total is $80.00. Is this relationship directly proportional, and what is the total for 12 GB?"
    )
    assert _exam_oracle_key(text) == _money_opt(_LABEL_YES, "120.00")
    assert _exam_kind(text) == "pure_rate"


# =============================================================================
# Enumeration: the two scenario kinds and both medium variants are exercised
# over the full draw space, not merely a seed sample.
# =============================================================================


def test_exam_enumeration_over_200_seeds_covers_both_kinds():
    kinds = {_exam_kind(_make_from_sources("exam", s).text) for s in range(200)}
    assert kinds == {"pure_rate", "rate_plus_fixed"}


def test_medium_enumeration_over_200_seeds_covers_both_variants():
    variants = {_is_medium_inverse(_make_from_sources("medium", s).text) for s in range(200)}
    assert variants == {True, False}


def test_medium_enumeration_over_400_seeds_covers_both_kinds_in_each_variant():
    kinds_extract = set()
    kinds_inverse = set()
    for s in range(400):
        q = _make_from_sources("medium", s)
        has_fee = "delivery" in q.text.lower()
        if _is_medium_inverse(q.text):
            kinds_inverse.add(has_fee)
        else:
            kinds_extract.add(has_fee)
    assert kinds_extract == {True, False}
    assert kinds_inverse == {True, False}


def test_hard_enumeration_over_400_seeds_covers_both_kinds():
    kinds = {"plus" in _make_from_sources("hard", s).text for s in range(400)}
    assert kinds == {True, False}


# =============================================================================
# No tie is possible, by construction -- one sentence per tier.
# =============================================================================


def test_no_tie_easy_all_four_options_distinct_by_value():
    """easy: all four options share the "Yes" label, so distinctness rests
    entirely on the VALUE; `check_pairwise_distinct` on the four raw Fraction
    cent candidates inside the generator's resample loop rules out a value
    collision before the (label, value) pair is ever formatted, so no two
    formatted options can read identically."""
    for seed in range(200):
        q = _make_from_sources("easy", seed)
        opts = mcq_options(q)
        assert len(set(opts)) == 4, (seed, q.text, opts)


@pytest.mark.parametrize("difficulty", ["medium", "hard", "exam"])
def test_no_tie_non_easy_tiers_all_four_options_distinct(difficulty):
    """medium/hard/exam: a tie would require either (a) two candidates with
    the same label AND the same value -- ruled out by the numeric
    `check_pairwise_distinct` call among same-labelled candidates -- or (b)
    a value collision that happens to land the SAME formatted decimal string
    despite different exact Fraction values (e.g. a `_money_rounded` half-up
    rounding landing on a `_money` exact-cent value) -- ruled out by the
    generator's own additional `check(len({answer_opt, *distractor_opts}) ==
    4, ...)` on the fully formatted strings themselves, inside the same
    resample-on-collision retry loop, before any question is ever returned."""
    for seed in range(200):
        q = _make_from_sources(difficulty, seed)
        opts = mcq_options(q)
        assert len(set(opts)) == 4, (difficulty, seed, q.text, opts)


def test_class_balance_medium_over_400_seeds():
    fixed = sum(1 for s in range(400) if "delivery" in _make_from_sources("medium", s).text.lower())
    assert 0 < fixed < 400


def test_class_balance_hard_over_400_seeds():
    fixed = sum(1 for s in range(400) if "plus" in _make_from_sources("hard", s).text)
    assert 0 < fixed < 400


def test_class_balance_exam_over_400_seeds():
    fixed = sum(1 for s in range(400) if _exam_kind(_make_from_sources("exam", s).text) == "rate_plus_fixed")
    assert 0 < fixed < 400
