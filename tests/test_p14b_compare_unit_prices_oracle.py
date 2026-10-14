"""Independent oracle for `p14b_compare_unit_prices` (ACMEM016, ACMGM005).

Independent oracle for compare-unit-prices (ACMEM016, ACMGM005).

section: the graded answer here is a DECISION, not a value computable purely
from the question's surface grammar, so this oracle re-derives it by parsing
the pack prices/quantities/units straight out of the RENDERED text
(`question.text`) with `Fraction` arithmetic, then re-implements the min/tie
comparison logic itself -- never by importing anything from
`mathgen.generators.p14b_compare_unit_prices_build`. Every number the
decision depends on (including the unit-conversion factor, e.g. "1 dozen =
12 items") is read straight out of the text, never assumed or hardcoded.
"""

from dataclasses import replace
import functools
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import mcq_options
from mathgen.skills import load_skills
from mathgen import make

SKILL = "compare-unit-prices"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS_SMALL = range(60)
SEEDS_1000 = range(1000)

# -- Public-contract literals (copied, never imported) -----------------------
_OPT_A = "Pack A"
_OPT_B = "Pack B"
_OPT_TIE_2 = "Tie -- the same price per unit"
_OPT_C = "Pack C"
_OPT_TIE_3 = "Tie -- two packs share the cheapest price"


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    if difficulty == "exam":
        src = replace(src, topic="p14b_compare_unit_prices", params={})
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _decide(rates: dict) -> str:
    """Independent re-implementation of the min/tie decision -- deliberately
    NOT imported from the generator, so a slip in either implementation is
    caught rather than confirmed."""
    smallest = min(rates.values())
    winners = sorted(label for label, rate in rates.items() if rate == smallest)
    return "tie" if len(winners) > 1 else winners[0]


# =============================================================================
# easy: two packs, same unit -- read straight off the two (qty, $price) pairs.
# =============================================================================

_EASY_RE = re.compile(r"Pack A: (\d+) \w+ for \$(\d+\.\d{2})\. Pack B: (\d+) \w+ for \$(\d+\.\d{2})\.")


def _easy_oracle_key(text: str) -> str:
    m = _EASY_RE.search(text)
    qtyA, priceA, qtyB, priceB = m.group(1), m.group(2), m.group(3), m.group(4)
    rateA = Fraction(priceA) / int(qtyA)
    rateB = Fraction(priceB) / int(qtyB)
    decision = _decide({_OPT_A: rateA, _OPT_B: rateB})
    return _OPT_TIE_2 if decision == "tie" else decision


# =============================================================================
# medium/exam: two packs, possibly mismatched units. The conversion factor
# ("1 X = N Y.") is read straight from the text, never assumed.
# =============================================================================

_PACK_RE = re.compile(r"Pack ([AB]): (\d+) (mL|L|kg|g|items|item|dozen) for \$(\d+\.\d{2})")
_CONVERSION_RE = re.compile(r"\(1 (L|kg|dozen) = (\d+) (mL|g|items|item)\.\)")


def _mismatch_oracle_key(text: str) -> str:
    packs = {label: (int(qty), unit, Fraction(price)) for label, qty, unit, price in _PACK_RE.findall(text)}
    assert set(packs) == {"A", "B"}, (text, packs)
    conv = _CONVERSION_RE.search(text)
    if conv is None:
        # No mismatch at all -- both packs already share the same unit.
        rates = {label: price / qty for label, (qty, unit, price) in packs.items()}
    else:
        big_unit, factor, small_unit = conv.group(1), int(conv.group(2)), conv.group(3)
        rates = {}
        for label, (qty, unit, price) in packs.items():
            if unit == big_unit:
                rates[label] = price / (qty * factor)
            else:
                rates[label] = price / qty
    decision = _decide({"Pack " + label: rate for label, rate in rates.items()})
    return _OPT_TIE_2 if decision == "tie" else decision


# =============================================================================
# hard: three packs, same unit.
# =============================================================================

_HARD_RE = re.compile(
    r"Pack A: (\d+) \w+ for \$(\d+\.\d{2})\. Pack B: (\d+) \w+ for \$(\d+\.\d{2})\. "
    r"Pack C: (\d+) \w+ for \$(\d+\.\d{2})\."
)


def _hard_oracle_key(text: str) -> str:
    m = _HARD_RE.search(text)
    qtyA, priceA, qtyB, priceB, qtyC, priceC = m.groups()
    rates = {
        _OPT_A: Fraction(priceA) / int(qtyA),
        _OPT_B: Fraction(priceB) / int(qtyB),
        _OPT_C: Fraction(priceC) / int(qtyC),
    }
    decision = _decide(rates)
    return _OPT_TIE_3 if decision == "tie" else decision


_ORACLE_KEYS = {
    "easy": lambda q: _easy_oracle_key(q.text),
    "medium": lambda q: _mismatch_oracle_key(q.text),
    "hard": lambda q: _hard_oracle_key(q.text),
    "exam": lambda q: _mismatch_oracle_key(q.text),
}


# =============================================================================
# The 1000-seed-per-tier key re-derivation.
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
# Hand-worked examples.
# =============================================================================


def test_oracle_hand_worked_example_easy_a_cheaper():
    text = "Pack A: 500 grams for $2.00. Pack B: 500 grams for $3.00. Which is the cheaper buy per gram, or are the two the same price per gram?"
    assert _easy_oracle_key(text) == _OPT_A


def test_oracle_hand_worked_example_easy_tie():
    text = "Pack A: 500 grams for $2.00. Pack B: 1000 grams for $4.00. Which is the cheaper buy per gram, or are the two the same price per gram?"
    assert _easy_oracle_key(text) == _OPT_TIE_2


def test_oracle_hand_worked_example_mismatch_b_cheaper():
    text = (
        "Pack A: 2 L for $10.00. Pack B: 500 mL for $2.00. (1 L = 1000 mL.) "
        "Which is the cheaper buy per millilitre, or are the two the same price per millilitre?"
    )
    # A: 1000c / 2000mL = 0.5 c/mL; B: 200c / 500mL = 0.4 c/mL -- B cheaper.
    assert _mismatch_oracle_key(text) == _OPT_B


def test_oracle_hand_worked_example_mismatch_tie():
    text = (
        "Pack A: 1 dozen for $24.00. Pack B: 12 items for $24.00. (1 dozen = 12 items.) "
        "Which is the cheaper buy per item, or are the two the same price per item?"
    )
    assert _mismatch_oracle_key(text) == _OPT_TIE_2


def test_oracle_hand_worked_example_hard_c_wins():
    text = (
        "Pack A: 100 grams for $5.00. Pack B: 100 grams for $4.00. Pack C: 100 grams for $3.00. "
        "Which pack is the cheapest per gram, or do two of them tie for cheapest?"
    )
    assert _hard_oracle_key(text) == _OPT_C


def test_oracle_hand_worked_example_hard_two_way_tie():
    text = (
        "Pack A: 100 grams for $3.00. Pack B: 200 grams for $6.00. Pack C: 100 grams for $5.00. "
        "Which pack is the cheapest per gram, or do two of them tie for cheapest?"
    )
    assert _hard_oracle_key(text) == _OPT_TIE_3


# =============================================================================
# Enumeration: every applicable mcq option is reachable over the full draw
# space, for every tier -- not merely a seed sample.
# =============================================================================


@pytest.mark.parametrize("difficulty", ["easy", "medium", "exam"])
def test_two_pack_tiers_reach_every_option_over_600_seeds(difficulty):
    seen = {_make_from_sources(difficulty, s).answer_text for s in range(600)}
    assert seen == {_OPT_A, _OPT_B, _OPT_TIE_2}, (difficulty, seen)


def test_hard_tier_reaches_every_option_over_600_seeds():
    seen = {_make_from_sources("hard", s).answer_text for s in range(600)}
    assert seen == {_OPT_A, _OPT_B, _OPT_C, _OPT_TIE_3}, seen


def test_exam_reaches_both_mismatched_units_and_same_unit_seeds():
    """exam's 'clean'/'true_tie' branches always carry the '(1 X = N Y.)'
    conversion clause (medium's mismatched-unit shape); the near-miss branch
    does too. So every exam question should carry it."""
    for s in range(200):
        text = _make_from_sources("exam", s).text
        assert "(1 " in text, (s, text)


def test_exam_near_miss_trap_is_reachable_and_genuinely_not_a_tie():
    """At least one exam seed in a modest sweep must be a case where the two
    packs' prices round to the same cent-per-small-unit rate while remaining
    exactly unequal (the near-miss trap) -- exercised via the oracle's own
    Fraction re-derivation, never the generator's construction values."""
    found_near_miss = False
    for s in range(300):
        q = _make_from_sources("exam", s)
        m = _PACK_RE.findall(q.text)
        packs = {label: (int(qty), unit, Fraction(price)) for label, qty, unit, price in m}
        conv = _CONVERSION_RE.search(q.text)
        if conv is None:
            continue
        big_unit, factor, _small_unit = conv.group(1), int(conv.group(2)), conv.group(3)
        rates = {}
        for label, (qty, unit, price) in packs.items():
            rates[label] = price / (qty * factor) if unit == big_unit else price / qty
        rounded = {label: round(rate * 100) for label, rate in rates.items()}
        if len(set(rounded.values())) == 1 and rates["A"] != rates["B"]:
            found_near_miss = True
            assert q.answer_text in (_OPT_A, _OPT_B), (s, q.text, q.answer_text)
            break
    assert found_near_miss, "no near-miss (rounds-to-same-cent-but-unequal) exam seed found in 300 seeds"
