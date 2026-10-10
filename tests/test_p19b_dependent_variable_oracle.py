"""Enumerates the template x scenario x order matrix from rendered text and asserts every combination is
coherent; re-derives the key from the stated aim/verb clause over >= 1000 questions, by a route independent of
the generator's own labelling. Imports NOTHING from mathgen.generators.p19b_dependent_variable_build; reads
only question.text, question.answer_text and question.distractors.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import pytest

from mathgen.grading.typed import mcq_options
from mathgen import make

TOPIC = "p19b_dependent_variable"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P19B_ORACLE_SEEDS", "250"))

_QUESTION_RE = re.compile(r"Which quantity is the dependent variable in this (experiment|study)\?$")

# Re-implemented fresh from the four aim/verb clause shapes the stimulus can state -- NOT read from
# the generator's own PAIRS table or tier builder functions. Checked in this priority order because
# "affects" also occurs inside the exam-tier aim clause. "is affected by X" names the subject
# (before the clause) as the dependent variable; "X affects Y" names Y (after the clause) as the
# dependent variable.
_IS_AFFECTED_BY_RE = re.compile(r"(?:whether|that) ([^.]+?) is affected by ([^.]+?)\.")
_CHANGES_ACCORDING_TO_RE = re.compile(r"whether ([^.]+?) changes according to ([^;]+?);")
_AFFECTS_RE = re.compile(r"affects ([^.]+?)\.")
_MEASURES_RE = re.compile(r"measures ([^,]+?)(?:,| while)")


def _oracle_derive_dependent(text: str) -> str:
    m = _IS_AFFECTED_BY_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _CHANGES_ACCORDING_TO_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _AFFECTS_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _MEASURES_RE.search(text)
    if m:
        return m.group(1).strip()
    raise AssertionError(f"no aim/verb clause matched in text: {text!r}")


def test_question_sentence_present_and_dependent_word_count_over_1000_questions():
    checked = 0
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            assert _QUESTION_RE.search(q.text), f"tier {tier} seed {seed}: no question sentence in {q.text!r}"
            assert q.text.count("dependent") == 1
            assert "independent" not in q.text
            checked += 1
    assert checked >= 1000
    for tier in TIERS:
        tier_texts = {make(TOPIC, seed=s, difficulty=tier, tier=tier).text for s in range(60)}
        assert len(tier_texts) >= 20, f"tier {tier}: only {len(tier_texts)} distinct texts over 60 seeds"


def test_oracle_rederives_the_dependent_variable_over_1000_questions():
    """The independently re-implemented clause parser must agree with
    answer_text on every question, across all four tiers, over >= 1000 draws."""
    checked = 0
    by_tier_answers = defaultdict(set)
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            derived = _oracle_derive_dependent(q.text)
            assert derived == q.answer_text, (
                f"tier {tier} seed {seed}: oracle derived {derived!r}, "
                f"generator gave {q.answer_text!r} -- text {q.text!r}"
            )
            assert q.answer_text not in q.distractors
            by_tier_answers[tier].add(q.answer_text)
            checked += 1
    assert checked >= 1000
    for tier in TIERS:
        # 15 scenarios in the pool -> up to 15 distinct dependent-variable texts per tier
        assert len(by_tier_answers[tier]) >= 8, (
            f"tier {tier}: only {len(by_tier_answers[tier])} distinct dependent-variable "
            f"texts seen over {N_SEEDS} seeds"
        )


def test_oracle_fires_red_first_on_a_corrupted_answer():
    """Proof the oracle is not vacuous: swap the generator's real answer_text for a
    distractor and confirm the independent parse still derives the TRUE dependent
    variable, so the oracle's own equality assertion would fail against the swap."""
    q = make(TOPIC, seed=0, difficulty="easy", tier="easy")
    derived = _oracle_derive_dependent(q.text)
    assert derived == q.answer_text
    corrupted_answer = q.distractors[0]
    assert derived != corrupted_answer, "the oracle must reject a corrupted (swapped) answer"


_FOR_EACH_UNIT_RE = re.compile(r"for (?:each|every) ([a-z]+)")
# Words that, when seen anywhere in the stimulus, license that per-case unit word.
# Independently enumerated from the rendered text, not read from the generator's PAIRS.
_CONTEXT_UNIT_WHITELIST = {
    "plant": {"plant", "seedling"},
    "seedling": {"seedling", "plant"},
    "student": {"student"},
    "participant": {"participant"},
    "patient": {"patient"},
    "person": {"person"},
    "musician": {"musician"},
    "store": {"store", "week"},
    "week": {"week", "store"},
}


def test_per_case_unit_word_matches_context_over_1000_questions():
    """Independently enumerate every 'for each/every <unit>' phrase the stimulus can
    state and confirm the unit word fits a context keyword also present in the same
    stimulus."""
    seen_units = set()
    checked = 0
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            for unit in _FOR_EACH_UNIT_RE.findall(q.text):
                if unit == "case":
                    continue  # context-neutral wording (medium tier); never a mismatch
                seen_units.add(unit)
                allowed = _CONTEXT_UNIT_WHITELIST.get(unit)
                assert allowed, f"unrecognised per-case unit word {unit!r} in {q.text!r}"
                assert any(word in q.text for word in allowed), (
                    f"tier {tier} seed {seed}: unit word {unit!r} has no matching context keyword in {q.text!r}"
                )
                checked += 1
    assert checked > 0
    # confirm the enumeration actually exercised more than one unit word (hard tier only)
    assert len(seen_units) >= 5, f"only saw unit words {seen_units}"


_INCOHERENT_HELD_CONSTANT_RE = re.compile(r"(measured|taken|score|speed|rate|severity)[^.,;]*\bwas kept the same\b")


def test_no_incoherent_held_constant_measurement_over_1000_questions():
    """Deny any sentence that describes a measured/recorded outcome as held constant."""
    checked = 0
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            assert not _INCOHERENT_HELD_CONSTANT_RE.search(q.text), (
                f"tier {tier} seed {seed}: incoherent held-constant measurement in {q.text!r}"
            )
            checked += 1
    assert checked >= 1000


def test_sentence_case_over_1000_questions():
    """Every sentence in question.text starts with a capital letter."""
    checked = 0
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            sentences = [s.strip() for s in q.text.split(". ") if s.strip()]
            for sentence in sentences:
                assert sentence[0].isupper(), (
                    f"tier {tier} seed {seed}: sentence does not start with a capital: {sentence!r} in {q.text!r}"
                )
            checked += 1
    assert checked >= 1000


def test_key_position_balanced_at_400_seeds():
    for tier in TIERS:
        first_count = 0
        total = 0
        for seed in range(400):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            options = mcq_options(q)
            if options[0] == q.answer_text:
                first_count += 1
            total += 1
        share = first_count / total
        # with 3-4 options, a uniform shuffle puts the key first ~1/3 to 1/4 of the time
        assert 0.15 <= share <= 0.45, f"tier {tier}: key-first share {share:.2f} over 400 seeds looks non-uniform"
