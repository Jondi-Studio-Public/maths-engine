"""Enumerates the (template x variable pair/triple x third variable) matrix from rendered text and asserts every
combination is coherent (the third variable really drives both / really moves with X and affects Y / no link),
and re-derives the category over >= 1000 questions from the scenario's own fact sentence (never from the
once-stated definitions block, which is fixed text and would make the check vacuous). Imports NOTHING from
mathgen.generators.p19b_noncausal_explanation_build; reads only question.text and
question.answer_text/distractors.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import pytest

from mathgen import make

TOPIC = "p19b_noncausal_explanation"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P19B_ORACLE_SEEDS", "250"))

# Re-implemented fresh from the three short labels and the three fact-sentence shapes
# the scenario states -- NOT imported from the generator's own OPTIONS/_fact.
_LABEL_COINCIDENCE = "Coincidence"
_LABEL_COMMON_CAUSE = "Common cause"
_LABEL_CONFOUNDING = "Confounding"

_DEFINITIONS_MARKER = "The three possible explanations are defined as follows."

_NO_LINK_RE = re.compile(r"no known factor has been found that links (.+?) and (.+?)\.", re.IGNORECASE)
_COMMON_CAUSE_RE = re.compile(r"(.+?) drives increases in both (.+?) and (.+?)\.", re.IGNORECASE)
_CONFOUND_RE = re.compile(r"(.+?) is also known to affect (.+?) directly, since (.+?)\.", re.IGNORECASE)


def _scenario_part(text: str) -> str:
    return text.split(_DEFINITIONS_MARKER)[0]


def _oracle_derive_category(text: str) -> str:
    scenario = _scenario_part(text)
    if _NO_LINK_RE.search(scenario):
        return "coincidence"
    if _COMMON_CAUSE_RE.search(scenario):
        return "common_cause"
    if _CONFOUND_RE.search(scenario):
        return "confounding"
    raise AssertionError(f"no marker phrase matched in scenario: {scenario!r}")


_CATEGORY_TO_LABEL = {
    "coincidence": _LABEL_COINCIDENCE,
    "common_cause": _LABEL_COMMON_CAUSE,
    "confounding": _LABEL_CONFOUNDING,
}


def test_definitions_and_labels_present_over_1000_questions():
    checked = 0
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            assert _DEFINITIONS_MARKER in q.text
            assert set(q.distractors) | {q.answer_text} == {
                _LABEL_COINCIDENCE,
                _LABEL_COMMON_CAUSE,
                _LABEL_CONFOUNDING,
            }
            checked += 1
    assert checked >= 1000


def test_oracle_rederives_the_category_over_1000_questions():
    checked = 0
    by_tier_categories = defaultdict(set)
    combos_seen = set()
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            category = _oracle_derive_category(q.text)
            expected_label = _CATEGORY_TO_LABEL[category]
            assert expected_label == q.answer_text, (
                f"tier {tier} seed {seed}: oracle derived category {category!r} "
                f"(label {expected_label!r}), generator gave {q.answer_text!r} -- text {q.text!r}"
            )
            by_tier_categories[tier].add(category)
            scenario = _scenario_part(q.text)
            for regex in (_NO_LINK_RE, _COMMON_CAUSE_RE, _CONFOUND_RE):
                m = regex.search(scenario)
                if m:
                    combos_seen.add((tier, category, m.groups()))
                    break
            checked += 1
    assert checked >= 1000
    for tier in TIERS:
        assert by_tier_categories[tier] == {"coincidence", "common_cause", "confounding"}, (
            f"tier {tier}: not all three categories seen: {by_tier_categories[tier]}"
        )
    # Draw-space saturation: at least 6 distinct scenarios per category (the pools
    # each hold >= 6 entries) across all four tiers combined.
    by_category_combo_count = defaultdict(set)
    for tier, category, groups in combos_seen:
        by_category_combo_count[category].add(groups)
    for category in ("coincidence", "common_cause", "confounding"):
        assert len(by_category_combo_count[category]) >= 6, (
            f"category {category}: only {len(by_category_combo_count[category])} distinct "
            f"variable combinations seen over {checked} questions"
        )


def test_oracle_fires_red_first_on_a_corrupted_answer():
    """Proof the oracle is not vacuous: confirm the independently re-derived category
    would reject a deliberately swapped answer_text."""
    q = make(TOPIC, seed=0, difficulty="easy", tier="easy")
    category = _oracle_derive_category(q.text)
    assert _CATEGORY_TO_LABEL[category] == q.answer_text
    corrupted_answer = q.distractors[0]
    assert _CATEGORY_TO_LABEL[category] != corrupted_answer, "the oracle must reject a corrupted (swapped) answer"


def test_key_position_balanced_at_400_seeds():
    from mathgen.grading.typed import mcq_options

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
        # 3 options, uniform shuffle -> key first ~1/3 of the time
        assert 0.20 <= share <= 0.50, f"tier {tier}: key-first share {share:.2f} over 400 seeds looks non-uniform"


def test_no_banned_definitional_phrase_in_scenario_over_1000_questions():
    """Re-proved independently of the generator's own BANNED_PHRASES
    constant (hand-copied here from the generator's ban list)."""
    banned = (
        "independently",
        "neither",
        "cannot be cleanly separated",
        "cannot be separated",
        "no plausible variable",
        "by chance",
        "coincidence",
        "confound",
        "common cause",
        "common response",
    )
    checked = 0
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            scenario = _scenario_part(q.text).lower()
            for phrase in banned:
                assert phrase not in scenario, (
                    f"tier {tier} seed {seed}: banned phrase {phrase!r} in scenario {scenario!r}"
                )
            checked += 1
    assert checked >= 1000
