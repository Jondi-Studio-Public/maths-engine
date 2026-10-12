"""Tests the implication-symbol questions: predicate names, ordered predicate pairs and answer checks.

Implication symbols build."""

from __future__ import annotations

from collections import Counter

import pytest
import yaml

from mathgen.grading.typed import GRADERS, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p9a_implication_symbols_build import (
    BICONDITIONAL,
    FORWARD_ONLY,
    UNRELATED,
    _PREDICATES,
    _forward_backward,
    _negation_via_identity,
    _transform_always_equivalent,
)

TOPIC1 = "implication_transform_identify"
TOPIC2 = "implication_equivalence_symbol_classify"
SKILL1_ID = "senior-use-implication-converse-equivalence-negation-contrapositive"
SKILL2_ID = "senior-use-the-symbols-for-implication-rightarrow-equivalence-longleftrightarrow"
SKILL1_PATH = "skills/senior-mathematics/" + SKILL1_ID + ".yaml"
SKILL2_PATH = "skills/senior-mathematics/" + SKILL2_ID + ".yaml"
SEEDS40 = range(40)


def _load_sources(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


SKILL1_SOURCES = _load_sources(SKILL1_PATH)
SKILL2_SOURCES = _load_sources(SKILL2_PATH)
TIER1 = {s["difficulty"]: s["params"] for s in SKILL1_SOURCES}
TIER2 = {s["difficulty"]: s["params"] for s in SKILL2_SOURCES}


# ---------------------------------------------------------------------------
# 0. The predicate-name invariant every clause/not_clause builder depends on.
# ---------------------------------------------------------------------------


def test_every_predicate_name_starts_with_is():
    assert all(name.startswith("is ") for name, _fn in _PREDICATES)


# ---------------------------------------------------------------------------
# 1. Load-bearing sweep: all 30 ordered (P, Q) pairs have a non-empty
#    candidates list (skill 1, step 2 of the construction).
# ---------------------------------------------------------------------------


def test_all_30_ordered_predicate_pairs_have_a_candidate():
    empty = []
    for i in range(len(_PREDICATES)):
        for j in range(len(_PREDICATES)):
            if i == j:
                continue
            _, p_fn = _PREDICATES[i]
            _, q_fn = _PREDICATES[j]
            candidates = [n for n in range(2, 41) if p_fn(n) != q_fn(n)]
            if not candidates:
                empty.append((i, j))
    assert not empty, f"pairs with no candidate n: {empty}"


def test_all_30_ordered_predicate_pairs_have_a_candidate_for_both_negation_targets():
    # Mirrors the sweep above, but for the target-filtered pools the hard
    # tier's negation_truth construction now draws from: negation_target=True
    # needs some n with P(n) and not Q(n); negation_target=False needs some n
    # with (not P(n)) and Q(n).
    empty = []
    for i in range(len(_PREDICATES)):
        for j in range(len(_PREDICATES)):
            if i == j:
                continue
            _, p_fn = _PREDICATES[i]
            _, q_fn = _PREDICATES[j]
            true_pool = [n for n in range(2, 41) if p_fn(n) and not q_fn(n)]
            false_pool = [n for n in range(2, 41) if (not p_fn(n)) and q_fn(n)]
            if not true_pool:
                empty.append((i, j, True))
            if not false_pool:
                empty.append((i, j, False))
    assert not empty, f"(i, j, negation_target) combinations with no candidate n: {empty}"


# ---------------------------------------------------------------------------
# 2. Load-bearing sweep: every curated triple classifies correctly for every
#    N in [24, 40] (skill 2's medium/hard construction note).
# ---------------------------------------------------------------------------


def test_forward_only_triples_hold_for_every_n_in_24_40():
    for n_hi in range(24, 41):
        for idx, (_, a_fn, _, b_fn) in enumerate(FORWARD_ONLY):
            forward, backward = _forward_backward(a_fn, b_fn, n_hi)
            assert forward and not backward, f"FORWARD_ONLY[{idx}] fails at N={n_hi}"


def test_biconditional_triples_hold_for_every_n_in_24_40():
    for n_hi in range(24, 41):
        for idx, (_, a_fn, _, b_fn) in enumerate(BICONDITIONAL):
            forward, backward = _forward_backward(a_fn, b_fn, n_hi)
            assert forward and backward, f"BICONDITIONAL[{idx}] fails at N={n_hi}"


def test_unrelated_triples_hold_for_every_n_in_24_40():
    for n_hi in range(24, 41):
        for idx, (_, a_fn, _, b_fn) in enumerate(UNRELATED):
            forward, backward = _forward_backward(a_fn, b_fn, n_hi)
            assert not forward and not backward, f"UNRELATED[{idx}] fails at N={n_hi}"


# ---------------------------------------------------------------------------
# 3. Determinism, both topics, every tier, >=40 seeds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,tier", [(TOPIC1, t) for t in ("easy", "medium", "hard")] + [(TOPIC2, t) for t in ("easy", "medium", "hard")]
)
def test_determinism(topic, tier):
    for seed in SEEDS40:
        a = make(topic, seed=seed, difficulty=tier)
        b = make(topic, seed=seed, difficulty=tier)
        assert a.to_json() == b.to_json()


# ---------------------------------------------------------------------------
# 4. Distractors: distinct, non-empty, never equal to the answer, and exactly
#    3 (the fixed distractor-candidate lists never collide with the answer).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,tier", [(TOPIC1, t) for t in ("easy", "medium", "hard")] + [(TOPIC2, t) for t in ("easy", "medium", "hard")]
)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS40:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) == 3


# ---------------------------------------------------------------------------
# 5. Real grader accepts the canonical answer and rejects every distractor
#    and a deliberately wrong index, every topic, every difficulty.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,tier", [(TOPIC1, t) for t in ("easy", "medium", "hard")] + [(TOPIC2, t) for t in ("easy", "medium", "hard")]
)
def test_real_grader_accepts_answer_and_rejects_distractors(topic, tier):
    assert GRADERS["mcq"] is grade_mcq
    for seed in SEEDS40:
        q = make(topic, seed=seed, difficulty=tier)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, str(i)).correct, f"seed={seed} option {i} {option!r} wrongly accepted"


# ---------------------------------------------------------------------------
# 6. The independent checks fire on corrupted input.
# ---------------------------------------------------------------------------


def test_transform_always_equivalent_fires_on_a_broken_contrapositive():
    broken = {"contrapositive": lambda p, q: (not q) or (not p)}  # should be q or (not p)
    assert _transform_always_equivalent("contrapositive", transforms=broken) is False


def test_transform_always_equivalent_true_only_for_contrapositive():
    assert _transform_always_equivalent("contrapositive") is True
    assert _transform_always_equivalent("converse") is False
    assert _transform_always_equivalent("negation") is False


def test_negation_via_identity_matches_direct_truth_table():
    for p_val in (False, True):
        for q_val in (False, True):
            direct = p_val and not q_val
            assert _negation_via_identity(p_val, q_val) == direct


def test_forward_backward_fires_on_a_swapped_pair():
    # FORWARD_ONLY[0] is (multiple of 4) => (even), forward-only. Swapping the
    # roles of A and B must flip which direction holds.
    _, a_fn, _, b_fn = FORWARD_ONLY[0]
    forward, backward = _forward_backward(a_fn, b_fn, 40)
    swapped_forward, swapped_backward = _forward_backward(b_fn, a_fn, 40)
    assert (forward, backward) != (swapped_forward, swapped_backward)


def test_implication_equivalence_symbol_classify_raises_on_a_corrupted_curated_pair(monkeypatch):
    import mathgen.generators.p9a_implication_symbols_build as mod

    # Swap A/B in every FORWARD_ONLY triple: what used to be forward-only
    # becomes backward-only, so the generator's own "forward and not backward"
    # check must now fail and raise.
    swapped = [(b_desc, b_fn, a_desc, a_fn) for a_desc, a_fn, b_desc, b_fn in FORWARD_ONLY]
    monkeypatch.setattr(mod, "FORWARD_ONLY", swapped)
    with pytest.raises(VerificationError):
        for seed in range(60):
            make(TOPIC2, seed=seed, difficulty="hard")


def test_implication_equivalence_symbol_classify_raises_on_a_corrupted_biconditional_pair(monkeypatch):
    import mathgen.generators.p9a_implication_symbols_build as mod

    # Feed the "⇔" branch a forward-only triple: it only implies one way, so
    # the generator's own "forward and backward" check must fail and raise.
    monkeypatch.setattr(mod, "BICONDITIONAL", FORWARD_ONLY)
    with pytest.raises(VerificationError):
        for seed in range(60):
            make(TOPIC2, seed=seed, difficulty="hard")


def test_implication_equivalence_symbol_classify_raises_on_a_corrupted_unrelated_pair(monkeypatch):
    import mathgen.generators.p9a_implication_symbols_build as mod

    # Feed the "unrelated" branch a biconditional triple: both directions
    # hold, so the generator's own "not forward and not backward" check must
    # fail and raise.
    monkeypatch.setattr(mod, "UNRELATED", BICONDITIONAL)
    with pytest.raises(VerificationError):
        for seed in range(60):
            make(TOPIC2, seed=seed, difficulty="hard")


def test_implication_transform_identify_raises_on_an_empty_candidate_pool(monkeypatch):
    import mathgen.generators.p9a_implication_symbols_build as mod

    # Two predicates that always agree (never P(n) != Q(n)) leave `candidates`
    # empty for every n, so the generator's own "no n found" check must raise
    # on the very first draw.
    monkeypatch.setattr(mod, "_PREDICATES", [("is even", lambda n: n % 2 == 0), ("is not odd", lambda n: n % 2 == 0)])
    with pytest.raises(VerificationError):
        make(TOPIC1, seed=0, difficulty="easy")


def test_implication_transform_identify_raises_on_a_broken_transform_table(monkeypatch):
    import mathgen.generators.p9a_implication_symbols_build as mod

    # _transform_always_equivalent's `transforms` default binds `mod._TRANSFORMS`
    # at def time, so mutate the dict IN PLACE rather than rebinding the name --
    # a plain monkeypatch.setattr on `_TRANSFORMS` would silently miss it.
    monkeypatch.setitem(mod._TRANSFORMS, "contrapositive", lambda p, q: (not q) or (not p))
    with pytest.raises(VerificationError):
        for seed in range(200):
            make(TOPIC1, seed=seed, difficulty="medium")


def test_equality_branch_check_is_not_independently_corruptible():
    # The "=" branch's check compares `sum(a for _ in range(b))` to `a * b`,
    # both computed from the same drawn (a, b) inside the function body --
    # there is no external table to monkeypatch the way FORWARD_ONLY/
    # BICONDITIONAL/UNRELATED/_PREDICATES can be. Recorded here rather than
    # left silently unexercised: repeated addition and `*` cannot disagree
    # for the small non-negative integers this branch draws (a, b in 2..12).
    for a in range(2, 13):
        for b in range(2, 13):
            assert sum(a for _ in range(b)) == a * b


# ---------------------------------------------------------------------------
# 7. Both skills load through mathgen.skills.load_skills correctly.
# ---------------------------------------------------------------------------


def test_both_skills_load_with_expected_shape():
    _domains, skills = load_skills()
    by_id = {s.id: s for s in skills}
    for skill_id, topic in ((SKILL1_ID, TOPIC1), (SKILL2_ID, TOPIC2)):
        skill = by_id[skill_id]
        assert skill.prerequisites == []
        assert skill.answer == "mcq"
        assert skill.placement_eligible is False
        assert skill.domain == "senior-mathematics"
        stages = set()
        for source in skill.sources:
            assert source.topic == topic
            stages.update(source.stages)
        assert {0, 1, 2, 3, 4, 5, 6, "maintenance"} <= stages


# ---------------------------------------------------------------------------
# 8. >=20 distinct rendered questions per tier over 60 seeds, both topics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,tier", [(TOPIC1, t) for t in ("easy", "medium", "hard")] + [(TOPIC2, t) for t in ("easy", "medium", "hard")]
)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in range(60)}
    assert len(texts) >= 20, f"{topic}[{tier}]: only {len(texts)} distinct questions over 60 seeds"


# ---------------------------------------------------------------------------
# 9. The skill's own YAML params reproduce exactly the bare-preset output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_skill1_yaml_params_match_bare_preset(tier):
    for seed in range(10):
        bare = make(TOPIC1, seed=seed, difficulty=tier)
        via_yaml = make(TOPIC1, seed=seed, difficulty=tier, **TIER1[tier])
        assert bare.to_json() == via_yaml.to_json()


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_skill2_yaml_params_match_bare_preset(tier):
    for seed in range(10):
        bare = make(TOPIC2, seed=seed, difficulty=tier)
        via_yaml = make(TOPIC2, seed=seed, difficulty=tier, **TIER2[tier])
        assert bare.to_json() == via_yaml.to_json()


# ---------------------------------------------------------------------------
# 10. Distribution sweeps -- report actual counts, never assume uniformity.
# ---------------------------------------------------------------------------


def test_medium_tier_s_kind_distribution_over_200_seeds():
    import re

    counts = Counter()
    for seed in range(200):
        q = make(TOPIC1, seed=seed, difficulty="medium")
        match = re.search(r"Consider its (\w+):", q.text)
        assert match
        counts[match.group(1)] += 1
    assert set(counts) == {"converse", "contrapositive", "negation"}, counts
    for kind in ("converse", "contrapositive", "negation"):
        assert counts[kind] >= 30, f"s_kind {kind} only occurred {counts[kind]}/200 times: {counts}"


def test_hard_tier_negation_truth_split_over_200_seeds():
    counts = Counter()
    for seed in range(200):
        q = make(TOPIC1, seed=seed, difficulty="hard")
        counts[q.answer_text.split(",")[0]] += 1  # "True, because ..." / "False, because ..."
    assert set(counts) == {"True", "False"}, counts
    assert counts["True"] > 0 and counts["False"] > 0, counts


def test_medium_tier_answer_balanced_over_400_seeds():
    """Neither True nor False should be much more than half of draws -- the
    equivalence_check tier's target is chosen before s_kind, not left to fall
    out of a uniform 3-way s_kind choice."""
    counts = Counter()
    for seed in range(400):
        q = make(TOPIC1, seed=seed, difficulty="medium")
        counts[q.answer_text] += 1
    total = sum(counts.values())
    for label, n in counts.items():
        share = n / total
        assert share <= 0.6, f"{label!r} is {share:.0%} of 400 draws: {dict(counts)}"


def test_hard_tier_answer_balanced_over_400_seeds():
    """Regression guard: the negation_truth tier was already close to fair
    (54/46 at 400 seeds) before the target-driven n-selection fix -- this
    should pass both before and after, so it is NOT a RED-to-GREEN test."""
    counts = Counter()
    for seed in range(400):
        q = make(TOPIC1, seed=seed, difficulty="hard")
        counts[q.answer_text] += 1
    total = sum(counts.values())
    for label, n in counts.items():
        share = n / total
        assert share <= 0.65, f"{label!r} is {share:.0%} of 400 draws: {dict(counts)}"


def test_skill2_hard_tier_4_class_distribution_over_200_seeds():
    counts = Counter()
    for seed in range(200):
        q = make(TOPIC2, seed=seed, difficulty="hard")
        counts[q.answer_text] += 1
    # The 4 reachable answer_text values at hard tier: "=", "⇒", "⇔", and
    # "No single symbol connects these" (the "unrelated" class).
    assert set(counts) == {"=", "⇒", "⇔", "No single symbol connects these"}, counts
    for label, n in counts.items():
        assert n >= 25, f"{label!r} only occurred {n}/200 times: {counts}"
