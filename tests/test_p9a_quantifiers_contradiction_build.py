"""Tests the quantifier-evaluation and proof-by-contradiction questions, including predicate pool sizes.

Quantifiers contradiction build."""

from __future__ import annotations

import random
import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
import mathgen.generators.p9a_quantifiers_contradiction_build as p9a_mod
from mathgen.generators.p9a_quantifiers_contradiction_build import (
    _PAIRS,
    _PREDICATES,
    _SELECT_OPTIONS,
    _adj_truth,
    _compound_claim,
    _false_statement_for,
    _predicate_pool_sizes,
    _single_tier,
)

QUANT_TOPIC = "quantifier_forall_exists_evaluate"
CONTRA_TOPIC = "proof_by_contradiction_identify"
SEEDS = range(60)

QUANT_VARIANT = {"easy": "exists", "medium": "forall", "hard": "select_quantifier"}
CONTRA_VARIANT = {"easy": "single_simple", "medium": "single_computed", "hard": "compound_claim"}
TIERS = ["easy", "medium", "hard"]


def _make_quant(seed, difficulty):
    return make(QUANT_TOPIC, seed=seed, difficulty=difficulty, variant=QUANT_VARIANT[difficulty])


def _make_contra(seed, difficulty):
    return make(CONTRA_TOPIC, seed=seed, difficulty=difficulty, variant=CONTRA_VARIANT[difficulty])


# ===========================================================================
# Pool-size report
# ===========================================================================


def test_predicate_pool_sizes_are_at_least_7_each_side():
    sizes = _predicate_pool_sizes()
    assert len(sizes) == 6
    for name, (passing, failing) in sizes.items():
        assert passing >= 7 and failing >= 7, f"{name}: passing={passing} failing={failing}"


# ===========================================================================
# quantifier_forall_exists_evaluate
# ===========================================================================


class TestQuantifierEvaluate:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = _make_quant(seed, difficulty)
            q2 = _make_quant(seed, difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = _make_quant(seed, difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = _make_quant(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, str(correct_index)).correct
            for dist in q.distractors:
                dist_index = options.index(dist)
                assert not grade_mcq(q, str(dist_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_in_options(self, difficulty):
        for seed in SEEDS:
            q = _make_quant(seed, difficulty)
            assert q.answer_text in mcq_options(q)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions(self, difficulty):
        texts = {_make_quant(seed, difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    @pytest.mark.parametrize("difficulty", ["easy", "medium"])
    def test_true_false_answer_shape(self, difficulty):
        for seed in SEEDS:
            q = _make_quant(seed, difficulty)
            # The options are reasoned verdicts ("True, because ..."); every distractor gives the wrong verdict.
            verdict = q.answer_text.split(",")[0]
            assert verdict in ("True", "False")
            wrong = "False" if verdict == "True" else "True"
            assert q.distractors and all(d.split(",")[0] == wrong for d in q.distractors), q.distractors

    def test_hard_tier_for_all_only_never_the_answer(self):
        for seed in SEEDS:
            q = _make_quant(seed, "hard")
            assert q.answer_text != _SELECT_OPTIONS[0]

    def test_hard_tier_reachable_class_distribution(self):
        """The 3 reachable classes should occur roughly 1/3 each over >=100 seeds."""
        counts = Counter()
        for seed in range(150):
            q = make(QUANT_TOPIC, seed=seed, difficulty="hard", variant="select_quantifier")
            counts[q.answer] += 1
        assert set(counts) <= {"both", "exists_only", "neither"}
        assert len(counts) == 3, f"not all 3 classes reached: {counts}"
        for cls, n in counts.items():
            assert n >= 30, f"class {cls} only reached {n}/150 times: {counts}"

    def test_set_notation_not_python_list(self):
        for difficulty in TIERS:
            for seed in range(10):
                q = _make_quant(seed, difficulty)
                assert "[" not in q.text and "]" not in q.text

    def test_exists_tier_independent_check_fires_on_a_corrupted_pad(self, monkeypatch):
        """Drives the ACTUAL `found == target` check inside the generator:
        force a single deterministic predicate and a `_pad_to_k` that always
        returns values failing it, then find a seed whose randomly-chosen
        target is True (so construction calls the now-broken `_pad_to_k` to
        seed a passing value) and confirm the corrupted build raises."""
        monkeypatch.setattr(p9a_mod, "_PREDICATES", [("is even", lambda n: n % 2 == 0)])
        monkeypatch.setattr(p9a_mod, "_pad_to_k", lambda base, k, rng: [3, 5, 7, 9, 11])
        raised = False
        for seed in range(50):
            try:
                make(QUANT_TOPIC, seed=seed, difficulty="easy", variant="exists")
            except VerificationError:
                raised = True
                break
        assert raised, "no seed in range(50) exercised the target=True (padded) branch"

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-use-the-quantifiers-for-all-and-there-exists"), None)
        assert skill is not None
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "mcq"
        assert skill.prerequisites == []
        assert skill.placement_eligible is False
        assert len(skill.sources) == 3


# ===========================================================================
# proof_by_contradiction_identify
# ===========================================================================


class TestProofByContradiction:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = _make_contra(seed, difficulty)
            q2 = _make_contra(seed, difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = _make_contra(seed, difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = _make_contra(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, str(correct_index)).correct
            for dist in q.distractors:
                dist_index = options.index(dist)
                assert not grade_mcq(q, str(dist_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_text_never_in_question_text(self, difficulty):
        """The direct test: the resolution is never given away."""
        for seed in SEEDS:
            q = _make_contra(seed, difficulty)
            assert q.answer_text not in q.text, f"seed {seed} {difficulty}: {q.answer_text!r} leaked into {q.text!r}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_word_contradiction_never_paired_with_a_specific_resolution(self, difficulty):
        """No adjective from the answer appears directly beside the word
        'contradiction'/'contradicts' in a way that would give the game away --
        approximated by checking the sentence containing 'contradict' never
        also contains the answer's own adjective clause."""
        for seed in SEEDS:
            q = _make_contra(seed, difficulty)
            for sentence in re.split(r"(?<=[.?])\s+", q.text):
                if "contradict" in sentence:
                    assert q.answer_text not in sentence

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions(self, difficulty):
        texts = {_make_contra(seed, difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_perfect_square_lambda_guarded_against_negative_n(self):
        """math.isqrt raises ValueError on a negative argument; the perfect-square
        lambdas in _PAIRS are guarded with `n >= 0 and ...` -- confirmed here by
        actually calling them on negative n, not by reading the code."""
        _, _, _, claimed_fn, _ = _PAIRS[4]
        actual_adj, actual_fn, claimed_adj, _, _ = _PAIRS[4]
        for n in (-1, -5, -30):
            assert actual_fn(n) is False
            assert claimed_fn(n) is True

    def test_pairs_complementary_and_mutually_exclusive(self):
        for actual_adj, actual_fn, claimed_adj, claimed_fn, domain in _PAIRS:
            for x in domain:
                assert actual_fn(x) != claimed_fn(x), f"{actual_adj}/{claimed_adj} at {x}"

    def test_false_statement_for_is_always_false(self):
        for i, (actual_adj, actual_fn, claimed_adj, claimed_fn, domain) in enumerate(_PAIRS):
            for n in domain[:20]:
                for j, other in enumerate(_PAIRS):
                    if j == i:
                        continue
                    stmt = _false_statement_for(n, other)
                    stated_adj = stmt[len(f"{n} is ") :]
                    assert _adj_truth(n, other, stated_adj) is False

    def test_compound_claim_false_clause_distribution(self):
        counts = Counter()
        for seed in range(250):
            q = make(CONTRA_TOPIC, seed=seed, difficulty="hard", variant="compound_claim")
            counts["first" if q.answer_text.startswith("The first") else "second"] += 1
        assert counts["first"] >= 90 and counts["second"] >= 90, f"skewed split: {dict(counts)}"

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-use-proof-by-contradiction"), None)
        assert skill is not None
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "mcq"
        assert skill.prerequisites == []
        assert skill.placement_eligible is False
        assert len(skill.sources) == 3

    def test_single_tier_independent_check_fires_on_a_broken_pair(self):
        """Drives the ACTUAL `_single_tier` construction with a deliberately
        non-complementary pair (claimed_fn is a copy of actual_fn, not its
        complement) -- its own `check(actual_fn(n) and not claimed_fn(n), ...)`
        must raise, not silently build a broken question."""
        broken_pair = ("even", lambda n: n % 2 == 0, "even-again", lambda n: n % 2 == 0, list(range(2, 10)))
        with pytest.raises(VerificationError):
            _single_tier(random.Random(0), [broken_pair])

    def test_compound_claim_independent_check_fires_when_no_intersection_exists(self, monkeypatch):
        """Drives the ACTUAL `_compound_claim` construction with two pairs
        whose `actual_fn`s can never both hold -- its own
        `check(intersection, ...)` (after the retry loop is exhausted) must
        raise."""
        never_pair = ("even", lambda n: False, "odd", lambda n: True, list(range(2, 10)))
        always_pair = ("odd", lambda n: True, "even", lambda n: False, list(range(2, 10)))
        monkeypatch.setattr(p9a_mod, "_PAIRS", [never_pair, always_pair])
        with pytest.raises(VerificationError):
            _compound_claim(random.Random(0))

    def test_adj_truth_raises_on_unknown_adjective(self):
        """A corrupted/unknown adjective is rejected, not silently mis-evaluated."""
        with pytest.raises(ValueError):
            _adj_truth(4, _PAIRS[0], "purple")
