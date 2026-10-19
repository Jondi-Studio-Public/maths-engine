"""Tests `counterexample_identify` and its predicate pools.

Examples counterexamples build."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
import mathgen.generators.p9a_examples_counterexamples_build as p9a_mod
from mathgen.generators.p9a_examples_counterexamples_build import (
    _CLASSIFY_OPTIONS,
    _DOMAIN,
    _EASY_INDICES,
    _MEDIUM_INDICES,
    _INDEPENDENT_CHECKS,
    _PREDICATES,
    _classify,
    _find_counterexample,
    _predicate_pool_sizes,
)

TOPIC = "counterexample_identify"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

_PARAMS = {
    "easy": dict(variant="classify", pool="easy"),
    "medium": dict(variant="classify", pool="medium"),
    "hard": dict(variant="find_counterexample"),
}


def _mk(seed, difficulty):
    return make(TOPIC, seed=seed, difficulty=difficulty, **_PARAMS[difficulty])


# ===========================================================================
# Pool-size report (every one of the 6 predicates)
# ===========================================================================


def test_predicate_pool_sizes_reported_and_nonempty():
    sizes = _predicate_pool_sizes()
    assert len(sizes) == 6
    for name, (passing, failing) in sizes.items():
        assert passing >= 1 and failing >= 1, f"{name}: passing={passing} failing={failing}"
        # hard tier needs >=3 passing, >=1 failing for every predicate
        assert passing >= 3, f"{name}: passing={passing} < 3, hard tier cannot sample 3"


# ===========================================================================
# Shared behaviour across all three tiers
# ===========================================================================


class TestCounterexampleIdentify:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = _mk(seed, difficulty)
            q2 = _mk(seed, difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, correct_index).correct
            for dist in q.distractors:
                dist_index = options.index(dist)
                assert not grade_mcq(q, dist_index).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_a_deliberately_wrong_index(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            wrong_index = (correct_index + 1) % len(options)
            assert not grade_mcq(q, wrong_index).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_in_options(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            assert q.answer_text in mcq_options(q)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions(self, difficulty):
        texts = {_mk(seed, difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_set_notation_not_python_list(self, difficulty):
        for seed in range(10):
            q = _mk(seed, difficulty)
            assert "[" not in q.text and "]" not in q.text

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-use-examples-and-counter-examples"), None)
        assert skill is not None
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "mcq"
        assert skill.prerequisites == ["f10-odd-even-properties"]
        assert skill.placement_eligible is False
        assert len([s for s in skill.sources if s.difficulty != "exam"]) == 3


# ===========================================================================
# classify variant (easy/medium): construct-to-target 50/50 measurement
# ===========================================================================


class TestClassifyConstructToTarget:
    @pytest.mark.parametrize("difficulty,indices", [("easy", _EASY_INDICES), ("medium", _MEDIUM_INDICES)])
    def test_example_counterexample_roughly_50_50_over_100_seeds(self, difficulty, indices):
        counts = Counter()
        for seed in range(150):
            q = _mk(seed, difficulty)
            counts[q.answer_text] += 1
        assert set(counts) <= {"Example", "Counterexample"}
        assert len(counts) == 2, f"{difficulty}: only one label reached: {dict(counts)}"
        for label, n in counts.items():
            assert n >= 45, f"{difficulty}: label {label!r} only reached {n}/150 times: {dict(counts)}"

    @pytest.mark.parametrize("difficulty", ["easy", "medium"])
    def test_answer_is_example_or_counterexample_never_both_or_neither(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            assert q.answer_text in ("Example", "Counterexample")

    @pytest.mark.parametrize("difficulty", ["easy", "medium"])
    def test_distractor_universe_is_fixed_four_options(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            assert set(q.distractors) <= set(_CLASSIFY_OPTIONS)
            assert set([q.answer_text] + q.distractors) == set(_CLASSIFY_OPTIONS)

    def test_easy_uses_only_even_odd_predicates(self):
        easy_names = {_PREDICATES[i][0] for i in _EASY_INDICES}
        assert easy_names == {"is even", "is odd"}

    def test_medium_uses_the_computed_predicate_slice(self):
        medium_names = {_PREDICATES[i][0] for i in _MEDIUM_INDICES}
        assert medium_names == {"is a multiple of 3", "is a multiple of 5", "is greater than 20", "is a perfect square"}

    def test_classify_independent_check_fires_on_a_corrupted_check(self, monkeypatch):
        """Drives the ACTUAL `_classify` construction with an independent-check
        route deliberately wired to disagree with the construction route --
        its own `check(independent(c) == want_true, ...)` must raise."""
        monkeypatch.setattr(p9a_mod, "_INDEPENDENT_CHECKS", [lambda n: None] * len(_PREDICATES))
        raised = False
        for seed in range(20):
            try:
                _classify(random.Random(seed), _EASY_INDICES)
            except VerificationError:
                raised = True
                break
        assert raised, "no seed exercised the independent-check mismatch"

    def test_classify_pool_check_fires_when_domain_is_emptied(self, monkeypatch):
        """Drives the ACTUAL `_classify` construction's `check(pool, ...)` --
        an emptied `_DOMAIN` leaves no candidate for either target, so this
        must raise before any padding is attempted."""
        monkeypatch.setattr(p9a_mod, "_DOMAIN", range(0))
        with pytest.raises(VerificationError):
            _classify(random.Random(0), _EASY_INDICES)

    def test_classify_c_in_d_check_fires_when_padding_excludes_c(self, monkeypatch):
        """Drives the ACTUAL `_classify` construction's `check(c in D, ...)` --
        a `_pad_to_5` wired to return values that never include `c` must
        raise, since `c` is supposed to always be one of the displayed
        values."""
        monkeypatch.setattr(p9a_mod, "_pad_to_5", lambda c, rng: [100, 101, 102, 103, 104])
        with pytest.raises(VerificationError):
            _classify(random.Random(0), _EASY_INDICES)

    def test_classify_independent_check_uses_a_different_route_than_the_lambda(self):
        """The independent checks are not the same formula copy-pasted: confirm
        each one is a distinct callable object from its own predicate's lambda,
        and that they still agree pointwise over the full domain."""
        for idx, (name, fn, _route) in enumerate(_PREDICATES):
            independent = _INDEPENDENT_CHECKS[idx]
            assert independent is not fn
            for n in _DOMAIN:
                assert independent(n) == fn(n), f"{name} disagrees with its independent route at n={n}"


# ===========================================================================
# find_counterexample variant (hard)
# ===========================================================================


class TestFindCounterexample:
    def test_answer_is_the_failing_value_distractors_are_passing(self):
        for seed in SEEDS:
            q = _mk(seed, "hard")
            answer_n = int(q.answer_text)
            distractor_ns = [int(d) for d in q.distractors]
            idx = next(i for i, (name, _fn, _r) in enumerate(_PREDICATES) if name in q.text)
            p_fn = _PREDICATES[idx][1]
            assert p_fn(answer_n) is False
            for d in distractor_ns:
                assert p_fn(d) is True

    def test_all_six_predicates_reachable_over_60_seeds(self):
        seen = set()
        for seed in range(120):
            q = make(TOPIC, seed=seed, difficulty="hard", variant="find_counterexample")
            for name, _fn, _r in _PREDICATES:
                if name in q.text:
                    seen.add(name)
        assert len(seen) == 6, f"only {len(seen)}/6 predicates reachable over 120 seeds: {seen}"

    def test_find_counterexample_failing_check_fires_on_corrupted_route(self, monkeypatch):
        """Drives the ACTUAL `_find_counterexample` construction's
        `check(independent(failing_selected) is False, ...)`: an independent
        route wired to always return True disagrees with `failing_selected`
        (which the lambda says fails the predicate), so this must raise
        before the passing-value loop is even reached."""
        monkeypatch.setattr(p9a_mod, "_INDEPENDENT_CHECKS", [lambda n: True] * len(_PREDICATES))
        raised = False
        for seed in range(20):
            try:
                _find_counterexample(random.Random(seed))
            except VerificationError:
                raised = True
                break
        assert raised, "no seed exercised the independent-check mismatch"

    def test_find_counterexample_passing_loop_check_fires_on_corrupted_route(self, monkeypatch):
        """Drives the ACTUAL passing-value loop's
        `check(independent(value) is True, ...)`: an independent route wired
        to always return False agrees with `failing_selected` (both False,
        so the first check passes) but disagrees with every passing value,
        so this must raise -- a different code path from the failing-value
        check above."""
        monkeypatch.setattr(p9a_mod, "_INDEPENDENT_CHECKS", [lambda n: False] * len(_PREDICATES))
        raised = False
        for seed in range(20):
            try:
                _find_counterexample(random.Random(seed))
            except VerificationError:
                raised = True
                break
        assert raised, "no seed exercised the passing-loop independent-check mismatch"

    def test_pool_size_guard_fires_on_a_shrunk_domain(self, monkeypatch):
        """Drives the ACTUAL pool-size `check()` with a predicate that never
        passes -- `len(passing_pool) >= 3` must fail and raise."""
        monkeypatch.setattr(p9a_mod, "_PREDICATES", [("never true", lambda n: False, "n/a")])
        monkeypatch.setattr(p9a_mod, "_INDEPENDENT_CHECKS", [lambda n: False])
        with pytest.raises(VerificationError):
            _find_counterexample(random.Random(0))
