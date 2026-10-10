"""Complex-plane transformation skills: senior-examine-and-use-multiplication-as-a-linear-transformation
(ACMSM085) and senior-examine-addition-of-complex-numbers-as-vector-addition (ACMSM072, ACMSM084)."""

from __future__ import annotations

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p9d_complex_plane_transform_build import (
    _addition_inverse,
    _check_complex_product,
    _check_complex_sum,
    _check_forward_sum,
    _sample_nonzero,
)

GRADER = GRADERS["pair"]
SEEDS = range(40)

MULT_TOPIC = "complex_mult_linear_transform"
ADD_TOPIC = "complex_addition_vector"


def _mk(topic, difficulty, seed):
    return make(topic, seed=seed, difficulty=difficulty)


class _ScriptedRng:
    """A fake `random.Random` that ignores `lo`/`hi` and yields a scripted
    sequence of `randint` results -- for proving a resampling loop is
    actually reachable and not dead code, rather than trusting that a random
    seed sweep happens to exercise it."""

    def __init__(self, values):
        self._values = list(values)

    def randint(self, lo, hi):
        return self._values.pop(0)


# ---------------------------------------------------------------------------
# Shared-shape checks, parametrized over both topics/skills
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", [MULT_TOPIC, ADD_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
class TestSharedShape:
    def test_determinism(self, topic, difficulty):
        for seed in SEEDS:
            q1 = _mk(topic, difficulty, seed)
            q2 = _mk(topic, difficulty, seed)
            assert q1.to_json() == q2.to_json()

    def test_answer_is_dict_pair_of_ints(self, topic, difficulty):
        for seed in SEEDS:
            q = _mk(topic, difficulty, seed)
            assert isinstance(q.answer, dict)
            assert len(q.answer) == 2
            for v in q.answer.values():
                assert isinstance(v, int) or (isinstance(v, float) and v == int(v))

    def test_distractors_nonempty_distinct_never_equal_answer(self, topic, difficulty):
        for seed in SEEDS:
            q = _mk(topic, difficulty, seed)
            assert len(q.distractors) >= 2, f"seed {seed}: too few distractors: {q.distractors}"
            assert len(set(q.distractors)) == len(q.distractors), f"seed {seed}: duplicate distractors: {q.distractors}"
            assert q.answer_text not in q.distractors, f"seed {seed}: a distractor equals the answer: {q.distractors}"

    def test_real_grader_accepts_canonical(self, topic, difficulty):
        for seed in SEEDS:
            q = _mk(topic, difficulty, seed)
            result = GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected canonical {q.answer_text!r}"

    def test_real_grader_rejects_distractors(self, topic, difficulty):
        for seed in SEEDS:
            q = _mk(topic, difficulty, seed)
            for distractor in q.distractors:
                result = GRADER(q, distractor)
                assert not result.correct, f"seed {seed}: grader accepted distractor {distractor!r}"

    def test_real_grader_rejects_deliberately_wrong_value(self, topic, difficulty):
        for seed in SEEDS:
            q = _mk(topic, difficulty, seed)
            ax, ay = q.answer["x"], q.answer["y"]
            wrong = f"x = {int(ax) + 1000}, y = {int(ay) + 1000}"
            result = GRADER(q, wrong)
            assert not result.correct, f"seed {seed}: grader accepted a wildly wrong value"

    def test_answer_and_distractors_formatted_alike(self, topic, difficulty):
        """No option is identifiable by shape alone: every option (answer and
        every distractor) is an `xy_solution`-shaped "x = <int>, y = <int>" string."""
        for seed in SEEDS:
            q = _mk(topic, difficulty, seed)
            for option in [q.answer_text] + q.distractors:
                assert option.startswith("x = ") and ", y = " in option, option

    def test_at_least_20_distinct_questions_over_60_seeds(self, topic, difficulty):
        texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in range(60)}
        assert len(texts) >= 20, f"only {len(texts)} distinct questions: {texts}"


@pytest.mark.parametrize("topic", [MULT_TOPIC, ADD_TOPIC])
def test_skill_loads_through_api(topic):
    """Both skills load through mathgen.skills.load_skills with the expected shape."""
    skill_id = {
        MULT_TOPIC: "senior-examine-and-use-multiplication-as-a-linear-transformation",
        ADD_TOPIC: "senior-examine-addition-of-complex-numbers-as-vector-addition",
    }[topic]
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == skill_id), None)
    assert skill is not None, f"{skill_id} not found in skills library"
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    stages_seen = set()
    for source in skill.sources:
        stages_seen.update(source.stages)
    assert {0, 1, 2, 3, 4, 5, 6, "maintenance"} <= stages_seen


def test_mult_skill_has_no_prerequisites():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "senior-examine-and-use-multiplication-as-a-linear-transformation")
    assert skill.prerequisites == []


def test_addition_skill_has_vector_addition_prerequisite():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "senior-examine-addition-of-complex-numbers-as-vector-addition")
    assert skill.prerequisites == ["vector-addition"]


# ---------------------------------------------------------------------------
# Skill A: multiplication as a linear transformation
# ---------------------------------------------------------------------------


class TestMultLinearTransform:
    def test_easy_multiplier_is_real_never_unit_or_zero(self):
        for seed in range(100):
            q = make(MULT_TOPIC, seed=seed, difficulty="easy")
            # multiplier k appears right after "w = " and before " is a linear"
            k_str = q.text.split("w = ")[1].split(" is a linear")[0]
            k = int(k_str)
            assert k not in (-1, 0, 1)
            assert -6 <= k <= 6

    def test_medium_multiplier_is_purely_imaginary(self):
        for seed in range(100):
            q = make(MULT_TOPIC, seed=seed, difficulty="medium")
            assert "purely imaginary number w = " in q.text
            m_str = q.text.split("w = ")[1].split("i is a linear")[0]
            m = int(m_str)
            assert m not in (-1, 0, 1)
            assert -6 <= m <= 6

    def test_hard_multiplier_has_both_components_nonzero(self):
        """a, b each drawn from {-5..-1, 1..5} -- neither is ever zero, so the
        multiplier text is always a genuine two-term "a + bi"/"a - bi" (never
        a bare real number or a bare "bi")."""
        for seed in range(100):
            q = make(MULT_TOPIC, seed=seed, difficulty="hard")
            w_str = q.text.split("w = ")[1].split(" is a linear")[0]
            assert " + " in w_str or " - " in w_str, w_str
            assert w_str.endswith("i"), w_str

    def test_point_never_the_origin(self):
        for difficulty in ["easy", "medium", "hard"]:
            for seed in range(100):
                q = make(MULT_TOPIC, seed=seed, difficulty=difficulty)
                z_str = q.text.split("z = ")[1].split(" under this")[0]
                assert z_str != "0"

    def test_independent_check_fires_on_corrupted_answer(self):
        _check_complex_product(complex(2, 0), complex(3, 4), 6, 8)  # must not raise
        with pytest.raises(VerificationError):
            _check_complex_product(complex(2, 0), complex(3, 4), 6, 9)

    def test_independent_check_agrees_by_hand(self):
        # w = 3+4i, z = 1+2i -> (3+4i)(1+2i) = 3+6i+4i+8i^2 = 3+10i-8 = -5+10i
        _check_complex_product(complex(3, 4), complex(1, 2), -5, 10)
        with pytest.raises(VerificationError):
            _check_complex_product(complex(3, 4), complex(1, 2), -5, 11)


# ---------------------------------------------------------------------------
# Skill B: addition of complex numbers as vector addition
# ---------------------------------------------------------------------------


class TestAdditionVector:
    def test_every_question_uses_vector_and_resultant_wording(self):
        for difficulty in ["easy", "medium", "hard"]:
            for seed in range(60):
                q = make(ADD_TOPIC, seed=seed, difficulty=difficulty)
                assert "vector" in q.text.lower(), q.text
                assert "resultant" in q.text.lower(), q.text

    def test_easy_second_vector_never_zero(self):
        for seed in range(200):
            q = make(ADD_TOPIC, seed=seed, difficulty="easy")
            z2_str = q.text.split("z2 = ")[1].split(" by the vector")[0]
            assert z2_str != "0"

    def test_chained_all_three_vectors_never_zero(self):
        for seed in range(200):
            q = make(ADD_TOPIC, seed=seed, difficulty="medium")
            for label, stop in [("z1 = ", ", z2"), ("z2 = ", " and z3"), ("z3 = ", ", each")]:
                value = q.text.split(label)[1].split(stop)[0]
                assert value != "0"

    def test_inverse_resultant_never_equals_z1(self):
        for seed in range(200):
            q = make(ADD_TOPIC, seed=seed, difficulty="hard")
            ax, ay = q.answer["x"], q.answer["y"]
            # z2 = R - z1; if R == z1 then z2 would be (0, 0), which the
            # generator's resample loop forbids by construction.
            assert (ax, ay) != (0, 0)

    def test_independent_check_sum_fires_on_corrupted_answer(self):
        _check_complex_sum([complex(1, 2), complex(3, 4)], 4, 6)  # must not raise
        with pytest.raises(VerificationError):
            _check_complex_sum([complex(1, 2), complex(3, 4)], 4, 7)

    def test_independent_check_forward_sum_fires_on_corrupted_answer(self):
        _check_forward_sum(complex(1, 2), complex(3, 4), 4, 6)  # must not raise
        with pytest.raises(VerificationError):
            _check_forward_sum(complex(1, 2), complex(3, 4), 4, 7)

    def test_independent_check_agrees_by_hand_chained(self):
        # (1+2i) + (3-4i) + (-2+5i) = 2+3i
        _check_complex_sum([complex(1, 2), complex(3, -4), complex(-2, 5)], 2, 3)
        with pytest.raises(VerificationError):
            _check_complex_sum([complex(1, 2), complex(3, -4), complex(-2, 5)], 2, 4)

    def test_inverse_tier_resample_loop_is_reachable_not_dead_code(self):
        """Script a resultant R equal to z1 on the first draw -- the
        `while (rx, ry) == (x1, y1)` resample loop must fire and the final
        R must differ from z1, proving the guard is live, not dead code."""
        rng = _ScriptedRng(
            [
                3,
                4,  # z1 = (3, 4)
                3,
                4,  # first draw of R collides with z1
                5,
                6,  # resampled R = (5, 6), distinct
            ]
        )
        q = _addition_inverse(rng)
        assert q["answer"] == {"x": 2, "y": 2}  # z2 = R - z1 = (5-3, 6-4)


# ---------------------------------------------------------------------------
# Degeneracy guards -- proven reachable with a scripted RNG, not just
# trusted to hold "usually" over a random seed sweep.
# ---------------------------------------------------------------------------


def test_sample_nonzero_resample_loop_is_reachable_not_dead_code():
    """Script the origin on the first draw -- the resample loop inside
    `_sample_nonzero` (shared by every tier's point/vector sampling) must
    fire and the final point must not be the origin."""
    rng = _ScriptedRng([0, 0, 3, 4])
    assert _sample_nonzero(rng) == (3, 4)
