"""The parallelogram diagonal law (senior-the-sum-of-the-squares-of-the-lengths, ACMSM041)."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p11c_parallelogram_build import (
    _A_B_LIMIT,
    _RATIO_THRESH,
    _TABLE,
    angle_between_sides_deg,
    is_plausible,
    raw_quadruples,
)

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "p11c_parallelogram_diagonal_law"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)


# --- the plausibility test pair -----


def test_unconstrained_table_contains_implausible_quadruples():
    """RED-PROVING test: an UNCONSTRAINED search over the exact same integer space the generator's own table is
    built from (``1<=a<=b<=20``, exact solutions of the parallelogram law, primitive) contains quadruples that
    fail the plausibility gate -- both by angle (a parallelogram so acute or obtuse it reads as a sliver) and
    by the diagonal/side ratio (a parallelogram so thin a diagonal is indistinguishable from a side). This is
    the assertion that was run BEFORE the generator applied any filter and failed -- it is the proof that the
    constraint added afterwards was load-bearing, not decorative."""
    raw = raw_quadruples(_A_B_LIMIT)
    assert len(raw) > 0, "the raw search space itself is empty -- nothing to prove red against"

    bad_angle = [q for q in raw if not (20.0 <= angle_between_sides_deg(q[0], q[1], q[2]) <= 160.0)]
    thin_ratio = [q for q in raw if min(q[2], q[3]) <= _RATIO_THRESH * max(q[0], q[1])]
    implausible = [q for q in raw if not is_plausible(*q)]

    assert len(bad_angle) > 0, "expected the unconstrained draw to include an unreasonable-angle quadruple"
    assert len(thin_ratio) > 0, "expected the unconstrained draw to include a thin-diagonal quadruple"
    assert len(implausible) > 0, "expected the unconstrained draw to include at least one implausible quadruple"

    # The degenerate extreme actually present in this search space: a=1,
    # b=2, d1=1, d2=3 is COLLINEAR (theta=180 deg exactly) -- arithmetically
    # a perfect solution of the law, and not a parallelogram at all.
    assert (1, 2, 1, 3) in raw
    assert not is_plausible(1, 2, 1, 3)


def test_every_shipped_quadruple_is_plausible():
    """GREEN test: every quadruple the generator can actually draw from
    (``_TABLE``, after the angle and ratio filters and the scale-widening)
    passes the same plausibility gate the previous test just proved the raw
    search space does NOT satisfy on its own."""
    assert len(_TABLE) >= 20
    for a, b, d1, d2 in _TABLE:
        assert is_plausible(a, b, d1, d2), f"shipped quadruple {(a, b, d1, d2)} is implausible"
        theta = angle_between_sides_deg(a, b, d1)
        assert 20.0 <= theta <= 160.0
        assert min(d1, d2) > float(_RATIO_THRESH) * max(a, b)


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_generator_is_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        repeat = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85


def test_bare_call_default_has_enough_distractors():
    """Mirrors `test_all_topics_usually_have_enough_distractors`'s own bar
    (>=2 distractors on >=85% of seeds) for a bare `make(topic, seed=s)` call
    with no `difficulty` argument."""
    seeds_with_at_least_two = 0
    for seed in range(100):
        question = make(TOPIC, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 100 >= 0.85


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_option_formatted_alike(difficulty):
    """The answer and every distractor are plain non-negative integer
    strings, never a mix of decimal and integer forms (`midpoint_coordinate_
    scalar`'s defect class)."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        for text in [question.answer_text, *question.distractors]:
            assert text.isdigit(), f"{difficulty} seed={seed}: option {text!r} is not a plain integer"


# --- answer-kind and variety -------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_a_positive_integer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, int)
        assert question.answer > 0
        assert question.answer_text == str(question.answer)


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(difficulty):
    texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        wrong = question.answer + 1
        assert not grade_number(question, str(wrong)).correct
        wrong2 = question.answer * 2 + 7
        assert not grade_number(question, str(wrong2)).correct


# --- independent check -------------------------------------------------------


def test_independent_check_fires_on_corrupted_diagonal_law():
    from mathgen.generators.p11c_parallelogram_build import verify_diagonal_law

    # 5^2 + 5^2 = 2*(3^2 + 4^2) is a genuine solution of the law (3,4,5,5) --
    # must not raise.
    verify_diagonal_law(5, 5, 3, 4)
    with pytest.raises(VerificationError):
        verify_diagonal_law(5, 6, 3, 4)


# --- hard-tier unit conversion, genuinely exercised -------------------------


def test_hard_tier_side_in_metres_variant_states_a_metre_value():
    """At least one hard-tier draw over a reasonable sample states one side
    in metres (the ``side_in_metres`` variant) -- confirms the variant is
    actually reachable, not merely coded and never drawn."""
    found = False
    for seed in range(60):
        question = make(TOPIC, seed=seed, difficulty="hard")
        if " m long" in question.text:
            found = True
            break
    assert found, "hard tier never drew the side-in-metres variant over 60 seeds"


def test_hard_tier_answer_in_millimetres_variant_is_reachable():
    found = False
    for seed in range(60):
        question = make(TOPIC, seed=seed, difficulty="hard")
        if "millimetres" in question.text:
            found = True
            assert question.answer % 10 == 0, "an exact cm-to-mm conversion must land on a multiple of 10"
            break
    assert found, "hard tier never drew the answer-in-millimetres variant over 60 seeds"


def test_hard_tier_raises_on_unknown_tier():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="impossible")


# --- skill contract -----------------------------------------------------------


def test_p11c_parallelogram_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    skill = by_id["senior-the-sum-of-the-squares-of-the-lengths"]
    assert skill.prerequisites == ["convert-metric-length"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert skill.learner_band == "general"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in skill.sources if source.difficulty != "exam"} == {TOPIC}
    assert [source.stages for source in skill.sources] == [[6], [6], [6, "maintenance"], ["exam"]]
