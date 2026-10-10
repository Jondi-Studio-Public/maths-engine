"""Unit-circle tan evaluation and the R cos(x-alpha) auxiliary-angle conversion.

Covers both skills built by mathgen/generators/p7a_unit_circle_aux_angle_build.py:
- senior-understand-the-unit-circle-definition-of-cos-theta-acmmm034 (ACMMM034, mcq)
- senior-convert-sums-mathrm-a-cos-mathrm-x-mathrm (ACMSM048, pair)

Generate through the skill's own `sources` params (never `make(topic, seed, difficulty=tier)` alone).
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import GRADER_TOLERANCE, VerificationError

SEEDS = range(40)
VARIETY_SEEDS = range(60)

TOPIC_A = "unit_circle_tan_evaluate"
TOPIC_B = "aux_angle_r_cos_conversion"

# (difficulty, params) exactly matching each skill's own `sources` entries.
TIERS_A = [
    ("easy", {"variant": "quadrant_one"}),
    ("medium", {"variant": "signed_quadrant"}),
    ("hard", {"variant": "mixed_with_quadrantal"}),
]
TIERS_B = [
    ("easy", {"variant": "cos_form_q1"}),
    ("medium", {"variant": "cos_form_mixed_sign"}),
    ("hard", {"variant": "sin_form_mixed_sign"}),
]


# ---------------------------------------------------------------------------
# Skill A: unit_circle_tan_evaluate (mcq)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty,params", TIERS_A)
def test_a_deterministic(difficulty, params):
    for seed in SEEDS:
        q1 = make(TOPIC_A, seed=seed, difficulty=difficulty, **params)
        q2 = make(TOPIC_A, seed=seed, difficulty=difficulty, **params)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty,params", TIERS_A)
def test_a_distractors_distinct_nonempty(difficulty, params):
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty, **params)
        assert len(q.distractors) == 3, f"seed={seed}: {q.distractors}"
        assert q.answer_text not in q.distractors
        assert len(set(q.distractors)) == 3, f"seed={seed}: duplicate distractors {q.distractors}"


@pytest.mark.parametrize("difficulty,params", TIERS_A)
def test_a_real_grader_accepts_answer_rejects_distractors(difficulty, params):
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty, **params)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        result = GRADERS["mcq"](q, str(correct_idx))
        assert result.correct, f"seed={seed}: grader rejected canonical answer {q.answer_text!r}"
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            wrong = GRADERS["mcq"](q, str(i))
            assert not wrong.correct, f"seed={seed}: grader accepted distractor option {opt!r}"


@pytest.mark.parametrize("difficulty,params", TIERS_A)
def test_a_deliberately_wrong_answer_rejected(difficulty, params):
    """An index that names neither the answer nor a distractor is rejected."""
    q = make(TOPIC_A, seed=0, difficulty=difficulty, **params)
    result = GRADERS["mcq"](q, "99")
    assert not result.correct


def test_a_independent_check_fires_on_corrupted_construction():
    """The generator's own Pythagorean/undefined-gate check raises on a bad point."""
    from mathgen.core import check

    with pytest.raises(VerificationError):
        check(3 * 3 + 4 * 4 == 6 * 6, "point is not on the unit circle")
    with pytest.raises(VerificationError):
        check(0 != 0, "quadrantal question: cos(theta) is zero")


@pytest.mark.parametrize("difficulty,params", TIERS_A)
def test_a_answer_satisfies_its_own_kind(difficulty, params):
    """Every option is either a reduced p/q fraction, a bare integer, or 'undefined'."""
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty, **params)
        for opt in [q.answer_text, *q.distractors]:
            if opt == "undefined":
                continue
            if "/" in opt:
                num, den = opt.split("/")
                import math

                assert math.gcd(abs(int(num)), int(den)) == 1, f"{opt} not in lowest terms"
            else:
                int(opt)  # bare integer distractor (quadrantal branch)


@pytest.mark.parametrize("difficulty,params", TIERS_A)
def test_a_at_least_20_distinct_questions_per_tier(difficulty, params):
    texts = {make(TOPIC_A, seed=seed, difficulty=difficulty, **params).text for seed in VARIETY_SEEDS}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"


def test_a_hard_tier_hits_undefined_more_than_once_in_60_seeds():
    count = sum(
        1
        for seed in VARIETY_SEEDS
        if make(TOPIC_A, seed=seed, difficulty="hard", variant="mixed_with_quadrantal").answer_text == "undefined"
    )
    assert count >= 3, f"only {count}/60 hard-tier questions were 'undefined'"


def test_a_undefined_appears_as_a_distractor_too():
    """'undefined' must not be shape-distinctive: it appears as a wrong option
    on some hard-tier questions, not only ever as the correct answer."""
    found = any(
        "undefined" in make(TOPIC_A, seed=seed, difficulty="hard", variant="mixed_with_quadrantal").distractors
        for seed in VARIETY_SEEDS
    )
    assert found, "'undefined' never appears as a distractor over 60 hard-tier seeds"


@pytest.mark.parametrize("difficulty,params", TIERS_A)
def test_a_options_formatted_alike_within_a_question(difficulty, params):
    import re

    frac_re = re.compile(r"^-?\d+/\d+$")
    int_re = re.compile(r"^-?\d+$")
    for seed in VARIETY_SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty, **params)
        opts = [q.answer_text, *q.distractors]
        shapes = set()
        for o in opts:
            if o == "undefined":
                shapes.add("undefined")
            elif frac_re.match(o):
                shapes.add("fraction")
            elif int_re.match(o):
                shapes.add("integer")
            else:
                shapes.add("other")
        # A single-shape question is fine; a lone-outlier-of-one-shape among
        # otherwise-uniform options is the defect this asserts against.
        if "undefined" in shapes:
            # quadrantal branch: 0/1/-1/undefined is the whole fixed pool by design
            assert shapes <= {"undefined", "integer"}, f"seed={seed}: {opts}"
        else:
            assert shapes == {"fraction"}, f"seed={seed}: {opts}"


# ---------------------------------------------------------------------------
# Skill B: aux_angle_r_cos_conversion (pair)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty,params", TIERS_B)
def test_b_deterministic(difficulty, params):
    for seed in SEEDS:
        q1 = make(TOPIC_B, seed=seed, difficulty=difficulty, **params)
        q2 = make(TOPIC_B, seed=seed, difficulty=difficulty, **params)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty,params", TIERS_B)
def test_b_distractors_distinct_nonempty(difficulty, params):
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty, **params)
        assert len(q.distractors) == 3, f"seed={seed}: {q.distractors}"
        assert q.answer_text not in q.distractors
        assert len(set(q.distractors)) == 3, f"seed={seed}: duplicate distractors {q.distractors}"


@pytest.mark.parametrize("difficulty,params", TIERS_B)
def test_b_real_grader_accepts_answer_rejects_distractors(difficulty, params):
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty, **params)
        result = GRADERS["pair"](q, q.answer_text)
        assert result.correct, f"seed={seed}: grader rejected canonical {q.answer_text!r}"
        for d in q.distractors:
            wrong = GRADERS["pair"](q, d)
            assert not wrong.correct, f"seed={seed}: grader accepted distractor {d!r}"


@pytest.mark.parametrize("difficulty,params", TIERS_B)
def test_b_deliberately_wrong_answer_rejected(difficulty, params):
    q = make(TOPIC_B, seed=0, difficulty=difficulty, **params)
    result = GRADERS["pair"](q, "R = -999, alpha = 179.9")
    assert not result.correct


def test_b_grader_check_is_first_thing_done_for_every_variant_seed0():
    """Confirm the real pair grader accepts seed 0 for all
    three variants -- the cheapest possible check, run explicitly here."""
    for difficulty, params in TIERS_B:
        q = make(TOPIC_B, seed=0, difficulty=difficulty, **params)
        assert GRADERS["pair"](q, q.answer_text).correct


def test_b_independent_check_fires_on_corrupted_construction():
    from mathgen.core import check

    with pytest.raises(VerificationError):
        check(abs(5 * 0.6 - 999) < 1e-6, "R*cos(alpha) does not reconstruct a")


@pytest.mark.parametrize("difficulty,params", TIERS_B)
def test_b_answer_satisfies_its_own_kind(difficulty, params):
    """R is an integer; alpha carries exactly 1 decimal place and sits on the
    stated interval -180 < alpha <= 180."""
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty, **params)
        r_str, alpha_str = q.answer_text.split(", ")
        r_val = int(r_str.split(" = ")[1])
        alpha_val = alpha_str.split(" = ")[1]
        assert "." in alpha_val and len(alpha_val.split(".")[1]) == 1, alpha_val
        alpha_float = float(alpha_val)
        assert -180 < alpha_float <= 180, alpha_float
        assert r_val > 0


@pytest.mark.parametrize("difficulty,params", TIERS_B)
def test_b_at_least_20_distinct_questions_per_tier(difficulty, params):
    texts = {make(TOPIC_B, seed=seed, difficulty=difficulty, **params).text for seed in VARIETY_SEEDS}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"


@pytest.mark.parametrize("difficulty,params", TIERS_B)
def test_b_every_option_carries_the_same_label_shape(difficulty, params):
    for seed in VARIETY_SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty, **params)
        for opt in [q.answer_text, *q.distractors]:
            assert opt.startswith("R = ") and ", alpha = " in opt, opt


def test_b_naive_atan_trap_exercisability():
    """Documents, does not merely assert: how often the no-quadrant-correction
    distractor is a genuinely different value from the answer (it collides
    with the answer, and is dropped, whenever atan2 needs no correction)."""

    def naive_matches(difficulty, params, seed):
        import math

        q = make(TOPIC_B, seed=seed, difficulty=difficulty, **params)
        a = int(q.text.split("a = ")[1].split(" and")[0])
        b = int(q.text.split("b = ")[1].split(".")[0])
        form = "cos" if "R cos" in q.text else "sin"
        if form == "cos":
            naive = math.degrees(math.atan(b / a))
        else:
            naive = math.degrees(math.atan(a / b))
        alpha_true = float(q.answer_text.split("alpha = ")[1])
        return abs(round(naive, 1) - alpha_true) <= float(GRADER_TOLERANCE)

    medium_active = sum(not naive_matches("medium", TIERS_B[1][1], s) for s in VARIETY_SEEDS)
    hard_active = sum(not naive_matches("hard", TIERS_B[2][1], s) for s in VARIETY_SEEDS)
    # Exact combinatorial counts: medium 32/64, hard 64/96
    # of the full draw space. Over a 60-seed sample both must still be > 0.
    assert medium_active > 0, "naive-atan trap never fires on medium over 60 seeds"
    assert hard_active > 0, "naive-atan trap never fires on hard over 60 seeds"


# ---------------------------------------------------------------------------
# Skill loading (both)
# ---------------------------------------------------------------------------


def test_skill_a_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next(
        (s for s in skills if s.id == "senior-understand-the-unit-circle-definition-of-cos-theta-acmmm034"), None
    )
    assert skill is not None
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard"}


def test_skill_b_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-convert-sums-mathrm-a-cos-mathrm-x-mathrm"), None)
    assert skill is not None
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert "right-triangle-angle" in skill.prerequisites
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard"}
