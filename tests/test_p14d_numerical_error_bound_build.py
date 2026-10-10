"""Focused contract tests for: ucv1-numerical-error-bound (answer: pair, labelled, exact Fraction components)."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import GRADER_TOLERANCE

SKILL_ID = "ucv1-numerical-error-bound"
SEEDS = range(40)
SWEEP_SEEDS = range(200)
TIERS = ["easy", "medium", "hard", "exam"]

_FRAC_RE = r"-?\d+(?:/\d+)?"
_LABELLED_PAIR_RE = re.compile(rf"(\w+) = ({_FRAC_RE}), (\w+) = ({_FRAC_RE})")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources(SKILL_ID)


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _parse_frac(text):
    if "/" in text:
        p, q = text.split("/")
        return Fraction(int(p), int(q))
    return Fraction(int(text))


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(difficulty, seed)
        assert len(q1.distractors) >= 2


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    """Since both components are exact Fractions, a value clearing the pair
    grader's own tolerance in EITHER component must be rejected -- trivial to
    assert directly. NOTE: a "0.05 off" check was once expected; the real pair
    grader's tolerance (`mathgen.grading._PAIR_TOLERANCE` /
    `mathgen.core.GRADER_TOLERANCE`) is 0.051, strictly ABOVE 0.05, so an
    offset of exactly 0.05 is legitimately ACCEPTED by the real grader, not
    rejected -- that was true until the stated-precision fix;
    the block at the end now asserts the opposite.
    The margin actually asserted here (0.061, i.e. tolerance + 0.01) is what
    is required: a component change big enough to guarantee
    rejection."""
    OFFSET = GRADER_TOLERANCE + Fraction(1, 100)  # 0.061, clears the 0.051 tolerance
    assert Fraction(1, 20) <= GRADER_TOLERANCE  # documents the 0.05-is-accepted fact
    for seed in SEEDS:
        q = _make(difficulty, seed)
        labels = list(q.answer.keys())

        wrong_abs = q.answer[labels[0]] + OFFSET
        correct_rel = q.answer[labels[1]]
        wrong_text = f"{labels[0]} = {wrong_abs}, {labels[1]} = {correct_rel}"
        assert not grade_pair(q, wrong_text).correct, (
            f"seed {seed} {difficulty}: {OFFSET}-off absolute component {wrong_text} not rejected"
        )

        correct_abs = q.answer[labels[0]]
        wrong_rel = q.answer[labels[1]] + OFFSET
        wrong_text2 = f"{labels[0]} = {correct_abs}, {labels[1]} = {wrong_rel}"
        assert not grade_pair(q, wrong_text2).correct, (
            f"seed {seed} {difficulty}: {OFFSET}-off relative component {wrong_text2} not rejected"
        )

        # The literal 0.05-off case. This used to assert
        # it was ACCEPTED, pinning the flat 0.051 pair tolerance. Now
        # a pair component is graded to the precision its canonical
        # displays, and both components here display exactly (an integer and
        # an a/b fraction), so 0.05 off is wrong, as intended.
        near_wrong = q.answer[labels[0]] + Fraction(1, 20)
        near_text = f"{labels[0]} = {near_wrong}, {labels[1]} = {correct_rel}"
        assert not grade_pair(q, near_text).correct, (
            f"seed {seed} {difficulty}: the 0.05-off case {near_text} was accepted for an exact component"
        )


@pytest.mark.parametrize("difficulty", TIERS)
def test_labelled_order_independence_accepted(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        m = _LABELLED_PAIR_RE.fullmatch(q.answer_text)
        assert m, q.answer_text
        label1, value1, label2, value2 = m.groups()
        swapped = f"{label2} = {value2}, {label1} = {value1}"
        assert grade_pair(q, swapped).correct, f"seed {seed} {difficulty}: label-order-swapped {swapped} not accepted"


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_an_exact_fraction_pair_labelled_correctly(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert _LABELLED_PAIR_RE.fullmatch(q.answer_text), q.answer_text
        assert isinstance(q.answer, dict)
        assert set(q.answer.keys()) == {"absolute_bound", "relative_bound"}
        for v in q.answer.values():
            assert isinstance(v, Fraction)


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        for d in q.distractors:
            assert _LABELLED_PAIR_RE.fullmatch(d), d


def test_easy_tier_distances_to_both_endpoints_are_equal():
    """easy draws ahat exactly at the midpoint, so min(d_lo, d_hi) ==
    max(d_lo, d_hi) -- the 'uses-closer-endpoint' misconception collapses
    onto the correct answer and is therefore never offered on this tier
    (the generator's own candidate list for easy omits it entirely -- see
    its source)."""
    for seed in SEEDS:
        q = _make("easy", seed)
        m = re.search(r"ahat = (" + _FRAC_RE + r").*?\[(" + _FRAC_RE + r"), (" + _FRAC_RE + r")\]", q.text)
        assert m, q.text
        ahat, lo, hi = (_parse_frac(g) for g in m.groups())
        assert ahat - lo == hi - ahat


def test_medium_and_hard_tiers_exercise_the_closer_endpoint_distractor():
    """Regression guard for a previously fixed defect."""
    for difficulty in ("medium", "hard"):
        fired = 0
        for seed in SWEEP_SEEDS:
            q = _make(difficulty, seed)
            true_absolute = q.answer["absolute_bound"]
            for d in q.distractors:
                m = _LABELLED_PAIR_RE.fullmatch(d)
                if _parse_frac(m.group(2)) < true_absolute:
                    fired += 1
                    break
        assert fired > 0, f"{difficulty}: closer-endpoint distractor never fired over {len(SWEEP_SEEDS)} seeds"


def test_hard_tier_is_a_fraction_near_miss_not_an_integer_tier():
    for seed in SEEDS:
        q = _make("hard", seed)
        assert "/" in q.text or "/" in q.answer_text


def test_exam_tier_never_shows_a_bracket_and_translates_centre_radius():
    """Distinct from bisection-interval-update's '[left, right]' phrasing --
    the exam tier never shows a bracket at all, only a centre and a radius."""
    for seed in SEEDS:
        q = _make("exam", seed)
        assert "[" not in q.text and "]" not in q.text
        assert "certified to lie within" in q.text


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.domain == "numerical-and-computational-mathematics"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "pair"
    assert SKILL.prerequisites == []
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = SOURCES["exam"]
    assert exam_src.stages == ["exam"]


_CONVENTION_RE = re.compile(r"relative error bound is the absolute error bound divided by \|ahat\|")


@pytest.mark.parametrize("difficulty", TIERS)
def test_prompt_states_the_relative_error_bound_convention(difficulty):
    """'relative error bound' is ambiguous unless the prompt itself states which denominator it means (|ahat|,
    not the worst-case true value the textbook definition would use). Every tier's prompt -- across every
    wording-variant template -- must state the convention explicitly, in BOTH text and latex."""
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert _CONVENTION_RE.search(q.text), f"seed {seed} {difficulty}: convention clause missing from text: {q.text}"
        assert _CONVENTION_RE.search(q.latex), (
            f"seed {seed} {difficulty}: convention clause missing from latex: {q.latex}"
        )


def test_generator_own_check_pairwise_distinct_fires_on_a_corrupted_answer():
    """The independent in-generator check (check_equivalent between the
    direct max(...) route and the half-width+offset identity route) is
    exercised on every draw already inside make(); this test additionally
    proves the check is load-bearing by corrupting one input and confirming
    the module raises rather than silently accepting."""
    from mathgen.generators.p14d_numerical_error_bound_build import _bound
    from mathgen.core import VerificationError

    ahat, lo, hi = Fraction(5), Fraction(1), Fraction(9)
    absolute_bound, relative_bound, d_lo, d_hi = _bound(ahat, lo, hi)
    assert absolute_bound == max(ahat - lo, hi - ahat)
    assert relative_bound == absolute_bound / abs(ahat)

    with pytest.raises(VerificationError):
        _bound(Fraction(50), lo, hi)  # ahat not strictly inside [lo, hi]
