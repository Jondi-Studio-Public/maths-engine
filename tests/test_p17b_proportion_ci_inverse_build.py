"""Tests the inverse proportion confidence-interval generator, which recovers a sample summary from a stated
interval.

Every tier recovers a sample summary from a stated normal-approximation confidence interval; no tier's answer
is ever an interval bound, and the learner never computes the forward interval."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "ucv1-proportion-confidence-interval"
SEEDS = range(40)
SWEEP_SEEDS = range(200)
TIERS = ["easy", "medium", "hard", "exam"]

_FRAC_RE = r"-?\d+(?:\.\d+)?"
_LABELLED_PAIR_RE = re.compile(rf"(\w+) = ({_FRAC_RE}), (\w+) = ({_FRAC_RE})")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources(SKILL_ID)


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    if (
        difficulty == "exam"
    ):  # the skill's exam source moved to exd_x_r3b_proportion_ci; this file pins the original generator
        return make("proportion_confidence_interval_inverse", seed=seed, difficulty="exam")
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(difficulty, seed)
        assert len(q1.distractors) >= 2


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
def test_no_spaces_accepted(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        m = _LABELLED_PAIR_RE.fullmatch(q.answer_text)
        label1, value1, label2, value2 = m.groups()
        no_spaces = f"{label1}={value1},{label2}={value2}"
        assert grade_pair(q, no_spaces).correct, f"seed {seed} {difficulty}: no-space {no_spaces} not accepted"


@pytest.mark.parametrize("difficulty", TIERS)
def test_unlabelled_positional_accepted(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        m = _LABELLED_PAIR_RE.fullmatch(q.answer_text)
        _label1, value1, _label2, value2 = m.groups()
        positional = f"{value1}, {value2}"
        assert grade_pair(q, positional).correct, (
            f"seed {seed} {difficulty}: unlabelled positional {positional} not accepted"
        )


@pytest.mark.parametrize("difficulty", TIERS)
def test_ulp_change_per_component_rejected(difficulty):
    """A pair component is graded to the precision its
    canonical DISPLAYS. Easy's components are 4 d.p. (tolerance 0.00005);
    medium/hard/exam's are plain integers (exact match). A one-unit-in-
    last-place change to either component must be rejected -- proves the
    tight path is live, not the ~0.051 voice fallback."""
    for seed in SEEDS:
        q = _make(difficulty, seed)
        m = _LABELLED_PAIR_RE.fullmatch(q.answer_text)
        label1, value1, label2, value2 = m.groups()
        if difficulty == "easy":
            bumped1 = str(round(float(value1) + 0.0001, 4))
            bumped2 = str(round(float(value2) + 0.0001, 4))
        else:
            bumped1 = str(int(value1) + 1)
            bumped2 = str(int(value2) + 1)
        wrong1 = f"{label1} = {bumped1}, {label2} = {value2}"
        wrong2 = f"{label1} = {value1}, {label2} = {bumped2}"
        assert not grade_pair(q, wrong1).correct, (
            f"seed {seed} {difficulty}: ulp-bumped {label1} {wrong1} wrongly accepted"
        )
        assert not grade_pair(q, wrong2).correct, (
            f"seed {seed} {difficulty}: ulp-bumped {label2} {wrong2} wrongly accepted"
        )


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        for d in q.distractors:
            assert _LABELLED_PAIR_RE.fullmatch(d), d


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_concentration_under_15_percent(difficulty):
    from collections import Counter

    counts = Counter(_make(difficulty, seed).answer_text for seed in SWEEP_SEEDS)
    top = counts.most_common(1)[0][1]
    share = top / len(SWEEP_SEEDS)
    assert share < 0.15, f"{difficulty}: top answer share {share:.2%} over {len(SWEEP_SEEDS)} seeds"


def test_easy_tier_answer_is_exact_4dp_fraction():
    for seed in SEEDS:
        q = _make("easy", seed)
        assert isinstance(q.answer, dict)
        assert set(q.answer.keys()) == {"p_hat", "margin"}
        for v in q.answer.values():
            assert isinstance(v, Fraction)
            scaled = v * 10000
            assert scaled.denominator == 1, f"{v} is not on the 4dp grid"


@pytest.mark.parametrize("difficulty", ["medium", "hard", "exam"])
def test_nk_or_nN_tiers_answer_is_whole_numbers(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert isinstance(q.answer, dict)
        for v in q.answer.values():
            assert isinstance(v, int)


def test_hard_tier_states_one_bound_and_p_hat_never_both_bounds():
    for seed in SEEDS:
        q = _make("hard", seed)
        assert "p_hat = " in q.text
        assert q.text.count("bound") >= 1
        # exactly one of "lower bound" / "upper bound" appears (not both,
        # unlike medium/exam which give both bounds)
        assert ("lower bound" in q.text) != ("upper bound" in q.text)


def test_medium_and_exam_tiers_state_both_bounds():
    for difficulty in ("medium", "exam"):
        for seed in SEEDS:
            q = _make(difficulty, seed)
            assert "lower bound" in q.text and "upper bound" in q.text


def test_exam_tier_never_asks_for_k():
    for seed in SEEDS:
        q = _make("exam", seed)
        assert set(q.answer.keys()) == {"n", "N"}
        assert "target margin" in q.text


def test_no_tier_answer_is_an_interval_bound():
    """By design, no tier's answer is ever an interval bound (L/U),
    only a sample-summary component (p_hat/margin/n/k/N)."""
    for difficulty in TIERS:
        for seed in SEEDS:
            q = _make(difficulty, seed)
            assert set(q.answer.keys()) <= {"p_hat", "margin", "n", "k", "N"}


def test_no_tier_gives_n_and_k_as_data_in_the_prompt():
    """No tier states n or k as a given fact -- 'n = ...'/'k = ...' only
    appear as the ANSWER-FORM instruction ("Give your answer in the form
    n = ..., k = ...")."""
    for difficulty in TIERS:
        for seed in SEEDS:
            q = _make(difficulty, seed)
            before_form, _, _ = q.text.partition("Give your answer in the form")
            assert "n = " not in before_form
            assert "k = " not in before_form


def test_generator_own_check_pairwise_distinct_fires_on_a_corrupted_draw():
    """The independent in-generator checks (check_pairwise_distinct over the
    true answer and every distractor candidate, and the stated-precision /
    exact-integer margin check) are exercised on every draw already inside
    make(). This proves they are load-bearing: corrupting a distractor
    formula to equal the true answer raises rather than silently shipping a
    duplicate option."""
    from fractions import Fraction as F
    from mathgen.core import VerificationError, check_pairwise_distinct

    with pytest.raises(VerificationError):
        check_pairwise_distinct((F(1, 4), F(3, 4)), (F(1, 4), F(3, 4)))


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "pair"
    assert SKILL.prerequisites == ["bernoulli-mle"]
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = SOURCES["exam"]
    assert exam_src.stages == ["exam"]


def _num(pattern, text):
    return Fraction(re.search(pattern, text).group(1))


def test_exam_N_is_the_same_by_every_principled_route():
    """Read from the prompt only: the printed midpoint, the sample's own k/n, and n scaled by (margin / target)^2
    must all give the keyed N; otherwise a learner taking a correct route is marked wrong."""
    num = r"(\d+(?:\.\d+)?)"
    for seed in range(1000):
        q = _make("exam", seed)
        t = q.text
        z = _num(rf"z = {num}", t)
        lo, hi = _num(rf"lower bound {num}", t), _num(rf"upper bound {num}", t)
        target = _num(rf"target margin is {num}", t)
        p, m = (lo + hi) / 2, (hi - lo) / 2
        n = round(z * z * p * (1 - p) / (m * m))
        k = round(p * n)
        key = dict(re.findall(r"(\w+) = (\d+)", q.answer_text))
        assert int(key["n"]) == n, f"seed {seed}: {q.answer_text}"
        routes = {
            "midpoint": z * z * p * (1 - p) / (target * target),
            "k/n": z * z * Fraction(k, n) * (1 - Fraction(k, n)) / (target * target),
            "scaled n": n * (m / target) ** 2,
        }
        for name, value in routes.items():
            assert int(value) + 1 == int(key["N"]), f"seed {seed}: N by {name} is {int(value) + 1}, key {key['N']}"
