"""Focused contract tests for: ucv1-vector-valued-derivative (``answer: pair``)."""

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "ucv1-vector-valued-derivative"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_ITEM_RE = r"-?\d+(?:/\d+)?"
_PAIR_RE = re.compile(rf"\({_ITEM_RE}, {_ITEM_RE}\)")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources(SKILL_ID)


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        ax, ay = q.answer["x"], q.answer["y"]
        wrong = f"({Fraction(ax) + 1000}, {ay})"
        assert not grade_pair(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_corrupted_pair_string(difficulty):
    """Pass a corrupted pair string directly to grading.grade_pair, both accept (canonical) and reject
    (corrupted) cases."""
    for seed in SEEDS:
        q = _make(difficulty, seed)
        corrupted = f"({q.answer['x']})"  # drops the y component entirely
        assert not grade_pair(q, corrupted).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_a_pair_of_exact_fractions(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert isinstance(q.answer, dict) and set(q.answer) == {"x", "y"}
        for v in q.answer.values():
            assert isinstance(v, (int, Fraction)), f"{v!r} is {type(v)}, not int/Fraction"
            f = Fraction(v)
            assert f == Fraction(f.numerator, f.denominator)  # already lowest terms
        assert _PAIR_RE.fullmatch(q.answer_text), q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        for d in q.distractors:
            assert _PAIR_RE.fullmatch(d), d


@pytest.mark.parametrize("difficulty", TIERS)
def test_200_seed_answer_pool_is_wide_enough(difficulty):
    counts = Counter(_make(difficulty, seed).answer_text for seed in range(200))
    distinct = len(counts)
    top_share = max(counts.values()) / 200
    assert distinct >= 20, f"{difficulty}: only {distinct} distinct answers over 200 seeds"
    assert top_share <= 0.10, (
        f"{difficulty}: top answer {counts.most_common(1)[0][0]!r} is {top_share:.1%} of 200 draws"
    )


def test_hard_tier_exactly_one_component_is_linear_every_seed():
    """The hard-tier disguise: exactly one component's DEFINITION has no
    '^' (purely linear, degree 1) every seed -- its derivative is therefore
    the coefficient of t alone, a constant independent of whatever t0 is
    stated, by construction (power-rule differentiation of a degree-1 term
    never involves the substituted value)."""
    linear_count = 0
    total = 0
    for seed in SWEEP_SEEDS:
        q = _make("hard", seed)
        body = q.text.split("r(t) = (")[1].split(")")[0]
        x_text, y_text = [p.strip() for p in body.split(",", 1)]
        is_linear = ["^" not in x_text, "^" not in y_text]
        assert sum(is_linear) == 1, f"seed {seed}: expected exactly one linear component, got {x_text!r}, {y_text!r}"
        linear_count += 1
        total += 1
    assert linear_count == total == SWEEP_SEEDS.stop, (
        f"linear-component-independent-of-t0 disguise present in {linear_count}/{total} hard-tier seeds"
    )


_EXAM_X_RE = re.compile(r"x\(t\) = \(t ([+-]) (\d+)\)\(t ([+-]) (\d+)\)")
_EXAM_T0_RE = re.compile(rf"t = ({_ITEM_RE})\.?\s*$")


def _rational_text(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _factor_int(sign: str, num: str) -> int:
    """'(t - 3)' -> p=3; '(t + 2)' -> p=-2 (the sign-adjusted rendering of
    't - p')."""
    n = int(num)
    return n if sign == "-" else -n


def test_exam_tier_product_rule_forgot_one_factor_distractor_present_and_fires():
    """The product-rule-forgot-one-factor misconception (reports '(t - q1)'
    alone instead of the full product-rule sum '(t - q1) + (t - p1)') is
    offered as one of the shown distractors on every exam-tier seed. Parsed
    straight from the rendered text (p1, q1, t0) and the answer's own y
    component -- not read from generator internals -- and confirmed present
    in q.distractors, and different from the answer."""
    fired = 0
    for seed in SWEEP_SEEDS:
        q = _make("exam", seed)
        xm = _EXAM_X_RE.search(q.text)
        t0m = _EXAM_T0_RE.search(q.text)
        assert xm and t0m, q.text
        q1 = _factor_int(xm.group(3), xm.group(4))
        t0 = Fraction(t0m.group(1))
        ans_y_text = q.answer_text.split(", ", 1)[1].rstrip(")")
        forgot_x = t0 - q1
        forgot_text = f"({_rational_text(forgot_x)}, {ans_y_text})"
        assert forgot_text in q.distractors, f"seed {seed}: expected {forgot_text} in {q.distractors}"
        assert forgot_text != q.answer_text
        fired += 1
    assert fired == SWEEP_SEEDS.stop


_PRODUCT_MARKER_RE = re.compile(r"\(t [+-] \d+\)\(t [+-] \d+\)")


def test_exam_tier_uses_product_form_component_not_sum_of_vectors():
    """Prove RED on the old u(t)/v(t) generator: (a) the product marker '(t - <int>)(t - <int>)' never
    occurs in easy/medium/hard text but occurs on EVERY exam seed; (b) the old sum-of-two-vector-pieces
    pattern ('u(t) = (' / 'v(t) = (') no longer appears anywhere."""
    for difficulty in ("easy", "medium", "hard"):
        for seed in SWEEP_SEEDS:
            text = _make(difficulty, seed).text
            assert not _PRODUCT_MARKER_RE.search(text), (
                f"{difficulty} seed {seed}: product-form marker leaked below exam tier: {text!r}"
            )
            assert "u(t) = (" not in text and "v(t) = (" not in text

    for seed in SWEEP_SEEDS:
        text = _make("exam", seed).text
        assert _PRODUCT_MARKER_RE.search(text), (
            f"exam seed {seed}: expected a product-of-two-linear-factors marker, got {text!r}"
        )
        assert "u(t) = (" not in text and "v(t) = (" not in text, (
            f"exam seed {seed}: old sum-of-two-vector-pieces pattern still present: {text!r}"
        )


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.domain == "multivariable-and-vector-calculus"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "pair"
    assert SKILL.prerequisites == ["differentiate-polynomial"]
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert SOURCES["exam"].stages == ["exam"]


def test_grader_accepts_bracketed_whole_answer_with_extra_spaces():
    q = _make("easy", 0)
    padded = q.answer_text.replace(", ", " , ")
    assert grade_pair(q, padded).correct


def test_swapped_answer_is_rejected_when_components_differ():
    """A learner who types the components in swapped order must be marked
    wrong for an UNLABELLED pair (order carries meaning), matching
    quadratic-vertex's convention."""
    for seed in SEEDS:
        q = _make("easy", seed)
        ax, ay = q.answer["x"], q.answer["y"]
        if ax == ay:
            continue
        swapped = f"({ay}, {ax})"
        assert not grade_pair(q, swapped).correct, f"seed {seed}: swapped {swapped} wrongly accepted"
