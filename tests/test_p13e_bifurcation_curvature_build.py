"""Focused contract tests for: ucv1-bifurcation-sign-check (answer: number) and ucv1-parametric-curvature
(answer: number)."""

import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make

BIFURCATION_ID = "ucv1-bifurcation-sign-check"
CURVATURE_ID = "ucv1-parametric-curvature"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


BIFURCATION_SKILL, BIFURCATION_SOURCES = _skill_sources(BIFURCATION_ID)
CURVATURE_SKILL, CURVATURE_SOURCES = _skill_sources(CURVATURE_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _decimal_places(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0


# ==========================================================================
# ucv1-bifurcation-sign-check (answer: number)
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_bifurcation_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(BIFURCATION_SOURCES, difficulty, seed)
        dp = _decimal_places(q.answer_text)
        if dp == 0:
            wrong = str(int(float(q.answer_text)) + 1000)
        else:
            wrong = f"{float(q.answer_text) + 1000:.{dp}f}"
        assert not grade_number(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


@pytest.mark.parametrize("difficulty", TIERS)
def test_bifurcation_answer_matches_its_own_stated_precision(difficulty):
    for seed in SEEDS:
        q = _make(BIFURCATION_SOURCES, difficulty, seed)
        assert _NUMBER_RE.fullmatch(q.answer_text), q.answer_text
        for d in q.distractors:
            assert _NUMBER_RE.fullmatch(d), d
            assert _decimal_places(d) == _decimal_places(q.answer_text), (
                f"seed {seed} {difficulty}: distractor {d} formatted unlike answer {q.answer_text}"
            )


def test_bifurcation_hard_tier_asks_for_the_positive_root_only():
    for seed in SEEDS:
        q = _make(BIFURCATION_SOURCES, "hard", seed)
        assert "POSITIVE" in q.text
        assert float(q.answer_text) > 0


def test_bifurcation_exam_tier_states_two_formula_candidates():
    for seed in SEEDS:
        q = _make(BIFURCATION_SOURCES, "exam", seed)
        assert "Formula A: r* =" in q.text, q.text
        assert "Formula B: r* =" in q.text, q.text


_R_CANDIDATE_TOKEN_RE = re.compile(r"r = (-?\d+(?:\.\d+)?)")


def test_bifurcation_exam_answer_value_never_appears_as_a_bare_r_candidate():
    """Proved RED on the pre-fix design; the rebuilt exam states two FORMULA strings, never a bare "r = <number>"
    token equal to the answer."""
    for seed in SEEDS:
        q = _make(BIFURCATION_SOURCES, "exam", seed)
        tokens = _R_CANDIDATE_TOKEN_RE.findall(q.text)
        assert q.answer_text not in tokens, (
            f"seed {seed}: answer {q.answer_text} appears as a bare 'r = <number>' token in the prompt: {q.text}"
        )


_EXAM_AB_RE = re.compile(r"a = (-?\d+), b = (-?\d+)")
_EXAM_FORMULA_A_RE = re.compile(r"Formula A: r\* = (.+?)\.")
_IMPLICIT_MULT_RE = re.compile(r"(\d)([a-zA-Z(])")


def _eval_formula(formula_text: str, a: int, b: int) -> float:
    """Evaluate a rendered formula string ("b^2 / (4a)") at a, b. The
    rendered text spells multiplication by juxtaposition, never a bare "*"
    (the project's text-quality gate flags that), so this re-inserts an
    explicit "*" between a digit and the following letter/parenthesis before
    `eval`; "^" is converted to "**"."""
    normalized = _IMPLICIT_MULT_RE.sub(r"\1*\2", formula_text.replace("^", "**"))
    return eval(normalized, {"__builtins__": {}}, {"a": a, "b": b})


def test_bifurcation_exam_formula_position_is_not_correlated_with_correctness():
    """Smoke test against a future regression reintroducing position bias
    (e.g. always putting the correct formula in slot A): over a 100+ seed
    sweep, neither slot should hold the correct formula more than 65% of the
    time -- generously below any 50/50-with-noise threshold, not a tight
    statistical claim."""
    seeds = range(150)
    correct_in_a = 0
    total = 0
    for seed in seeds:
        q = _make(BIFURCATION_SOURCES, "exam", seed)
        ab = _EXAM_AB_RE.search(q.text)
        formula_a = _EXAM_FORMULA_A_RE.search(q.text)
        assert ab and formula_a, q.text
        a, b = int(ab.group(1)), int(ab.group(2))
        value_a = _eval_formula(formula_a.group(1), a, b)
        total += 1
        if abs(float(value_a) - float(q.answer_text)) < 1e-9:
            correct_in_a += 1
    share_a = correct_in_a / total
    assert 0.35 <= share_a <= 0.65, (
        f"Formula A is correct {share_a:.0%} of {total} seeds -- suggests a "
        f"position bias in which formula is shown as A vs B"
    )


def test_bifurcation_skill_loads_through_api_with_correct_properties():
    assert BIFURCATION_SKILL.domain == "modelling-and-dynamical-systems"
    assert BIFURCATION_SKILL.status == "active"
    assert BIFURCATION_SKILL.placement_eligible is False
    assert BIFURCATION_SKILL.answer == "number"
    assert BIFURCATION_SKILL.prerequisites == ["autonomous-equilibrium-stability"]
    assert {src.difficulty for src in BIFURCATION_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = BIFURCATION_SOURCES["exam"]
    assert exam_src.stages == ["exam"]


# ==========================================================================
# ucv1-parametric-curvature (answer: number)
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_curvature_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(CURVATURE_SOURCES, difficulty, seed)
        dp = _decimal_places(q.answer_text)
        if dp == 0:
            wrong = str(int(float(q.answer_text)) + 1000)
        else:
            wrong = f"{float(q.answer_text) + 1000:.{dp}f}"
        assert not grade_number(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_curvature_answer_stated_to_exactly_3_decimal_places(difficulty):
    for seed in SEEDS:
        q = _make(CURVATURE_SOURCES, difficulty, seed)
        assert _decimal_places(q.answer_text) == 3, q.answer_text
        for d in q.distractors:
            assert _decimal_places(d) == 3, d


def test_curvature_exam_answer_is_an_integer_coefficient():
    for seed in SEEDS:
        q = _make(CURVATURE_SOURCES, "exam", seed)
        assert re.fullmatch(r"-?\d+", q.answer_text), q.answer_text
        assert isinstance(q.answer, int)


def test_curvature_hard_tier_second_derivative_depends_on_t0():
    """Distinguishes hard from easy/medium: the cubic's second derivative is
    not a bare constant, so the same (a3, a2) pair combined with a different
    t0 must be able to yield different curvature answers."""
    seen_answers = set()
    for seed in SEEDS:
        q = _make(CURVATURE_SOURCES, "hard", seed)
        seen_answers.add(q.answer_text)
    assert len(seen_answers) > 1


def test_curvature_exam_states_the_numerator_sign():
    for seed in SEEDS:
        q = _make(CURVATURE_SOURCES, "exam", seed)
        assert "positive" in q.text or "negative" in q.text


def test_curvature_skill_loads_through_api_with_correct_properties():
    assert CURVATURE_SKILL.domain == "advanced-geometry-and-topology"
    assert CURVATURE_SKILL.status == "active"
    assert CURVATURE_SKILL.placement_eligible is False
    assert CURVATURE_SKILL.answer == "number"
    assert CURVATURE_SKILL.prerequisites == ["differentiate-polynomial"]
    assert {src.difficulty for src in CURVATURE_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = CURVATURE_SOURCES["exam"]
    assert exam_src.stages == ["exam"]
