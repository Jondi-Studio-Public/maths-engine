"""Trigonometric equations, technology as the platform.

Tests for senior-solve-equations-involving-trigonometric-functions-using-technology-and
(ACMMM043, topic p14a_solve_trig_equation_technology).
"""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance

TOPIC = "p14a_solve_trig_equation_technology"
SKILL_ID = "senior-solve-equations-involving-trigonometric-functions-using-technology-and"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
VARIANTS = {
    "easy": "direct",
    "medium": "coefficient",
    "hard": "multi_period",
    "exam": "technology_applied",
}
DP_BY_TIER = {"easy": 1, "medium": 1, "hard": 1, "exam": 3}
SEEDS = range(40)


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------

# --- Distractor gap exceeds the stated-precision tolerance, every seed -----


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_gap_exceeds_stated_precision_tolerance(skills_dict, difficulty):
    tol = stated_precision_tolerance(DP_BY_TIER[difficulty])
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        values = [Fraction(q.answer_text)] + [Fraction(d) for d in q.distractors]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                gap = abs(values[i] - values[j])
                assert gap > tol, (difficulty, seed, q.answer_text, q.distractors, gap)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        dp = DP_BY_TIER[difficulty]
        step = float(stated_precision_tolerance(dp)) * 4
        wrong = str(round(float(q.answer) + step, dp))
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer and distractors state exactly the tier's own precision --------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_state_exact_precision(skills_dict, difficulty):
    dp = DP_BY_TIER[difficulty]
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        for text in [q.answer_text] + list(q.distractors):
            assert "." in text, (difficulty, seed, text)
            assert len(text.split(".")[-1].lstrip("-")) == dp, (difficulty, seed, text)


# --- Instruction phrase present in both text and latex ---------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_instruction_phrase_in_both_text_and_latex(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert "decimal place" in q.text.lower(), (difficulty, seed, q.text)
        assert "decimal place" in q.latex.lower(), (difficulty, seed, q.latex)


# --- At least 20 distinct rendered questions per tier over 60 seeds ------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(difficulty):
    variant = VARIANTS[difficulty]
    seen = {make(TOPIC, seed=s, difficulty=difficulty, variant=variant).text for s in range(60)}
    assert len(seen) >= 20, (difficulty, variant, len(seen))


# --- ValueError on a nonsensical preset -----------------------------------


def test_raises_value_error_on_unknown_variant():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, variant="bogus")


# --- The skill's own YAML wires 4 sources, one per tier, exam under
# stages: [exam] ------------------------------------------------------------


def test_skill_sources_and_stages(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["right-triangle-angle"]
    by_difficulty = {s.difficulty: s for s in skill.sources}
    assert set(by_difficulty) == {"easy", "medium", "hard", "exam"}
    assert by_difficulty["exam"].stages == ["exam"]
    assert by_difficulty["exam"].stages != []


# --- Never the right-triangle-angle shape: no "opposite"/"adjacent"/
# "hypotenuse" side-ratio framing anywhere in this skill's questions --------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_never_a_right_triangle_side_ratio_question(skills_dict, difficulty):
    forbidden = ["opposite", "adjacent", "hypotenuse", "right-angled triangle", "right angled triangle"]
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        lower = q.text.lower()
        for word in forbidden:
            assert word not in lower, (difficulty, seed, word, q.text)


# --- Naming-uniqueness discipline: the naming rule's OWN assumed solution
# count is confirmed by the generator's numeric scan on every draw, not
# assumed -- re-derive independently here (a second, test-file-local numeric
# scan, deliberately not calling the generator's own `_enumerate_solutions`) -
_NUM_RE = r"-?\d+(?:\.\d+)?"
_EASY_MEDIUM_RE = re.compile(
    rf"Solve (?P<lhs>.+?) = (?P<k>{_NUM_RE}) for x in the interval \[0.{{1,2}}, 360.{{1,2}}\)\. "
    rf"Give the (?P<which>smaller|larger) solution correct to 1 decimal place\.$"
)
_HARD_RE = re.compile(
    rf"Solve (?P<kind>sin|cos)\((?P<b>\d+)x\) = (?P<k>{_NUM_RE}) for x in the interval "
    rf"\[0.{{1,2}}, 360.{{1,2}}\)\. Give the (?P<n>\d+)\w\w smallest solution \(counting from 1\) "
    rf"correct to 1 decimal place\.$"
)


def _scan(kind, b, k, hi, n_points=9000):
    """A from-scratch numeric enumerator, local to this test file, that never
    calls `mathgen.generators.p14a_trig_equation_technology_build._enumerate_solutions`."""
    import math as _math

    def f(x):
        val = _math.sin(_math.radians(b * x)) if kind == "sin" else _math.cos(_math.radians(b * x))
        return val - k

    step = hi / n_points
    roots = []
    prev_x, prev_f = 0.0, f(0.0)
    for i in range(1, n_points + 1):
        x = i * step
        cur_f = f(x)
        if prev_f * cur_f < 0.0:
            lo_x, hi_x = prev_x, x
            for _ in range(50):
                mid = (lo_x + hi_x) / 2.0
                if prev_f * f(mid) <= 0.0:
                    hi_x = mid
                else:
                    lo_x = mid
            roots.append((lo_x + hi_x) / 2.0)
        prev_x, prev_f = x, cur_f
    roots.sort()
    deduped = []
    for r in roots:
        if not deduped or r - deduped[-1] > 1e-3:
            deduped.append(r)
    return deduped


@pytest.mark.parametrize("difficulty,variant", [("easy", "direct"), ("medium", "coefficient")])
def test_easy_medium_naming_exactly_two_solutions(difficulty, variant):
    """Confirms, over 60 seeds each, that the domain the question states
    genuinely contains EXACTLY two solutions -- the precondition
    "smaller"/"larger" naming needs to be unambiguous by construction."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty, variant=variant)
        m = _EASY_MEDIUM_RE.search(q.text)
        assert m, (difficulty, seed, q.text)
        lhs, k = m["lhs"], float(m["k"])
        kind = "sin" if "sin" in lhs else "cos"
        if difficulty == "medium":
            a_match = re.match(r"(-?\d+) (sin|cos)\(x\)", lhs)
            a_coeff = int(a_match.group(1))
            k = k / a_coeff
        roots = _scan(kind, 1, k, 360.0)
        assert len(roots) == 2, (difficulty, seed, q.text, roots)


def test_hard_naming_ordinal_within_verified_count():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard", variant="multi_period")
        m = _HARD_RE.search(q.text)
        assert m, (seed, q.text)
        kind, b, k, n = m["kind"], int(m["b"]), float(m["k"]), int(m["n"])
        roots = _scan(kind, b, k, 360.0)
        assert len(roots) == 2 * b, (seed, q.text, roots)
        assert 1 <= n <= len(roots), (seed, q.text, n, len(roots))
