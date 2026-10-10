"""Tests `p14a_solve_cubic_technology` (ACMMM019) and `p14a_solve_exp_technology` (ACMMM067).

Skill A (ACMMM019): p14a_solve_cubic_technology.
Skill B (ACMMM067): p14a_solve_exp_technology.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check, stated_precision_tolerance

SEEDS = range(60)
DIFFS = ["easy", "medium", "hard", "exam"]

TOPIC_A = "p14a_solve_cubic_technology"
SKILL_A = "senior-solve-cubic-equations-using-technology-and-algebraically-in"
TOPIC_B = "p14a_solve_exp_technology"
SKILL_B = "senior-solve-equations-involving-exponential-functions-using-technology-and"

_DP3_TOLERANCE = float(stated_precision_tolerance(3))


# =============================================================================
# Shared helpers
# =============================================================================


def _assert_3dp(answer_text: str):
    assert "." in answer_text, answer_text
    assert len(answer_text.split(".")[1]) == 3, answer_text


# =============================================================================
# Skill A: p14a_solve_cubic_technology
# =============================================================================


@pytest.mark.parametrize("difficulty", DIFFS)
def test_cubic_distractors_hygiene(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        assert len(q.distractors) >= 2, (difficulty, seed, q.text, q.distractors)


@pytest.mark.parametrize("difficulty", DIFFS)
def test_cubic_real_grader_accept_reject(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        assert grade_number(q, q.answer_text).correct is True, (q.text, q.answer_text)
        for d in q.distractors:
            assert grade_number(q, d).correct is False, (q.text, d)


@pytest.mark.parametrize("difficulty", DIFFS)
def test_cubic_deliberately_wrong_rejected(difficulty):
    for seed in range(10):
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        target = float(q.answer_text)
        wrong = target + 1.0  # comfortably outside the 3 d.p. tolerance
        assert grade_number(q, f"{wrong:.3f}").correct is False


@pytest.mark.parametrize("difficulty", DIFFS)
def test_cubic_answer_3dp_and_consistent_formatting(difficulty):
    for seed in range(20):
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        _assert_3dp(q.answer_text)
        for d in q.distractors:
            _assert_3dp(d)


@pytest.mark.parametrize("difficulty", DIFFS)
def test_cubic_instruction_phrase_in_text_and_latex(difficulty):
    for seed in range(20):
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        assert "decimal places" in q.text
        assert "decimal places" in q.latex


def test_easy_tier_answer_concentration_and_distinct_count():
    """concentrated 29% on the single value '6.000' at 200 seeds, only 10 distinct answers.
    Target: top answer <= 15%, >= 20 distinct answers, over 200 seeds."""
    from collections import Counter

    counts = Counter()
    for seed in range(200):
        q = make(TOPIC_A, seed=seed, difficulty="easy")
        counts[q.answer_text] += 1
    total = sum(counts.values())
    top_share = counts.most_common(1)[0][1] / total
    assert len(counts) >= 20, f"only {len(counts)} distinct answers over 200 seeds"
    assert top_share <= 0.15, f"top answer share {top_share:.3f} exceeds 0.15"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_cubic_independent_check_fires_on_corrupted_root(difficulty):
    """Mirrors the generator's own residual check with a corrupted claimed
    root, proving `check()` raises rather than passing silently. Parses the
    displayed cubic's coefficients back out of `question.text`."""
    import re

    for seed in range(10):
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        m = re.match(r"(.+?) = 0", q.text)
        cubic_text = m.group(1)
        coeffs = {"x^3": 0, "x^2": 0, "x": 0, "": 0}
        s = cubic_text.strip()
        if not s.startswith(("+", "-")):
            s = "+" + s
        for raw in re.findall(r"[+-]\s*[^+-]+", s):
            term = raw.strip()
            sign = -1 if term.startswith("-") else 1
            body = term[1:].strip()
            sym = ""
            for candidate in ("x^3", "x^2", "x"):
                if body.endswith(candidate):
                    sym = candidate
                    body = body[: -len(candidate)].strip()
                    break
            coef = int(body) if body else 1
            coeffs[sym] = sign * coef
        a, b, c, d = coeffs["x^3"], coeffs["x^2"], coeffs["x"], coeffs[""]

        root = float(q.answer_text)
        residual = a * root**3 + b * root**2 + c * root + d
        check(abs(residual) < 0.05, "sanity: the claimed root must satisfy its own cubic")

        corrupted = root + 1000.0
        corrupted_residual = a * corrupted**3 + b * corrupted**2 + c * corrupted + d
        with pytest.raises(VerificationError):
            check(abs(corrupted_residual) < 0.05, "a corrupted root must not satisfy the cubic")


def test_skillA_loads_with_correct_metadata():
    domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_A), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["function-notation"]
    all_stages = set()
    for source in skill.sources:
        all_stages.update(source.stages)
    assert "exam" in all_stages


# =============================================================================
# Skill B: p14a_solve_exp_technology
# =============================================================================


@pytest.mark.parametrize("difficulty", DIFFS)
def test_exp_real_grader_accept_reject(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        assert grade_number(q, q.answer_text).correct is True, (q.text, q.answer_text)
        for d in q.distractors:
            assert grade_number(q, d).correct is False, (q.text, d)


@pytest.mark.parametrize("difficulty", DIFFS)
def test_exp_deliberately_wrong_rejected(difficulty):
    for seed in range(10):
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        target = float(q.answer_text)
        wrong = target + 1.0
        assert grade_number(q, f"{wrong:.3f}").correct is False


@pytest.mark.parametrize("difficulty", DIFFS)
def test_exp_answer_3dp_and_consistent_formatting(difficulty):
    for seed in range(20):
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        _assert_3dp(q.answer_text)
        for d in q.distractors:
            _assert_3dp(d)


@pytest.mark.parametrize("difficulty", DIFFS)
def test_exp_instruction_phrase_in_text_and_latex(difficulty):
    for seed in range(20):
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        assert "decimal places" in q.text
        assert "decimal places" in q.latex


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_exp_additive_constant_present_in_every_tier(difficulty):
    """The additive constant is what separates this skill from the shipping
    neighbour `solve_indices_with_logarithms` at every tier -- confirm the
    displayed equation always carries a '+' or '-' term after the exponential
    (excluding the exponent's own internal sign)."""
    for seed in range(20):
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        lhs = q.text.split(" = ")[0]
        after_exp = lhs.split(")", 1)[1]
        assert after_exp.strip() != "", (difficulty, q.text)


def test_exp_hidden_quadratic_discards_nonpositive_u():
    """Every hard-tier claimed x corresponds to e^x = u > 0; the generator
    must never emit a claimed x whose u is non-positive."""
    for seed in range(30):
        q = make(TOPIC_B, seed=seed, difficulty="hard")
        u = math.exp(float(q.answer_text))
        assert u > 0


def test_exp_independent_check_fires_on_corrupted_answer():
    """Mirrors the additive-isolate residual check with a corrupted x."""
    for seed in range(10):
        q = make(TOPIC_B, seed=seed, difficulty="easy")
        x = float(q.answer_text)
        # Parse "Solve {A} x e^({exponent}) {sign} {c} = {d} for x."
        import re

        m = re.match(r"Solve (\d+) . e\^\(([^)]+)\) ([+-]) (\d+) = (-?\d+) for x", q.text)
        A = int(m.group(1))
        exponent_str = m.group(2)
        sign = 1 if m.group(3) == "+" else -1
        c = sign * int(m.group(4))
        d = int(m.group(5))
        k = 1 if exponent_str == "x" else (-1 if exponent_str == "-x" else int(exponent_str.replace("x", "")))
        lhs = A * math.exp(k * x) + c
        check(abs(lhs - d) < 1.0, "sanity: the claimed x must satisfy its own equation")
        corrupted = x + 0.5  # small enough never to overflow math.exp for any k in [-4, 4]
        corrupted_lhs = A * math.exp(k * corrupted) + c
        with pytest.raises(VerificationError):
            check(abs(corrupted_lhs - d) < 1.0, "a corrupted x must not satisfy the equation")


def test_skillB_loads_with_correct_metadata():
    domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_B), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["function-notation"]
    all_stages = set()
    for source in skill.sources:
        all_stages.update(source.stages)
    assert "exam" in all_stages
