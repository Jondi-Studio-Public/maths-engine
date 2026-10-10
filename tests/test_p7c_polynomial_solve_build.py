"""Tests `simple_polynomial_equation` (ACMSM091) and `factorise_cubic_linear_factor` (ACMMM018).

Skill 1 (ACMSM091): simple_polynomial_equation.
Skill 2 (ACMMM018): factorise_cubic_linear_factor.
"""

from __future__ import annotations

import re

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check

SEEDS = range(60)

TOPIC1 = "simple_polynomial_equation"
SKILL1 = "senior-solve-simple-polynomial-equations"
TOPIC2 = "factorise_cubic_linear_factor"
SKILL2 = "senior-factorise-cubic-polynomials-in-cases-where-a-linear"


# === Skill 1: simple_polynomial_equation =====================================


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_simple_polynomial_distractors_hygiene(difficulty):
    for seed in SEEDS:
        q = make(TOPIC1, seed=seed, difficulty=difficulty)
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_simple_polynomial_real_grader_accept_reject(difficulty):
    for seed in SEEDS:
        q = make(TOPIC1, seed=seed, difficulty=difficulty)
        assert grade_expression(q, q.answer_text).correct is True, q.answer_text
        for d in q.distractors:
            assert grade_expression(q, d).correct is False, d


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_simple_polynomial_deliberately_wrong_rejected(difficulty):
    for seed in range(10):
        q = make(TOPIC1, seed=seed, difficulty=difficulty)
        assert grade_expression(q, "999 + 999i").correct is False


def test_simple_polynomial_grader_accepts_alternate_typed_forms():
    q = make(TOPIC1, seed=0, difficulty="medium")  # answer_text == "1 + 5i"
    assert q.answer_text == "1 + 5i"
    assert grade_expression(q, "5i + 1").correct is True  # reordered
    assert grade_expression(q, "5*i + 1").correct is True  # explicit multiplication
    assert grade_expression(q, "1 + 5I").correct is False  # capital I is not accepted
    assert grade_expression(q, "1 - 5i").correct is False  # conjugate

    qe = make(TOPIC1, seed=1, difficulty="easy")  # answer_text == "3i"
    assert qe.answer_text == "3i"
    assert grade_expression(qe, "3*i").correct is True
    assert grade_expression(qe, "3I").correct is False
    assert grade_expression(qe, "sqrt(-9)").correct is False


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_simple_polynomial_answer_format_consistent(difficulty):
    pattern = re.compile(r"^-?\d*i$|^-?\d+ [+-] \d*i$")
    for seed in range(20):
        q = make(TOPIC1, seed=seed, difficulty=difficulty)
        assert pattern.match(q.answer_text), q.answer_text
        for d in q.distractors:
            assert pattern.match(d), d


_EASY_RE = re.compile(r"z² \+ (\d+) = 0")
_MEDIUM_RE = re.compile(r"\(z ([+-]) (\d+)\)² \+ (\d+) = 0")
_HARD_RE = re.compile(r"z² ([+-]) (\d+)z \+ (\d+) = 0")


def _derive_B_C(text: str):
    m = _EASY_RE.search(text)
    if m:
        return 0, int(m.group(1))
    m = _MEDIUM_RE.search(text)
    if m:
        sign, absp, k = m.groups()
        p = -int(absp) if sign == "+" else int(absp)
        return -2 * p, p * p + int(k)
    m = _HARD_RE.search(text)
    sign, b, c = m.groups()
    b = int(b)
    B = b if sign == "+" else -b
    return B, int(c)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_simple_polynomial_independent_check_fires_on_corrupted_root(difficulty):
    """Mirrors the generator's own `root**2 - 2*p*root + (p**2+k) == 0`
    residual check (B, C form) with a corrupted root, proving `check()`
    raises rather than passing silently."""
    for seed in range(10):
        q = make(TOPIC1, seed=seed, difficulty=difficulty)
        B, C = _derive_B_C(q.text)
        # the real root (parsed from answer_text) satisfies the residual
        m = re.match(r"^(-?\d+)\s*\+\s*(\d*)i$|^(\d*)i$", q.answer_text)
        if m.group(3) is not None:
            coef = m.group(3)
            root = complex(0, int(coef) if coef else 1)
        else:
            coef = m.group(2)
            root = complex(int(m.group(1)), int(coef) if coef else 1)
        residual = root**2 + B * root + C
        check(abs(residual) < 1e-9, "sanity: real root must satisfy its own equation")

        corrupted = root + 1
        corrupted_residual = corrupted**2 + B * corrupted + C
        with pytest.raises(VerificationError):
            check(abs(corrupted_residual) < 1e-9, "corrupted root must fail")


def test_skill1_loads_with_correct_metadata():
    domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL1), None)
    assert skill is not None
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["function-notation"]
    all_stages = set()
    for source in skill.sources:
        all_stages.update(source.stages)
    assert 6 in all_stages


# === Skill 2: factorise_cubic_linear_factor ==================================


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_factorise_cubic_distractors_hygiene(difficulty):
    for seed in SEEDS:
        q = make(TOPIC2, seed=seed, difficulty=difficulty)
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_factorise_cubic_real_grader_accept_reject(difficulty):
    for seed in SEEDS:
        q = make(TOPIC2, seed=seed, difficulty=difficulty)
        assert grade_expression(q, q.answer_text).correct is True, q.answer_text
        for d in q.distractors:
            assert grade_expression(q, d).correct is False, d


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_factorise_cubic_deliberately_wrong_rejected(difficulty):
    for seed in range(10):
        q = make(TOPIC2, seed=seed, difficulty=difficulty)
        assert grade_expression(q, "(x - 999)(x^2 + 999x + 999)").correct is False


def test_factorise_cubic_boundary1_bracket_mechanism():
    """Boundary 1's mechanical proof: reordered factors and caret/explicit-mult
    variants of the SAME factorisation grade correct; the expanded (unfactored)
    cubic and a bracket-free roots list both grade incorrect."""
    q = make(TOPIC2, seed=0, difficulty="medium")
    assert q.answer_text == "(x - 2)(x^2 - 3x + 1)"
    assert grade_expression(q, "(x^2 - 3x + 1)(x - 2)").correct is True  # reordered factors
    assert grade_expression(q, "(x-2)(x**2-3*x+1)").correct is True  # explicit ** and *
    unfactored = q.text.split("Factorise ")[1].split(" completely")[0]
    assert grade_expression(q, unfactored).correct is False  # unfactored cubic
    assert grade_expression(q, "1, 2, 3").correct is False  # bracket-free roots list


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_factorise_cubic_answer_and_distractors_bracketed(difficulty):
    for seed in range(20):
        q = make(TOPIC2, seed=seed, difficulty=difficulty)
        assert "(" in q.answer_text
        for d in q.distractors:
            assert "(" in d, d


_QUESTION_RE = re.compile(r"Factorise (.+) completely as a product")
_LINEAR_ANS_RE = re.compile(r"^(-?\d+)?\(x ([+-]) (\d+)\)")


def _cubic_coeffs_from_text(text: str):
    inner = _QUESTION_RE.search(text).group(1)
    s = inner.strip()
    if not s.startswith(("+", "-")):
        s = "+" + s
    coeffs = {"x^3": 0, "x^2": 0, "x": 0, "": 0}
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
    return coeffs["x^3"], coeffs["x^2"], coeffs["x"], coeffs[""]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_factorise_cubic_independent_check_fires_on_corrupted_divisor(difficulty):
    """Mirrors the generator's own sympy.div-by-(x - p) check with a
    corrupted p, proving a nonzero remainder is caught."""
    for seed in range(10):
        q = make(TOPIC2, seed=seed, difficulty=difficulty)
        A, B, C, D = _cubic_coeffs_from_text(q.text)
        m = _LINEAR_ANS_RE.match(q.answer_text)
        sign, absp = m.group(2), int(m.group(3))
        p = -absp if sign == "+" else absp

        def synthetic_divide(root):
            out = [A]
            for c in (B, C, D):
                out.append(c + root * out[-1])
            return out.pop()

        check(synthetic_divide(p) == 0, "sanity: real p must divide exactly")
        # +1000 rather than +1: a small offset can coincidentally land on
        # ANOTHER genuine integer root of the same cubic (a small-p, small-q,r
        # cubic can have more than one), which would make this a false
        # negative for the corruption test, not a defect in the generator.
        with pytest.raises(VerificationError):
            check(synthetic_divide(p + 1000) == 0, "corrupted p must not divide exactly")


def test_skill2_loads_with_correct_metadata():
    domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL2), None)
    assert skill is not None
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["function-notation"]
    all_stages = set()
    for source in skill.sources:
        all_stages.update(source.stages)
    assert 6 in all_stages
