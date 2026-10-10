"""Evaluate and use inverse trigonometric functions."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5e_inverse_trig_use_build import (
    _EASY_ROWS,
    _MEDIUM_ROWS,
    _TRIPLES,
    _HARD_KINDS,
    _lookup_question,
    _composition_question,
)
from sympy import sqrt, Rational, pi, asin, acos, atan, sin, cos, tan, simplify, sympify

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

TOPIC = "find_and_use_the_inverse_trigonometric_functions_arcsine"


# --- Determinism & distractor hygiene ----------------------------------------


def test_bare_call_default_has_enough_distractors():
    """Bare `make(topic, seed=s)` calls must have >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(TOPIC, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_from_answer_sympy_level(difficulty):
    """Distractors must be symbolically distinct from the answer."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        answer_expr = sympify(question.answer)
        for distractor in question.distractors:
            # Convert ^ back to ** for sympy parsing
            distractor_expr = sympify(distractor.replace("^", "**"))
            diff = simplify(answer_expr - distractor_expr)
            assert diff != 0, f"Seed {seed}/{difficulty}: distractor {distractor} equals answer symbolically"


# --- Real grader testing -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_rejects_distractors(difficulty):
    """Real grader must accept answer and reject distractors."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        result = grade_expression(question, question.answer_text)
        assert result.correct, f"Seed {seed}/{difficulty}: canonical not accepted"
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, f"Seed {seed}/{difficulty}: distractor {distractor} was accepted"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_deliberately_wrong_answer(difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        # For expression answers, add 1 to make it wrong
        wrong = question.answer_text + "+1"
        result = grade_expression(question, wrong)
        assert not result.correct, f"Seed {seed}/{difficulty}: wrong answer '{wrong}' was accepted"


# --- Exhaustive fixed-case testing -----


def test_all_easy_rows_exact_symbolic_and_numeric():
    """Verify all 12 easy rows: exact symbolic equality and numeric cross-check."""
    fn_map = {"asin": asin, "acos": acos, "atan": atan}

    for fn, display, input_expr, answer_str, distractors in _EASY_ROWS:
        # Symbolic equality: parsed answer equals function evaluation
        answer_parsed = sympify(answer_str)
        func_eval = fn_map[fn](input_expr)
        assert simplify(answer_parsed - func_eval) == 0, f"Easy {fn}({display}): symbolic equality failed"

        # Numeric cross-check: both evaluate to within 1e-9
        numeric_answer = float(answer_parsed.evalf())
        numeric_func = float(func_eval.evalf())
        assert abs(numeric_answer - numeric_func) < 1e-9, f"Easy {fn}({display}): numeric check failed"

        # Verify distractors are distinct by value
        for distractor_str in distractors:
            distractor_parsed = sympify(distractor_str)
            assert simplify(answer_parsed - distractor_parsed) != 0, (
                f"Easy {fn}({display}): distractor {distractor_str} equals answer"
            )


def test_all_medium_rows_exact_symbolic_and_numeric():
    """Verify all 11 medium rows: exact symbolic equality and numeric cross-check."""
    fn_map = {"asin": asin, "acos": acos, "atan": atan}

    for fn, display, input_expr, answer_str, distractors in _MEDIUM_ROWS:
        # Symbolic equality
        answer_parsed = sympify(answer_str)
        func_eval = fn_map[fn](input_expr)
        assert simplify(answer_parsed - func_eval) == 0, f"Medium {fn}({display}): symbolic equality failed"

        # Numeric cross-check
        numeric_answer = float(answer_parsed.evalf())
        numeric_func = float(func_eval.evalf())
        assert abs(numeric_answer - numeric_func) < 1e-9, f"Medium {fn}({display}): numeric check failed"

        # Verify distractors are distinct by value
        for distractor_str in distractors:
            distractor_parsed = sympify(distractor_str)
            assert simplify(answer_parsed - distractor_parsed) != 0, (
                f"Medium {fn}({display}): distractor {distractor_str} equals answer"
            )


def test_all_hard_rows_pythagorean_and_sympy():
    """Verify all 20 hard rows: Pythagorean identity and sympy symbolic equality."""
    for a, b, c in _TRIPLES:
        # Verify Pythagorean triple
        assert a**2 + b**2 == c**2, f"Invalid triple: {a}^2 + {b}^2 != {c}^2"

        for kind in _HARD_KINDS:
            if kind == "sin_acos":
                expected = Rational(b, c)
                sympy_result = sin(acos(Rational(a, c)))
            elif kind == "cos_asin":
                expected = Rational(b, c)
                sympy_result = cos(asin(Rational(a, c)))
            elif kind == "tan_acos":
                expected = Rational(b, a)
                sympy_result = tan(acos(Rational(a, c)))
            elif kind == "tan_asin":
                expected = Rational(a, b)
                sympy_result = tan(asin(Rational(a, c)))

            # Sympy auto-simplification should match expected
            assert simplify(sympy_result - expected) == 0, f"Hard {kind} triple ({a}, {b}, {c}): sympy check failed"


def test_all_easy_rows_via_generator():
    """Generate and grade all 12 easy rows."""
    fn_map = {"asin": asin, "acos": acos, "atan": atan}

    for fn, display, input_expr, answer_str, distractors in _EASY_ROWS:
        # Generate the question using the builder
        question_dict = _lookup_question(fn, display, input_expr, answer_str, distractors)
        # Parse to construct a proper Question object
        from mathgen.core import Question

        question = Question(
            topic=TOPIC,
            level=21,
            seed=0,  # Seed doesn't matter for fixed tables
            text=question_dict["text"],
            latex=question_dict["latex"],
            answer=question_dict["answer"],
            answer_text=question_dict["answer_text"],
            answer_latex=question_dict.get("answer_latex", ""),
            distractors=question_dict["distractors"],
        )

        # Real grader must accept the canonical answer
        result = grade_expression(question, question.answer_text)
        assert result.correct, f"Easy {fn}({display}): grader rejected canonical answer"

        # Real grader must reject each distractor
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, f"Easy {fn}({display}): grader accepted distractor {distractor}"


def test_all_medium_rows_via_generator():
    """Generate and grade all 11 medium rows."""
    for fn, display, input_expr, answer_str, distractors in _MEDIUM_ROWS:
        # Generate the question using the builder
        question_dict = _lookup_question(fn, display, input_expr, answer_str, distractors)
        # Parse to construct a proper Question object
        from mathgen.core import Question

        question = Question(
            topic=TOPIC,
            level=21,
            seed=0,
            text=question_dict["text"],
            latex=question_dict["latex"],
            answer=question_dict["answer"],
            answer_text=question_dict["answer_text"],
            answer_latex=question_dict.get("answer_latex", ""),
            distractors=question_dict["distractors"],
        )

        # Real grader must accept the canonical answer
        result = grade_expression(question, question.answer_text)
        assert result.correct, f"Medium {fn}({display}): grader rejected canonical answer"

        # Real grader must reject each distractor
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, f"Medium {fn}({display}): grader accepted distractor {distractor}"


def test_all_hard_rows_via_generator():
    """Generate and grade all 20 hard rows."""
    from mathgen.core import Question

    for a, b, c in _TRIPLES:
        for kind in _HARD_KINDS:
            # Generate the question using the builder
            question_dict = _composition_question(a, b, c, kind)
            question = Question(
                topic=TOPIC,
                level=21,
                seed=0,
                text=question_dict["text"],
                latex=question_dict["latex"],
                answer=question_dict["answer"],
                answer_text=question_dict["answer_text"],
                answer_latex=question_dict.get("answer_latex", ""),
                distractors=question_dict["distractors"],
            )

            # Real grader must accept the canonical answer
            result = grade_expression(question, question.answer_text)
            assert result.correct, f"Hard {kind} ({a},{b},{c}): grader rejected canonical answer"

            # Real grader must reject each distractor
            for distractor in question.distractors:
                result = grade_expression(question, distractor)
                assert not result.correct, f"Hard {kind} ({a},{b},{c}): grader accepted distractor {distractor}"


def test_no_arcsin_arccos_arctan_in_answer_or_distractors():
    """Assert 'arcsin', 'arccos', 'arctan' never appear in answer_text or distractors."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            for pattern in ["arcsin(", "arccos(", "arctan("]:
                assert pattern not in question.answer_text, f"Seed {seed}/{difficulty}: '{pattern}' in answer_text"
                for distractor in question.distractors:
                    assert pattern not in distractor, (
                        f"Seed {seed}/{difficulty}: '{pattern}' in distractor {distractor}"
                    )


# --- Skill loading -----


def test_skill_loads_with_correct_properties():
    """Inverse trig skill must load with correct properties."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-find-and-use-the-inverse-trigonometric-functions-arcsine"), None)
    assert skill is not None, "Inverse trig skill not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible == False
    assert "differentiate-polynomial" in skill.prerequisites


def test_skill_sources_cover_correct_stages():
    """Skill must have three sources covering stages [0,1,2], [3,4], [5,6,maintenance]."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-find-and-use-the-inverse-trigonometric-functions-arcsine"), None)
    assert skill is not None
    assert len(skill.sources) == 3
    # Check that all sources have the correct topic
    for source in skill.sources:
        assert source.topic == "find_and_use_the_inverse_trigonometric_functions_arcsine"
    # Check stages for each difficulty
    assert set(skill.sources[0].stages) == {0, 1, 2}, "easy stages"
    assert set(skill.sources[1].stages) == {3, 4}, "medium stages"
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}, "hard stages"
