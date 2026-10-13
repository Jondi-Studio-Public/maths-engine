"""Learners read maths as it is written on paper: `x^2`, `2x`, `6 × 8`, `√21`, `π`, `e^(2x)`.

`mathgen.core.paper_notation` rewrites Python's spelling (`x**2`, `2*x`, `sqrt(21)`, `pi`, `exp(2*x)`) and
`core.display_question` applies it to the strings a learner reads. The stored question is not changed. These
tests pin the rewrite itself, that a question's stored fields are untouched, that the graders accept the
answer as displayed, and that no active skill displays a Python spelling (`mathgen.text_quality`).
"""

from __future__ import annotations

import pytest

from mathgen import core, make
from mathgen.core import display_question, mcq_options
from mathgen.engine import grade, load_skills
from mathgen.skills import Source
from mathgen.text_quality import PYTHON_NOTATION_PATTERNS, find_python_notation

REWRITES = [
    ("f(x) = 3*x**2 - 9, find f(1)", "f(x) = 3x^2 - 9, find f(1)"),
    ("Simplify: 5**3 × 5**4", "Simplify: 5^3 × 5^4"),
    ("Estimate 192 * 544", "Estimate 192 × 544"),
    ("6*8 = 48", "6 × 8 = 48"),
    ("2 * x", "2x"),
    ("x*y", "xy"),
    ("2*(x + 1)", "2(x + 1)"),
    ("(a + 1)*(a - 1)", "(a + 1)(a - 1)"),
    ("x**2*y**2", "x^2 y^2"),
    ("x*sin(x)", "x sin(x)"),
    ("2*sin(x)*cos(x)", "2 sin(x) cos(x)"),
    ("3*-1", "3 × (-1)"),
    ("2*-x", "2(-x)"),
    ("lambda*t = 5", "lambda × t = 5"),
    ("lambda*I", "lambda × I"),
    ("sum(x_i*y_i)", "sum(x_i × y_i)"),
    ("6*sqrt(21)", "6√21"),
    ("sqrt(2)/2", "√2/2"),
    ("sqrt(x + 1)", "√(x + 1)"),
    ("sqrt(pi)", "√(π)"),
    ("sqrt(3)y", "√3 y"),
    ("3*sqrt(2)*x", "3√2 x"),
    ("sqrt(2)**3", "√(2)^3"),
    ("-5*exp(4*x) + C", "-5 e^(4x) + C"),
    ("exp(x)", "e^x"),
    ("exp(-2*x)*sin(4*x)", "e^(-2x) sin(4x)"),
    ("81*pi/2", "81π/2"),
    ("5pi/6", "5π/6"),
    ("2*pi*r", "2πr"),
    ("pi*r**2", "πr^2"),
    ("e^(2 pi i t)", "e^(2 π i t)"),
    ("1/2 - 9*I/2", "1/2 - 9i/2"),
    ("(2*x + 1)^2*(x^2 - 5)^2", "(2x + 1)^2(x^2 - 5)^2"),
]

UNCHANGED = [
    "Simplify 3x + 2x",
    "A spin, a pick and a pinch of salt",
    "expand(x + 1) and sexp(3)",
    "the minimiser is x* = 2",
    "6 × 8",
    "x^2 + √3 + π",
    "",
]


@pytest.mark.parametrize(("python", "paper"), REWRITES)
def test_python_spelling_is_rewritten_as_on_paper(python, paper):
    assert core.paper_notation(python) == paper


@pytest.mark.parametrize(("python", "paper"), REWRITES)
def test_rewrite_is_idempotent(python, paper):
    assert core.paper_notation(paper) == paper


@pytest.mark.parametrize("text", UNCHANGED)
def test_text_without_python_spelling_is_untouched(text):
    assert core.paper_notation(text) == text


def test_non_strings_pass_through():
    assert core.paper_notation(None) is None
    assert core.paper_notation(5) == 5


# --- the patterns ---------------------------------------------------------------------------------------------


def _find(name):
    return next(p for p in PYTHON_NOTATION_PATTERNS if p.name == name).find


@pytest.mark.parametrize(
    ("name", "bad", "fine"),
    [
        ("sqrt_call", "Simplify sqrt(50)", "Simplify √50, the square root of 50"),
        ("sqrt_call", "x = 1 + sqrt (5)", "isqrt is not used here"),
        ("exp_call", "Differentiate exp(2x)", "Differentiate e^(2x); expand (x + 1)"),
        ("spelled_pi", "an angle of 5pi/6", "a spinning pinch, pick and pie"),
        ("spelled_pi", "Give the answer in terms of pi.", "Give the answer in terms of π."),
        ("star_operator", "Estimate 192 * 544", "the minimiser x* = 2 is a point"),
        ("double_star", "5**3", "5^3"),
    ],
)
def test_patterns_have_no_false_positives(name, bad, fine):
    assert _find(name)(bad) is not None
    assert _find(name)(fine) is None


# --- display_question leaves the stored question alone ----------------------------------------------------------


def test_the_stored_question_keeps_the_generators_spelling():
    q = make("differentiation", seed=0, difficulty="hard")
    assert "*" in q.text
    assert "*" in q.answer_text


def test_display_question_rewrites_every_field_and_changes_nothing_stored():
    q = make("differentiation", seed=0, difficulty="hard")
    before = q.to_json()
    shown = display_question(q)
    assert q.to_json() == before
    assert shown["text"] == core.paper_notation(q.text)
    assert shown["answer_text"] == core.paper_notation(q.answer_text)
    assert not find_python_notation(shown)


def test_display_options_keep_the_index_order_of_mcq_options():
    for seed in range(20):
        q = make("like_terms_select", seed=seed)
        assert display_question(q)["options"] == [core.paper_notation(o) for o in mcq_options(q)]


def test_display_question_covers_each_part_of_a_multi_part_question():
    q = make("exam_parts_implicit_tangent", seed=0, difficulty="exam")
    shown = display_question(q)
    assert [p["label"] for p in shown["parts"]] == [p.label for p in q.parts]
    assert not find_python_notation(shown)


def test_engine_exports_the_helpers():
    from mathgen import engine

    assert engine.paper_notation is core.paper_notation
    assert engine.display_question is core.display_question


# --- grading: the answer as displayed is accepted, the wrong options as displayed are not -----------------------

GRADED = [
    "differentiation",
    "integration",
    "partial_derivatives",
    "rearrange",
    "factor_gcf",
    "simplify_expression",
    "expand_bracket",
    "differential_equation",
    "p13c_antideriv_notation",
    "t1_simplify_surds",
    "exd_x_g1_vector_magnitude",
    "exd_x_c1_chain",
    "exd_x_c1_trig",
    "exd_x_c4_solids_of_revolution",
    "t4_degrees_radians_pi_form",
    "integrate_by_parts",
    "exd_x_c4_by_parts_cyclic",
    "p15e_fourier_transform_gaussian",
    "p13e_heat_equation_mode",
]


# Single-answer exam generators whose skill's exam tier is now a multi-part chain. The generators are still
# registered, so they are checked here under the skill that used them.
RETIRED_EXAM = {
    "exd_x_g1_vector_magnitude": "vector-magnitude",
    "exd_x_c1_chain": "differentiate-chain-rule",
    "exd_x_c1_trig": "differentiate-trig",
    "exd_x_c4_solids_of_revolution": "senior-determine-volumes-of-solids-of-revolution-about-either",
    "exd_x_c4_by_parts_cyclic": "senior-integrate-by-parts",
}


@pytest.fixture(scope="module")
def sources_for_topic():
    by_topic = {}
    skills = load_skills()[1]
    for skill in skills:
        if skill.status == "active":
            for src in skill.sources:
                by_topic.setdefault(src.topic, []).append((skill, src))
    by_id = {skill.id: skill for skill in skills}
    for topic, skill_id in RETIRED_EXAM.items():
        if topic not in by_topic:
            source = Source(topic=topic, params={"tier": "exam"}, difficulty="exam", stages=["exam"])
            by_topic[topic] = [(by_id[skill_id], source)]
    return by_topic


def _make(src, seed):
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (src.params or {}).items()}
    return make(src.topic, seed=seed, difficulty=src.difficulty, **params)


@pytest.mark.parametrize("topic", GRADED)
def test_the_displayed_answer_is_accepted_and_the_displayed_wrong_options_are_not(topic, sources_for_topic):
    changed = 0
    for skill, src in sources_for_topic[topic]:
        if skill.answer == "mcq":
            continue  # an option is picked, not typed
        for seed in range(4):
            q = _make(src, seed)
            shown = display_question(q)
            if shown["answer_text"] != q.answer_text:
                changed += 1
            assert grade(skill, q, shown["answer_text"]).correct, (skill.id, seed, shown["answer_text"])
            assert grade(skill, q, q.answer_text).correct, (skill.id, seed, q.answer_text)
            for wrong in (*shown["distractors"], *q.distractors):
                assert not grade(skill, q, wrong).correct, (skill.id, seed, wrong)
    assert changed, f"{topic} shows no Python spelling to rewrite; pick a topic that does"


# --- every active skill ----------------------------------------------------------------------------------------

_ACTIVE = {s.id: s for s in load_skills()[1] if s.status == "active" and s.sources}


@pytest.mark.parametrize("skill_id", sorted(_ACTIVE))
def test_no_active_skill_displays_python_notation(skill_id):
    from scripts.regrade_sweep import _difficulties_for, _tuplify

    for src in _ACTIVE[skill_id].sources:
        for difficulty in _difficulties_for(src.topic, src.difficulty):
            for seed in range(3):
                q = make(src.topic, seed=seed, difficulty=difficulty, **_tuplify(src.params))
                found = find_python_notation(display_question(q))
                assert not found, f"{src.topic}/{difficulty} seed={seed}: {found}"
