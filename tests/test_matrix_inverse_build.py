"""Matrix inverses: the skills `matrix-inverse-2x2` and `matrix-inverse-3x3`, their practice topics and exam chains.

The oracle reads the matrices out of the RENDERED TEXT, inverts or solves with SymPy (`Matrix.inv`, `LUsolve`,
`det`), and compares with the key entry by entry. It never imports the generator module. Every distractor is
checked three ways: it is a different matrix from the key, the real matrix grader refuses it, and the grader still
refuses it when it is written as a scalar times an integer matrix (the form the prompts invite).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy import Matrix, Symbol, parse_expr

from mathgen import make
from mathgen.engine import grade as skill_grade
from mathgen.grading import GRADERS, grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")
SKILLS = {
    "matrix-inverse-2x2": ("mat_inv_2x2", "exd_x_mat_inv_2x2", 2),
    "matrix-inverse-3x3": ("mat_inv_3x3", "exd_x_mat_inv_3x3", 3),
}
PRACTICE = [(v[0], t) for v in SKILLS.values() for t in TIERS]
EXAMS = [v[1] for v in SKILLS.values()]
_SKILLS = {s.id: s for s in load_skills()[1]}
K = Symbol("k")


def _implicit(s: str) -> str:
    return re.sub(r"(?<=\d)(?=k)", "*", s)


def _matrix_after(text: str, name: str) -> Matrix:
    m = re.search(rf"\b{name} = (\[\[.*?\]\])", text)
    return parse_expr("Matrix(" + _implicit(m.group(1)) + ")", local_dict={"Matrix": Matrix, "k": K})


def _answer_matrix(rows) -> Matrix:
    def entry(e):
        return sympy.Rational(e["p"], e["q"]) if isinstance(e, dict) else sympy.Integer(e)

    return Matrix([[entry(e) for e in row] for row in rows])


def _text_matrix(text: str) -> Matrix:
    body = text.split("=", 1)[1] if "=" in text else text
    rows = re.findall(r"\[([^\[\]]+)\]", body)
    return Matrix(
        [
            [sympy.Rational(Fraction(e.strip()).numerator, Fraction(e.strip()).denominator) for e in r.split(",")]
            for r in rows
        ]
    )


def _scalar_form(M: Matrix) -> str:
    """`M` as (1/L) times an integer matrix, the form a learner is invited to type."""
    L = 1
    for e in M:
        L = math.lcm(L, int(sympy.fraction(e)[1]))
    ints = (M * L).tolist()
    return f"1/{L} [[" + "], [".join(", ".join(str(int(v)) for v in row) for row in ints) + "]]"


def _oracle(q) -> Matrix:
    """The matrix the printed question asks for."""
    text = q.stem or q.text
    if "satisfy" in text:
        A, B = _matrix_after(text, "A"), _matrix_after(text, "B")
        return A.inv() * B if "satisfy AX = B" in text else B * A.inv()
    if "Ax = b" in text:
        A, b = _matrix_after(text, "A"), _matrix_after(text, "b")
        return A.LUsolve(b)
    A = _matrix_after(text, "A")
    m = re.search(r"when k = (-?\d+)", text)
    if m:
        A = A.subs(K, int(m.group(1)))
    return A.inv()


# -- practice --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("topic,tier", PRACTICE)
def test_the_key_is_the_inverse_or_solution_of_the_printed_matrices(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert _answer_matrix(q.answer) == _oracle(q), f"{topic}/{tier} seed={seed}: {q.text}"
        assert _text_matrix(q.answer_text) == _oracle(q)
        assert q.answer_text.startswith(("A⁻¹ = ", "X = ", "x = "))
        assert grade("matrix", q, q.answer_text).correct
        assert grade("matrix", q, _scalar_form(_oracle(q))).correct, "the scalar-times-matrix form is accepted"


@pytest.mark.parametrize("topic,tier", PRACTICE)
def test_distractors_are_real_slips_refused_in_both_spellings(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        key = _oracle(q)
        assert len(q.distractors) >= 2 and len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            wrong = _text_matrix(d)
            assert wrong != key, f"{topic}/{tier} seed={seed}: {d} is the key"
            assert not grade("matrix", q, d).correct
            assert not grade("matrix", q, _scalar_form(wrong)).correct, f"{d} accepted as a scalar multiple"
            assert q.option_notes[d] in CATALOGUE
            assert _answer_matrix(q.distractor_values[d]) == wrong


def _inverse_slip_families(q):
    return {q.option_notes[d] for d in q.distractors}


def test_the_catalogued_slips_are_the_ones_the_brief_names():
    seen_2, seen_3, seen_solve = set(), set(), set()
    for s in SEEDS:
        seen_2 |= _inverse_slip_families(make("mat_inv_2x2", seed=s, difficulty="medium"))
        seen_3 |= _inverse_slip_families(make("mat_inv_3x3", seed=s, difficulty="medium"))
        seen_solve |= _inverse_slip_families(make("mat_inv_2x2", seed=s, difficulty="hard"))
    assert {
        "inverse-diagonal-not-swapped",
        "inverse-offdiag-not-negated",
        "inverse-not-divided-by-det",
        "inverse-global-sign",
    } <= seen_2
    assert {"inverse-cofactors-not-transposed", "inverse-cofactor-signs-missed", "inverse-not-divided-by-det"} <= seen_3
    assert "matrix-equation-wrong-side" in seen_solve


@pytest.mark.parametrize("topic,tier", PRACTICE)
def test_deterministic_and_clean(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()
        assert find_text_defects(q) == [], (topic, tier, seed, find_text_defects(q))


@pytest.mark.parametrize("topic,tier", PRACTICE)
def test_variety(topic, tier):
    answers = Counter(make(topic, seed=s, difficulty=tier).answer_text for s in SEEDS)
    assert len(answers) >= 12 and max(answers.values()) <= 4


# -- the tiers genuinely escalate ------------------------------------------------------------------------------------


def test_2x2_escalates_from_unimodular_to_fractions_to_equations():
    easy = [make("mat_inv_2x2", seed=s, difficulty="easy") for s in SEEDS]
    medium = [make("mat_inv_2x2", seed=s, difficulty="medium") for s in SEEDS]
    hard = [make("mat_inv_2x2", seed=s, difficulty="hard") for s in SEEDS]
    assert all(abs(_matrix_after(q.text, "A").det()) == 1 for q in easy)
    assert all(all(isinstance(e, int) for row in q.answer for e in row) for q in easy)
    assert all(abs(_matrix_after(q.text, "A").det()) >= 2 for q in medium)
    assert all(any(isinstance(e, dict) for row in q.answer for e in row) for q in medium)
    assert all("scalar times a matrix" in q.text for q in medium + hard)
    assert all("scalar" not in q.text for q in easy)
    assert all(re.search(r"satisfy (AX|XA) = B|has a parameter k", q.text) for q in hard)
    assert {m for q in hard for m in re.findall(r"AX = B|XA = B|parameter k", q.text)} == {
        "AX = B",
        "XA = B",
        "parameter k",
    }


def test_3x3_escalates_from_unimodular_to_small_determinants_to_systems():
    easy = [make("mat_inv_3x3", seed=s, difficulty="easy") for s in SEEDS]
    medium = [make("mat_inv_3x3", seed=s, difficulty="medium") for s in SEEDS]
    hard = [make("mat_inv_3x3", seed=s, difficulty="hard") for s in SEEDS]
    assert all(abs(_matrix_after(q.text, "A").det()) == 1 for q in easy)
    assert all(2 <= abs(_matrix_after(q.text, "A").det()) <= 4 for q in medium)
    assert all(re.search(r"Ax = b|has a parameter k", q.text) for q in hard)
    assert any("Ax = b" in q.text for q in hard) and any("parameter k" in q.text for q in hard)
    assert all(len(q.answer) == 3 and len(q.answer[0]) == 3 for q in easy + medium)


# -- exam chains -----------------------------------------------------------------------------------------------------


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _exam_matrix(q) -> Matrix:
    return parse_expr(
        "Matrix(" + _implicit(re.search(r"A = (\[\[.*?\]\])", q.stem).group(1)) + ")",
        local_dict={"Matrix": Matrix, "k": K},
    )


@pytest.mark.parametrize("topic", EXAMS)
def test_exam_chain_shape(topic):
    q = _exam(topic, 4)
    assert tuple(p.answer_kind for p in q.parts) == ("expression", "number", "matrix", "matrix")
    assert [p.label for p in q.parts] == list("abcd")
    assert [p.marks for p in q.parts] == [1, 2, 2, 2]
    for p in q.parts:
        assert len(p.distractors) >= 2
        assert GRADERS[p.answer_kind](p, p.answer_text).correct
        assert not any(GRADERS[p.answer_kind](p, d).correct for d in p.distractors), p.distractors
        assert all(n in CATALOGUE for n in p.option_notes.values())


@pytest.mark.parametrize("topic", EXAMS)
def test_exam_answers_rederived_from_the_text(topic):
    n = 2 if "2x2" in topic else 3
    for seed in range(60):
        q = _exam(topic, seed)
        A = _exam_matrix(q)
        det = sympy.expand(A.det())
        a, b, c, d = q.parts
        assert (
            sympy.expand(parse_expr(_implicit(a.answer_text).replace(" ", "")) - det) == 0
            or sympy.expand(parse_expr(a.answer) - det) == 0
        )
        assert sympy.expand(parse_expr(a.answer) - det) == 0
        k0 = sympy.solve(det, K)
        assert k0 == [b.answer], (q.stem, k0, b.answer)
        k1 = int(re.search(r"when k = (-?\d+)", c.prompt).group(1))
        assert k1 != b.answer and det.subs(K, k1) != 0
        assert _answer_matrix(c.answer) == A.subs(K, k1).inv()
        rhs = _matrix_after(d.prompt, "B" if n == 2 else "b")
        want = A.subs(K, k1).inv() * rhs
        assert _answer_matrix(d.answer) == want
        assert A.subs(K, k1) * want == rhs
        if n == 2:
            for text in d.distractors:
                assert _text_matrix(text) != want and not grade("matrix", d, _scalar_form(_text_matrix(text))).correct
        for part in (c, d):
            for text in part.distractors:
                assert not GRADERS["matrix"](part, _scalar_form(_text_matrix(text))).correct


@pytest.mark.parametrize("topic", EXAMS)
def test_exam_variety_over_300_seeds(topic):
    texts = Counter(_exam(topic, s).answer_text for s in range(300))
    assert len(texts) >= 100 and max(texts.values()) <= 6


def test_the_2x2_chain_offers_the_wrong_side_slip_in_part_d():
    notes = Counter(n for s in range(60) for n in _exam("exd_x_mat_inv_2x2", s).parts[3].option_notes.values())
    assert notes["matrix-equation-wrong-side"] > 30


@pytest.mark.parametrize("topic", EXAMS)
def test_exam_deterministic_clean_and_gradable_through_the_skill(topic):
    skill = next(_SKILLS[sid] for sid, v in SKILLS.items() if v[1] == topic)
    for seed in range(30):
        q = _exam(topic, seed)
        assert q.to_json() == _exam(topic, seed).to_json()
        assert find_text_defects(q) == [], (topic, seed, find_text_defects(q))
        result = skill_grade(skill, q, {p.label: p.answer_text for p in q.parts})
        assert result.correct and result.marks == result.max_marks == 7
        # a wrong inverse in (c) costs those marks only
        wrong = {p.label: p.answer_text for p in q.parts}
        wrong["c"] = q.parts[2].distractors[0]
        partial = skill_grade(skill, q, wrong)
        assert partial.marks == 5 and not partial.correct


# -- the skills ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", SKILLS)
def test_the_skill_is_active_with_a_tier_ladder_and_an_exam(skill_id):
    skill = _SKILLS[skill_id]
    practice, exam, _n = SKILLS[skill_id]
    assert skill.status == "active" and not getattr(skill, "deferred_reason", None)
    assert skill.answer == "matrix"
    assert [(s.topic, s.difficulty) for s in skill.sources] == [(practice, t) for t in TIERS] + [(exam, "exam")]
    assert skill.sources[-1].stages == ["exam"]
    ids = [m.id for m in skill.misconceptions]
    assert len(ids) >= 4 and all(i in CATALOGUE for i in ids)


@pytest.mark.parametrize("skill_id", SKILLS)
def test_every_distractor_misconception_is_documented_on_the_skill(skill_id):
    practice, exam, _n = SKILLS[skill_id]
    used = set()
    for topic, tier in [(practice, t) for t in TIERS] + [(exam, "exam")]:
        for seed in range(30):
            q = make(topic, seed=seed, difficulty=tier)
            used |= set(q.option_notes.values())
            for p in q.parts:
                used |= set(p.option_notes.values())
    assert used <= set(CATALOGUE)
    assert len(used & {m.id for m in _SKILLS[skill_id].misconceptions}) >= 3


def test_answer_latex_is_plain_so_the_sweep_does_not_see_a_letter_run():
    q = make("mat_inv_2x2", seed=2, difficulty="medium")
    assert q.answer_latex.startswith("plain:")
    assert json.loads(q.to_json())["answer_kind"] if "answer_kind" in json.loads(q.to_json()) else True
