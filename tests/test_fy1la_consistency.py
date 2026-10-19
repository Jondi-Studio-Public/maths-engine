"""`ucv1-system-consistency-parameter`: the k that gives no solution, no unique solution or infinitely many.

The oracle reads the system out of the rendered text and decides every case with SymPy ranks and determinants; it
never imports the generator.
"""

from __future__ import annotations

import re

import pytest
import sympy

from fy1la_support import (
    TIERS,
    check_exam,
    check_practice,
    check_skill,
    check_variety,
    exam,
    parse,
    system_in,
)
from mathgen import make

SKILL = "ucv1-system-consistency-parameter"
PRACTICE = "fy1la_consistency"
EXAM = "exd_x_fy1la_consistency"
K = sympy.Symbol("k")


def _outcome(a, b, k0):
    """'none', 'one' or 'many' for the system at k = k0."""
    ak = a.subs(K, k0)
    aug = ak.row_join(b)
    if ak.rank() == aug.rank():
        return "one" if ak.rank() == ak.cols else "many"
    return "none"


def _singular_values(a):
    return sympy.solve(a.det(), K)


def _oracle_easy(q):
    a, b, _ = system_in(q.stem or q.text)
    assert a.shape == (2, 2)
    roots = _singular_values(a)
    assert roots == [q.answer] and _outcome(a, b, q.answer) == "none"
    assert _outcome(a, b, q.answer + 1) == "one" and _outcome(a, b, q.answer - 1) == "one"


def _oracle_medium(q):
    a, b, _ = system_in(q.stem or q.text)
    assert a.shape == (3, 3)
    assert _singular_values(a) == [q.answer]
    assert _outcome(a, b, q.answer) in ("none", "many")


def _oracle_hard(q):
    a, b, _ = system_in(q.stem or q.text)
    roots = _singular_values(a)
    assert len(roots) == 2
    outcomes = {r: _outcome(a, b, r) for r in roots}
    assert sorted(outcomes.values()) == ["many", "none"], outcomes
    assert outcomes[q.answer] == "many"


@pytest.mark.parametrize("tier,oracle", list(zip(TIERS, (_oracle_easy, _oracle_medium, _oracle_hard))))
def test_key_matches_the_oracle_and_distractors_are_refused(tier, oracle):
    check_practice(PRACTICE, tier, oracle)


def test_the_hard_tier_offers_the_other_singular_value():
    for seed in range(40):
        q = make(PRACTICE, seed=seed, difficulty="hard")
        a, b, _ = system_in(q.text)
        other = next(r for r in _singular_values(a) if r != q.answer)
        assert str(other) in q.distractors
        assert q.option_notes[str(other)] == "singular-means-infinitely-many"


def _exam_oracle(q):
    a, b, symbols = system_in(q.stem)
    det_part, k_part, how_part, value_part = q.parts
    det = sympy.expand(a.det())
    assert sympy.expand(parse(det_part.answer_text, k=K) - det) == 0
    assert _singular_values(a) == [k_part.answer]
    k0 = k_part.answer
    outcome = _outcome(a, b, k0)
    assert how_part.answer == {"none": "No solution", "many": "Infinitely many solutions"}[outcome]
    k1 = int(re.search(r"k = (-?\d+)", value_part.prompt).group(1))
    unknown = re.search(r"value of ([xyz])", value_part.prompt).group(1)
    assert k1 != k0 and a.subs(K, k1).det() != 0
    solution = a.subs(K, k1).LUsolve(b)
    assert solution[[s.name for s in symbols].index(unknown)] == value_part.answer


def test_exam_chain_matches_the_oracle_and_marks():
    check_exam(EXAM, SKILL, _exam_oracle, [2, 1, 2, 3])


def test_both_singular_outcomes_occur_in_the_exam():
    answers = {exam(EXAM, s).parts[2].answer for s in range(40)}
    assert answers == {"No solution", "Infinitely many solutions"}


def test_a_wrong_classification_option_is_refused_and_tagged():
    for s in range(20):
        q = exam(EXAM, s)
        notes = set(q.parts[2].option_notes.values())
        assert "singular-still-unique" in notes and len(notes) == 2


def test_variety_over_200_seeds():
    check_variety(PRACTICE, "easy", minimum=100)
    check_variety(PRACTICE, "hard", minimum=100)
    check_variety(EXAM, "exam", minimum=100)


def test_skill_is_active_with_a_ladder_and_documented_misconceptions():
    check_skill(SKILL, PRACTICE, EXAM, "number")
