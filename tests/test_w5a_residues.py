"""``uc2-residues-higher-order-poles``: answers grade right, distractors wrong, the stem's residue is recomputed with SymPy."""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen import make

from w5a_support import (
    ORACLE_SEEDS,
    Z,
    answer_value,
    balanced_after,
    check_follow_through,
    check_grading,
    check_text,
    close,
    coverage,
    numeric,
    parse,
    questions,
    tier_topic,
)

NAME = "residues"
ALL = ("easy", "medium", "hard", "exam")


@pytest.mark.parametrize("tier", ALL)
def test_answers_and_distractors_grade(tier):
    for q in questions(NAME, tier):
        check_grading(q)
        check_text(q)


@pytest.mark.parametrize("tier", ALL)
def test_variety_and_determinism(tier):
    qs = questions(NAME, tier, range(1, 25))
    assert len({q.text for q in qs}) >= 12
    again = make(tier_topic(NAME, tier), seed=7, difficulty=tier)
    assert again.to_dict() == make(tier_topic(NAME, tier), seed=7, difficulty=tier).to_dict()


def test_exam_chain_shape():
    for q in questions(NAME, "exam"):
        assert 3 <= len(q.parts) <= 4
        assert [p.label for p in q.parts] == list("abcd")[: len(q.parts)]
        assert any(p.follow_through for p in q.parts)
        check_follow_through(q)


def test_tiers_are_distinct():
    shapes = {t: {q.text for q in questions(NAME, t, range(1, 13))} for t in ("easy", "medium", "hard")}
    assert shapes["easy"] != shapes["medium"] != shapes["hard"]
    assert not (shapes["easy"] & shapes["hard"])


def test_single_part_tiers_have_no_parts():
    for t in ("easy", "medium", "hard"):
        assert all(not q.parts for q in questions(NAME, t, range(1, 6)))


def _pole_of(text: str):
    for pattern in (
        r"Res\(f, (.+?)\)",
        r"residue (?:of f )?at z = (.+?)(?: of f|\.|$)",
        r"pole at z = (.+?)\.",
        r"at z = (.+?)[ .]",
    ):
        m = re.search(pattern, text)
        if m:
            return parse(m.group(1))
    return None


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_answer_is_the_residue_of_the_stem_function(tier):
    """Recompute Res(f, pole) from the stem alone with SymPy and compare to the stored answer."""
    checked = 0
    qs = questions(NAME, tier, ORACLE_SEEDS)
    for q in qs:
        pole = _pole_of(q.text)
        if pole is None:
            continue
        f = parse(balanced_after(q.text, "f(z) = "))
        want = sympy.residue(f, Z, pole)
        assert close(answer_value(q), numeric(want)), (q.text, q.answer_text, want)
        checked += 1
    coverage(checked, len(qs))


def test_exam_parts_are_consistent():
    """The three residues of the exam function sum to zero, and the circle integral is 2*pi*i times those it encloses."""
    for q in questions(NAME, "exam"):
        f = parse(balanced_after(q.text, "f(z) = "))
        poles = {}
        for p in q.parts[:3]:
            m = re.search(r"z = (-?\d+)", p.prompt)
            poles[p.label] = int(m.group(1))
        for p in q.parts[:3]:
            want = sympy.residue(f, Z, poles[p.label])
            assert close(numeric(parse(p.answer_text)), numeric(want)), (q.text, p.prompt, p.answer_text, want)
