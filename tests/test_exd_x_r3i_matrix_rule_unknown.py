"""Exam source of the matrix-notation skill: a rule with an unknown constant, then entries read from it."""

from __future__ import annotations

import re

from sympy import Symbol, solve
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication, split_symbols, standard_transformations
from sympy import parse_expr

SKILL_ID = "senior-understand-the-matrix-definition-and-notation"
TOPIC = "exd_x_r3i_matrix_rule_unknown"
LABELS = ["a", "b", "c", "d"]
I, J, K = Symbol("i"), Symbol("j"), Symbol("k")
_TR = standard_transformations + (split_symbols, implicit_multiplication, convert_xor)

from collections import Counter

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
SEEDS = range(300)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert 3 <= len(q.parts) <= 4
        assert all(p.distractors and len(p.distractors) >= 2 for p in q.parts)
        assert sum(1 for p in q.parts if p.follow_through) >= 1


def test_no_text_defects(questions):
    for q in questions[:100]:
        assert find_text_defects(q, include_working=True) == [], q.text


def test_key_rederived_from_prompt(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: str(d)})
                assert result.parts[p.label].marks == 0, (q.text, p.label, str(d))


def test_distractors_match_key_layout(questions):
    for q in questions:
        for p in q.parts:
            if p.answer_kind != "number":
                continue
            for d in p.distractors:
                assert ("." in d) == ("." in p.answer_text), (p.label, d, p.answer_text)
                if "." in d:
                    assert len(d.split(".")[1]) == len(p.answer_text.split(".")[1]), (p.label, d, p.answer_text)


def test_distractors_are_tagged(questions):
    from mathgen.misconceptions import CATALOGUE

    for q in questions[:100]:
        for p in q.parts:
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (q.text, p.label, d)


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        reply = {p.label: ("999999.5" if p.answer_kind == "number" else "999999/7") for p in q.parts}
        result = grade(_SKILL, q, reply)
        assert result.marks == 0 and not result.correct


def _facts(q):
    rule_text = re.search(r"a_\{i,j\} = (.*?)(?:, where| for a|, for)", q.stem).group(1)
    rule = parse_expr(
        re.sub(r"(\d)(?=[ijk])", r"\1*", rule_text),
        transformations=_TR,
        local_dict={"i": I, "j": J, "k": K},
    )
    m = re.search(r"row (\d+) and column (\d+) is (-?\d+)", q.stem) or re.search(r"a_\{(\d+),(\d+)\} = (-?\d+)", q.stem)
    p, qq, given = (int(v) for v in m.groups())
    r, s = (int(v) for v in re.search(r"a_\{(\d+),(\d+)\}", q.parts[1].prompt).groups())
    t, target = (int(v) for v in re.search(r"row (\d+) equals (-?\d+)", q.parts[2].prompt).groups())
    return rule, (p, qq, given), (r, s), (t, target)


def _rule_at(rule, i, j, k):
    return int(rule.subs({I: i, J: j, K: k}))


def oracle(q):
    rule, (p, qq, given), (r, s), (t, target) = _facts(q)
    (k,) = solve(rule.subs({I: p, J: qq}) - given, K)
    (j0,) = solve(rule.subs({I: t, K: k}) - target, J)
    return {
        "a": str(k),
        "b": str(_rule_at(rule, r, s, k)),
        "c": str(j0),
        "d": str(_rule_at(rule, int(j0), t, k)),
    }


def test_follow_through_from_wrong_constant(questions):
    for q in questions[:150]:
        rule, _, (r, s), (t, _target) = _facts(q)
        wk = int(q.parts[0].distractors[0])
        if wk == 0:
            continue  # no follow-through from a reply of zero
        canon = _canonical(q)
        reply = {**canon, "a": str(wk), "b": str(_rule_at(rule, r, s, wk))}
        j0 = int(canon["c"])
        reply["d"] = str(_rule_at(rule, j0, t, wk))
        if str(wk) in (reply["b"], reply["d"]):
            continue  # a reply that merely repeats the earlier one earns no follow-through
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, reply)
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)
