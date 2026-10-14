"""Exam source of `finite-set-operations`: a compound set expression, its size, and a count of its subsets."""

from __future__ import annotations

import re
from itertools import combinations

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u1_set_expression_counts"
SKILL_ID = "finite-set-operations"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed)


class S(frozenset):
    """A frozenset with the operators of the notation in the prompt; `~` is the complement in U."""

    universe: frozenset = frozenset()

    def __or__(self, other):
        return S(frozenset.__or__(self, other))

    def __and__(self, other):
        return S(frozenset.__and__(self, other))

    def __sub__(self, other):
        return S(frozenset.__sub__(self, other))

    def __xor__(self, other):
        return S(frozenset.__xor__(self, other))

    def __invert__(self):
        return S(self.universe - self)


def _sets(stem):
    found = {name: S(int(v) for v in body.split(",")) for name, body in re.findall(r"([UABC]) = \{([^}]*)\}", stem)}
    S.universe = frozenset(found["U"])
    return found


def oracle(q):
    sets = _sets(q.stem)
    expr = re.search(r"R = (.+?)(?: \(the|\.$)", q.stem).group(1)
    expr = re.sub(r"(\([^()]*\)|[ABC])′", r"(~\1)", expr)
    for symbol, op in (("∪", "|"), ("∩", "&"), ("∖", "-"), ("△", "^")):
        expr = expr.replace(symbol, op)
    result = eval(expr, {"__builtins__": {}}, {k: sets[k] for k in "ABC"})  # noqa: S307 - the test's own strings
    members = sorted(result)
    k = len(members)
    prompt = q.parts[2].prompt
    if "non-empty proper" in prompt:
        count = sum(1 for r in range(1, k) for _ in combinations(members, r))
    elif "exactly two" in prompt:
        count = sum(1 for _ in combinations(members, 2))
    else:
        count = sum(1 for r in range(k + 1) for _ in combinations(members, r))
    return "{" + ", ".join(map(str, members)) + "}", str(k), str(count)


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.answer_kind for p in q.parts] == ["list", "number", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]


def test_oracle_agrees(questions):
    for q in questions:
        canon = _canonical(q)
        assert oracle(q) == (canon["a"], canon["b"], canon["c"]), q.text


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_count(questions):
    """A wrong size in (b) used correctly in (c) earns the marks for (c)."""
    checked = 0
    for q in questions:
        k = int(q.parts[1].answer_text)
        wrong_k = k + 1
        prompt = q.parts[2].prompt
        if "non-empty proper" in prompt:
            wrong_c = 2**wrong_k - 2
        elif "exactly two" in prompt:
            wrong_c = wrong_k * (wrong_k - 1) // 2
        else:
            wrong_c = 2**wrong_k
        if wrong_c == wrong_k:  # repeating the earlier number is not following through (the engine withholds it)
            continue
        result = grade(_SKILL, q, {**_canonical(q), "b": str(wrong_k), "c": str(wrong_c)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["c"].follow_through is True
        checked += 1
    assert checked >= 250
