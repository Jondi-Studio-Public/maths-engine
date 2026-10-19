"""Exam source of the Euler-formula skill: edges and vertices from face lengths, the 3v - 6 bound, spare edges."""

from __future__ import annotations

import re

SKILL_ID = "senior-apply-euler-s-formula-v-f-e-2"
TOPIC = "exd_x_r3i_euler_edge_room"
LABELS = ["a", "b", "c", "d"]

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


def _faces(text):
    outer = int(re.search(r"(?:outer face has|outer face has|unbounded region has) (\d+)", text).group(1))
    counts = {w: 0 for w in ("triangles", "quadrilaterals", "pentagons")}
    for n, w in re.findall(r"(\d+) are (triangles|quadrilaterals|pentagons)", text):
        counts[w] = int(n)
    for w in re.findall(r"\b1 is an? (triangle|quadrilateral|pentagon)", text):  # the singular reads "1 is a ..."
        counts[w + "s"] = 1
    sides = outer + 3 * counts["triangles"] + 4 * counts["quadrilaterals"] + 5 * counts["pentagons"]
    return sides, 1 + sum(counts.values())


def oracle(q):
    sides, f = _faces(q.stem)
    e = sides // 2
    v = 2 - f + e
    return {"a": str(e), "b": str(v), "c": str(3 * v - 6), "d": str(3 * v - 6 - e)}


def test_follow_through_from_wrong_edge_count(questions):
    checked = 0
    for q in questions[:200]:
        _, f = _faces(q.stem)
        wa = int(q.parts[0].distractors[0])
        wv = 2 + wa - f
        if wv <= 0:
            continue
        reply = {"a": str(wa), "b": str(wv), "c": str(3 * wv - 6), "d": str(3 * wv - 6 - wa)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, reply)
        if 3 * wv - 6 > 0:
            assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
        checked += 1
    assert checked > 100
