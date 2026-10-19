"""Shared helpers for the second-year probability and statistics tests (`tests/test_r3k_*.py`).

Each test file reads the data of a question back out of its TEXT and re-derives the key with SymPy or exact fractions,
so the oracle shares no code with the generator. The helpers here do the generic checks: the key is accepted, every
distractor of every part is rejected and tagged, text quality, determinism, and the skill's wiring.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects


def skill(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def stem(q):
    return q.stem if q.parts else q.text


def as_fraction(value):
    """A stored `fraction`/`number` answer (an int, `{"p", "q"}` or a float) as an exact Fraction."""
    if isinstance(value, dict):
        return Fraction(value["p"], value["q"])
    return Fraction(str(value))


def first_fraction(text):
    """The first `a/b` written in `text`, as a Fraction."""
    m = re.search(r"(\d+)/(\d+)", text)
    return Fraction(int(m.group(1)), int(m.group(2)))


def all_fractions(text):
    return [Fraction(int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", text)]


def check_single_tier(sk, make_question, seeds=60):
    """The key is accepted, every distractor is rejected, tagged in the catalogue and stored with its value."""
    for seed in range(seeds):
        q = make_question(seed)
        assert grade(sk, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(sk, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == make_question(seed).to_json()


def check_exam_grading(sk, make_exam, seeds=40):
    """Full marks for the canonical reply; zero for a part's own distractor (the other parts correct)."""
    for seed in range(seeds):
        q = make_exam(seed)
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks, (seed, full)
        canon = {p.label: p.answer_text for p in q.parts}
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == make_exam(seed).to_json()
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(sk, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE


def check_wiring(sk, topic, exam_topic, prerequisites, kind):
    assert sk.answer == kind
    assert sk.status == "active"
    assert [(s.topic, s.difficulty) for s in sk.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam_topic, "exam"),
    ]
    assert sk.sources[-1].stages == ["exam"]
    for pre in prerequisites:
        assert pre in sk.prerequisites
    assert "university-year-2" in sk.tags
    assert list(sk.year_level) == ["university"]


def check_follow_through(sk, make_exam, label, seeds=30):
    """A learner who is wrong in an earlier part but consistent after it still earns the dependent part's marks."""
    carried = 0
    for seed in range(seeds):
        q = make_exam(seed)
        part = next(p for p in q.parts if p.label == label)
        assert part.follow_through
        named = [
            n[len("part_") :]
            for n in sorted({str(s) for s in sympy.sympify(part.follow_through).free_symbols})
            if n.startswith("part_")
        ]
        wrong_label = named[0]
        wrong = next(p for p in q.parts if p.label == wrong_label)
        canon = {p.label: p.answer_text for p in q.parts}
        for text in wrong.distractors:
            reply = {**canon, wrong_label: text}
            expr = sympy.sympify(part.follow_through)
            value = {sympy.Symbol("part_" + wrong_label): _numeric(wrong, text)}
            others = {
                sympy.Symbol("part_" + p.label): _numeric(p, p.answer_text) for p in q.parts if p.label != wrong_label
            }
            val = expr.subs({**others, **value})
            if val.is_Rational:
                reply[label] = str(val)
                result = grade(sk, q, reply)
                if result.parts[label].marks == part.marks and result.parts[wrong_label].marks == 0:
                    carried += 1
    assert carried > 0
    return carried


def _numeric(part, text):
    return sympy.nsimplify(sympy.sympify(text.replace("^", "**")))


@pytest.fixture(autouse=True, scope="module")
def second_route_on():
    """Switch on the generators' slow independent cross-checks (quadrature, a second symbolic derivation) for a test
    module. At serving time they are off; here every draw is verified by a second route as well."""
    from mathgen.generators import _r3k_kit

    _r3k_kit.SECOND_ROUTE = True
    yield
    _r3k_kit.SECOND_ROUTE = False


def check_no_follow_through_collision(make_exam, seeds=40):
    """A reply that is right by follow-through from a wrong earlier part must never be one of the later part's own
    distractors (which the grader would score 0)."""
    for seed in range(seeds):
        q = make_exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for part in q.parts:
            if not part.follow_through:
                continue
            expr = sympy.sympify(part.follow_through.replace("^", "**"))
            names = sorted(str(s) for s in expr.free_symbols if str(s).startswith("part_"))
            for name in names:
                source = next(p for p in q.parts if p.label == name[len("part_") :])
                for wrong in source.distractors:
                    subs = {
                        sympy.Symbol("part_" + p.label): _numeric(p, canon[p.label])
                        for p in q.parts
                        if p.label != source.label and "part_" + p.label in names
                    }
                    subs[sympy.Symbol(name)] = _numeric(source, wrong)
                    value = sympy.nsimplify(expr.subs(subs))
                    for text in part.distractors:
                        assert sympy.simplify(value - _numeric(part, text)) != 0, (seed, part.label, wrong, text)
