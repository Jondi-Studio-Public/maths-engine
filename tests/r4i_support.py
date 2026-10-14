"""Shared helpers for the second-year statistics and probability tests (`tests/test_r4i_*.py`).

Each test file re-derives the key of a question from the data written in its TEXT, with exact fractions or mpmath, so
the oracle shares no code with the generator. The helpers here do the generic checks through the real grader: the key is
accepted, every distractor is rejected, follow-through works, a follow-through value carried from a wrong earlier part
never equals a later part's distractor, text quality, determinism and the skill's wiring.
"""

from __future__ import annotations

import time
from fractions import Fraction

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


def check_single_tier(sk, make_question, seeds=60):
    """The key is accepted, every distractor is rejected, tagged in the catalogue and stored with its value."""
    for seed in range(seeds):
        q = make_question(seed)
        assert grade(sk, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        assert len(set(q.distractors)) == len(q.distractors), seed
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
        assert 3 <= len(q.parts) <= 4
        assert any(p.follow_through for p in q.parts)
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
    assert sk.id.startswith("uc2-")


def _value(part, text):
    return sympy.nsimplify(sympy.sympify(text.replace("^", "**")), rational=True)


def _state(part, value):
    """`value` stated the way `part` states it: rounded to its decimal places, or exact for a fraction."""
    value = sympy.nsimplify(sympy.N(value, 30), rational=True) if not value.is_Rational else value
    if part.answer_kind == "number" and "." in part.answer_text:
        places = len(part.answer_text.split(".")[1])
        scale = sympy.Integer(10) ** places
        return sympy.floor(value * scale + sympy.Rational(1, 2)) / scale
    return value


def _names(part):
    expr = sympy.sympify(part.follow_through.replace("^", "**"))
    return expr, sorted(str(s) for s in expr.free_symbols if str(s).startswith("part_"))


def check_follow_through(sk, make_exam, label, seeds=30):
    """A learner who is wrong in an earlier part but consistent after it still earns the dependent part's marks."""
    carried = 0
    for seed in range(seeds):
        q = make_exam(seed)
        part = next(p for p in q.parts if p.label == label)
        assert part.follow_through
        expr, names = _names(part)
        wrong_label = names[0][len("part_") :]
        wrong = next(p for p in q.parts if p.label == wrong_label)
        canon = {p.label: p.answer_text for p in q.parts}
        for text in wrong.distractors:
            reply = {**canon, wrong_label: text}
            subs = {sympy.Symbol("part_" + p.label): _value(p, reply[p.label]) for p in q.parts}
            val = _state(part, expr.subs(subs))
            reply[label] = (
                str(sympy.N(val, 12)) if part.answer_kind == "number" and "." in part.answer_text else str(val)
            )
            if part.answer_kind == "number" and "." in part.answer_text:
                places = len(part.answer_text.split(".")[1])
                reply[label] = f"{float(val):.{places}f}"
            result = grade(sk, q, reply)
            if result.parts[label].marks == part.marks and result.parts[wrong_label].marks == 0:
                carried += 1
    assert carried > 0
    return carried


def check_no_follow_through_collision(make_exam, seeds=40):
    """A reply that is right by follow-through from a wrong earlier part (directly, or carried through the parts in
    between) must never be one of the later part's own distractors, which the grader would score 0."""
    for seed in range(seeds):
        q = make_exam(seed)
        canon = {p.label: _value(p, p.answer_text) for p in q.parts}
        for j, source in enumerate(q.parts):
            for wrong_text in source.distractors:
                chained = dict(canon)
                chained[source.label] = _value(source, wrong_text)
                for part in q.parts[j + 1 :]:
                    if not part.follow_through:
                        continue
                    expr, names = _names(part)
                    # direct: only the one wrong part differs
                    direct_subs = {
                        sympy.Symbol(n): (chained if n == "part_" + source.label else canon)[n[5:]] for n in names
                    }
                    direct = _state(part, expr.subs(direct_subs))
                    # carried: the parts in between follow through consistently
                    carried_subs = {sympy.Symbol(n): chained[n[5:]] for n in names}
                    carried_value = _state(part, expr.subs(carried_subs))
                    if any(n[5:] == source.label or chained[n[5:]] != canon[n[5:]] for n in names):
                        chained[part.label] = carried_value
                    for text in part.distractors:
                        d = _value(part, text)
                        assert d != direct, (seed, source.label, wrong_text, part.label, text)
                        assert d != carried_value, (seed, source.label, wrong_text, part.label, text)


def check_speed(make_question, limit_ms=40.0, seeds=40):
    """Median CPU time of make() stays under `limit_ms`."""
    times = []
    for seed in range(1000, 1000 + seeds):
        start = time.process_time()
        make_question(seed)
        times.append(time.process_time() - start)
    times.sort()
    assert times[len(times) // 2] * 1000 < limit_ms, times[len(times) // 2]
