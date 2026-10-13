"""Shared checks for the multi-part measurement exam sources (`exd_x_m5_*`).

Each test file supplies its own `oracle`, which re-derives every part's answer from the rendered stem and
prompts with independent code; the checks here cover shapes, grading and follow-through.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from mathgen import make
from mathgen.engine import grade
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


def load_skill(skill_id):
    _, skills = load_skills()
    return next(s for s in skills if s.id == skill_id)


def exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def dp_of(text):
    return len(text.split(".")[1]) if "." in text else 0


def round_half_up(value, dp):
    scaled = Fraction(value) * 10**dp
    return Fraction(int(scaled + Fraction(1, 2)) if scaled >= 0 else -int(-scaled + Fraction(1, 2)), 10**dp)


def show(value, dp):
    scaled = int(round_half_up(value, dp) * 10**dp)
    if dp == 0:
        return str(scaled)
    return f"{scaled // 10**dp}.{scaled % 10**dp:0{dp}d}"


def number(text):
    return float(re.search(r"-?\d+(?:\.\d+)?", text).group(0))


def check_shapes(topic, skill_id):
    skill = load_skill(skill_id)
    exam_shapes = {prompt_shape(exam(topic, s)) for s in range(60)}
    hard_src = next(s for s in skill.sources if s.difficulty == "hard")
    hard_shapes = {prompt_shape(make(hard_src.topic, seed=s, **hard_src.params)) for s in range(60)}
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard_shapes
    for s in range(60):
        assert find_text_defects(exam(topic, s)) == []


def check_source_wired(topic, skill_id):
    skill = load_skill(skill_id)
    srcs = [s for s in skill.sources if s.difficulty == "exam"]
    assert [s.topic for s in srcs] == [topic]
    assert srcs[0].params == {"tier": "exam"}
    assert list(srcs[0].stages) == ["exam"]


def check_oracle(topic, oracle, seeds=300):
    for seed in range(seeds):
        q = exam(topic, seed)
        assert oracle(q) == canonical(q), (seed, q.text)


def check_answer_space(topic, label, minimum=100, seeds=300):
    answers = [canonical(exam(topic, s))[label] for s in range(seeds)]
    assert len(set(answers)) >= minimum
    assert max(answers.count(a) for a in set(answers)) <= 0.05 * seeds


def check_grading(topic, skill_id, seeds=120):
    skill = load_skill(skill_id)
    shown = 0
    for seed in range(seeds):
        q = exam(topic, seed)
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, (seed, q.answer_text)
        canon = canonical(q)
        for p in q.parts:
            assert p.distractors and p.answer_kind == "number"
            for d in p.distractors:
                shown += 1
                res = grade(skill, q, {**canon, p.label: d})
                assert res.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= 300


def _eval(expr, values):
    local = {f"part_{k}": sympy.Rational(str(v)) for k, v in values.items()}
    return sympy.sympify(expr, locals=local).evalf(30)


def _chain(q, first):
    """Replies that carry the wrong (a) `first` through every later part, or None when a part collapses to 0."""
    replies = {"a": first}
    for p in q.parts[1:]:
        value = _eval(p.follow_through, replies)
        if not value.is_real:
            return None
        shown = show(Fraction(str(value)), dp_of(p.answer_text))
        if Fraction(shown) <= 0 or shown == p.answer_text:
            return None
        replies[p.label] = shown
    return replies


def check_follow_through(topic, skill_id, seeds=60):
    """A wrong (a) used correctly in later parts earns those parts' marks; (a) itself earns none."""
    skill = load_skill(skill_id)
    checked = 0
    for seed in range(seeds):
        q = exam(topic, seed)
        replies = next((r for r in (_chain(q, d) for d in q.parts[0].distractors) if r), None)
        if replies is None:  # every wrong (a) makes a later part impossible, so there is nothing to carry forward
            continue
        checked += 1
        result = grade(skill, q, replies)
        assert result.parts["a"].marks == 0
        for p in q.parts[1:]:
            assert result.parts[p.label].marks == p.marks, (seed, p.label, replies)
            assert result.parts[p.label].follow_through is True
        assert 0 < result.marks < result.max_marks and not result.correct
    assert checked >= 0.7 * seeds


def check_all_wrong(topic, skill_id):
    skill = load_skill(skill_id)
    for seed in range(20):
        q = exam(topic, seed)
        result = grade(skill, q, {p.label: "987654.3" for p in q.parts})
        assert result.marks == 0


def check_deterministic(topic):
    assert exam(topic, 7).to_json() == exam(topic, 7).to_json()
