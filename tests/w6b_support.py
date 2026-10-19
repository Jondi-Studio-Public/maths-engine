"""Shared checks for the ``test_w6b_*`` files: key accepted, every distractor rejected and tagged, exam marks,
determinism, text quality and answer variety. The per-skill oracle tests live in each ``test_w6b_*`` file."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import sympy

import mathgen.generators  # noqa: F401  (registers every topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TIERS = ("easy", "medium", "hard")


def skill(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def q_of(topic, tier, seed):
    return make(topic, seed=seed, difficulty=tier)


def check_wiring(sk, topic, exam_topic, answer, prerequisites=()):
    assert sk.answer == answer
    assert [(s.topic, s.difficulty) for s in sk.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam_topic, "exam"),
    ]
    assert sk.sources[-1].stages == ["exam"]
    for pre in prerequisites:
        assert pre in sk.prerequisites


def check_tier(sk, topic, tier, seeds=100, min_distractors=2, text_of=None):
    """Key accepted; every distractor rejected, tagged and valued; text clean; deterministic."""
    for seed in range(seeds):
        q = q_of(topic, tier, seed)
        assert grade(sk, q, q.answer_text).correct, (tier, seed, q.text, q.answer_text)
        assert len(q.distractors) >= min_distractors, (tier, seed, q.text, q.distractors)
        for text in q.distractors:
            assert not grade(sk, q, text).correct, (tier, seed, q.text, q.answer_text, text)
            assert q.option_notes[text] in CATALOGUE, (tier, seed, text)
        assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))
        if seed < 25:
            assert q.to_json() == q_of(topic, tier, seed).to_json()


def check_variety(topic, tier, seeds=60, min_texts=45, min_answers=2):
    questions = [q_of(topic, tier, s) for s in range(seeds)]
    assert len({q.text for q in questions}) >= min_texts
    assert len({q.answer_text for q in questions}) >= min_answers


def check_exam_generic(sk, topic, n=80, marks=None, kinds=None):
    """Full marks for the canonical reply; every part's distractor earns nothing; tagged; deterministic."""
    for seed in range(n):
        q = make(topic, seed=seed, difficulty="exam")
        if marks is not None:
            assert [p.marks for p in q.parts] == list(marks)
        if kinds is not None:
            assert [p.answer_kind for p in q.parts] == list(kinds)
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts), (seed, q.answer_text)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(sk, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE
            if p.answer_kind in ("number", "fraction", "expression"):
                assert p.sensible_range is None or p.answer is not None
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        if seed < 20:
            assert q.to_json() == make(topic, seed=seed, difficulty="exam").to_json()


def check_follow_through(sk, topic, source_label, wrong_replies, followed_label, followed_value, n=40):
    """Reply wrongly to part ``source_label`` and consistently in ``followed_label``: the later part keeps its marks.

    ``wrong_replies(q)`` gives replies for the earlier part; ``followed_value(q, reply)`` the later reply that follows
    from it, or None to skip this seed."""
    checked = 0
    for seed in range(n):
        q = make(topic, seed=seed, difficulty="exam")
        canon = {p.label: p.answer_text for p in q.parts}
        later = next(p for p in q.parts if p.label == followed_label)
        for reply in wrong_replies(q):
            follow = followed_value(q, reply)
            if follow is None:
                continue
            result = grade(sk, q, {**canon, source_label: reply, followed_label: follow})
            assert result.parts[source_label].marks == 0, (seed, reply)
            assert result.parts[followed_label].marks == later.marks, (
                seed,
                reply,
                follow,
                result.parts[followed_label],
            )
            checked += 1
    assert checked >= n // 2, checked
    return checked


# -- reading a complex number back out of text ------------------------------------------------------------------

_I = sympy.I


def parse_cx(text):
    """``-2 + 3i`` / ``1 - i`` / ``(√3/2)i`` / ``2√3 - 2i`` style text as a SymPy number."""
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    t = re.sub(r"√(\d+)", r"sqrt(\1)", text).replace("−", "-").replace(" ", "")
    t = re.sub(r"(?<=[\d)])i(?![\w(])", "*I", t)
    t = re.sub(r"(?<![\w*])i(?![\w(])", "I", t)
    return parse_expr(
        t,
        local_dict={"I": _I, "sqrt": sympy.sqrt},
        transformations=standard_transformations + (implicit_multiplication_application,),
    )


COMPLEX_RE = re.compile(r"-?\d*i|-?\d+(?:\s*[+-]\s*\d*i)?|-?\d+")


def numbers_in(text):
    return [int(m) for m in re.findall(r"-?\d+", text)]


def counts(values):
    return Counter(values)


def frac(text):
    return Fraction(text)
