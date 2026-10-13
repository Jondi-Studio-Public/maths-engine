"""Shared checks for the `tests/test_salg_*.py` files: every key grades correct through `mathgen.grading.grade`, every
distractor grades wrong, tags are catalogue ids, and an exam chain's follow-through works."""

from __future__ import annotations

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    split_symbols,
    standard_transformations,
)

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects


def skill(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def check_single(sk, topic, tier, seeds=60, min_distractors=2):
    """Key accepted, every distractor rejected and tagged, no text defects, deterministic."""
    for seed in range(seeds):
        q = make(topic, seed=seed, difficulty=tier)
        assert grade(sk, q, q.answer_text).correct, (tier, seed, q.answer_text)
        assert len(q.distractors) >= min_distractors, (tier, seed)
        for text in q.distractors:
            assert not grade(sk, q, text).correct, (tier, seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE, (tier, seed, text)
        assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()


def check_exam(sk, topic, seeds=60):
    """Full marks for the key, no marks for any part's distractor, tags in the catalogue."""
    for seed in range(seeds):
        q = make(topic, seed=seed, difficulty="exam")
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == make(topic, seed=seed, difficulty="exam").to_json()
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks, (seed, full)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(sk, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE, (seed, p.label, text)


def check_follow_through(sk, topic, label, wrong_reply, later, seeds=30, min_checked=None):
    """A wrong reply to part `label` still earns part `later` when `later` is right for that reply: `wrong_reply(q)`
    gives (reply to `label`, {later label: reply}) or None to skip the seed."""
    checked = 0
    for seed in range(seeds):
        q = make(topic, seed=seed, difficulty="exam")
        spec = wrong_reply(q)
        if spec is None:
            continue
        reply, rest = spec
        canon = {p.label: p.answer_text for p in q.parts}
        result = grade(sk, q, {**canon, label: reply, **rest})
        assert result.parts[label].marks == 0, (seed, reply)
        for lab in later:
            part = result.parts[lab]
            assert part.marks == part.max_marks and part.follow_through is True, (seed, lab, reply, result)
        checked += 1
    assert checked >= (seeds // 2 if min_checked is None else min_checked), checked


_PARSE = (*standard_transformations, split_symbols, implicit_multiplication_application)


def parse(text):
    """A typed expression (`x^3 + 3x^2y`, `-2sqrt(3)/3`) as a SymPy expression, for the tests' own oracles."""
    return parse_expr(text.replace("^", "**"), transformations=_PARSE, local_dict={"i": sympy.I})
