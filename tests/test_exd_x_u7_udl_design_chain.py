"""Tests the exam tier of `max-bending-moment-udl`: load from the design moment, then shear, then a section moment."""

from __future__ import annotations

import re

import sympy

import mathgen.generators.exd_x_u7_udl_design_chain  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u7_udl_design_chain"
SKILL_ID = "max-bending-moment-udl"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _givens(q):
    L = int(re.search(r"(\d+) m", q.text).group(1))
    M = int(re.search(r"(\d+) kN·m", q.text).group(1))
    x = int(re.search(r"section (\d+) m from one support", q.text).group(1))
    return L, M, x


def oracle(q) -> dict[str, int]:
    """Solve for the load from the design moment by locating the maximum of M(x), then integrate the shear."""
    L, M, x = _givens(q)
    w, s, pos = sympy.symbols("w s pos", positive=True)
    reaction = w * L / 2
    moment_at = reaction * s - w * s**2 / 2
    peak = moment_at.subs(s, L / sympy.Integer(2))  # the shear vanishes at midspan
    load = sympy.solve(sympy.Eq(peak, M), w)[0]
    shear_max = reaction.subs(w, load)
    return {"a": int(load), "b": int(shear_max), "c": int(moment_at.subs({w: load, s: x}))}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert find_text_defects(q) == []


def test_prompt_shapes_beyond_the_easy_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    easy = {prompt_shape(make("structural", seed=s, kinds=["max_bm_udl"])) for s in range(60)}
    assert len(exam) >= 3 and exam - easy


def test_key_matches_independent_derivation_over_320_seeds():
    bad = []
    for s in range(320):
        q = _exam(s)
        if {k: str(v) for k, v in oracle(q).items()} != _canonical(q):
            bad.append(s)
    assert bad == []


def test_key_accepted_and_every_distractor_rejected():
    for seed in range(120):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 5
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_load():
    checked = 0
    for seed in range(200):
        q = _exam(seed)
        L, _, x = _givens(q)
        w_wrong = int(q.parts[0].distractors[0])
        shear = w_wrong * L // 2
        if (w_wrong * x * (L - x)) % 2:
            continue  # a half-kN·m reply is not a whole-number answer
        section = w_wrong * x * (L - x) // 2
        if shear in (w_wrong,) or section in (w_wrong, shear):  # repeating an earlier reply is not following through
            continue
        result = grade(_SKILL, q, {"a": str(w_wrong), "b": str(shear), "c": str(section)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True
        assert result.marks == 3 and not result.correct
        checked += 1
    assert checked >= 50
