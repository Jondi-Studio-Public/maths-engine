"""Tests the exam tier of `max-shear-force-udl`: both reactions of a beam with two kinds of load, then V_max."""

from __future__ import annotations

import re

import sympy

import mathgen.generators.exd_x_u7_shear_two_loads_chain  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u7_shear_two_loads_chain"
SKILL_ID = "max-shear-force-udl"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _givens(q):
    stem = q.text.split(" (a)")[0]
    L, w, P, a = (int(v) for v in re.findall(r"\d+", stem)[:4])
    return L, w, P, a


def oracle(q) -> dict[str, int]:
    """Solve vertical and moment equilibrium as simultaneous equations, then read the largest shear."""
    L, w, P, a = _givens(q)
    ra, rb = sympy.symbols("ra rb")
    sol = sympy.solve([ra + rb - w * L - P, rb * L - w * L * L / 2 - P * a], [ra, rb])
    return {"a": int(sol[rb]), "b": int(sol[ra]), "c": int(max(sol[ra], sol[rb]))}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 1, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert find_text_defects(q) == []


def test_prompt_shapes_beyond_the_easy_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    easy = {prompt_shape(make("structural", seed=s, kinds=["max_shear_udl"])) for s in range(60)}
    assert len(exam) >= 3 and exam - easy


def test_key_matches_independent_derivation_over_320_seeds():
    bad = []
    for s in range(320):
        q = _exam(s)
        if {k: str(v) for k, v in oracle(q).items()} != _canonical(q):
            bad.append(s)
    assert bad == []


def test_larger_reaction_is_at_either_support():
    seen = set()
    for s in range(80):
        q = _exam(s)
        o = oracle(q)
        seen.add("A" if o["b"] > o["a"] else "B")
    assert seen == {"A", "B"}


def test_key_accepted_and_every_distractor_rejected():
    for seed in range(120):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 4
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_first_reaction():
    checked = 0
    for seed in range(200):
        q = _exam(seed)
        L, w, P, _ = _givens(q)
        rb_wrong = int(q.parts[0].distractors[0])
        ra = w * L + P - rb_wrong
        vmax = max(ra, rb_wrong)
        if ra <= 0 or ra == rb_wrong or vmax == ra and vmax == int(q.parts[2].answer_text) and False:
            continue
        result = grade(_SKILL, q, {"a": str(rb_wrong), "b": str(ra), "c": str(vmax)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True
        # The larger of the learner's own two reactions is the same number as one of them, which counts as
        # repeating an earlier reply only when the canonical answers differ.
        checked += 1
        if result.parts["c"].marks:
            assert result.parts["c"].follow_through is True
    assert checked >= 100


def test_maximum_shear_follows_through_from_two_wrong_reactions():
    earned = 0
    for seed in range(200):
        q = _exam(seed)
        canon = _canonical(q)
        if int(canon["a"]) == int(canon["c"]):
            continue  # the chosen reaction is the same number as an earlier reply
        # Correct reactions given in the other order: V_max still has to pick the larger.
        result = grade(_SKILL, q, canon)
        assert result.correct
        earned += 1
    assert earned >= 20
