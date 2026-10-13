"""Tests the skill `ucv1-multivariable-chain-rule`: an independent oracle that re-derives every key from the question
TEXT (substitute the path, then differentiate), the typed grader on every key and distractor, and the four-part exam
chain with follow-through."""

from __future__ import annotations

import re

import pytest
import sympy

from tests import fy1ms_support as T

SKILL_ID = "ucv1-multivariable-chain-rule"
TOPIC = "fy1ms_chain_rule"
EXAM = "exd_x_fy1ms_chain_rule"
SK = T.load(SKILL_ID)
TIERS = ("easy", "medium", "hard")
x, y, t, u, v = sympy.symbols("x y t u v")
SYMS = dict(x=x, y=y, t=t, u=u, v=v)


def oracle(q) -> str:
    text = q.text
    m = re.fullmatch(
        r"Let z = (.*?), where x = (.*?) and y = (.*?)\. Find (dz/dt|d²z/dt²|∂z/∂u|∂z/∂v) when (.*)\.", text
    )
    assert m, text
    f = T.parse_plain(m.group(1), **SYMS)
    xs, ys = T.parse_plain(m.group(2), **SYMS), T.parse_plain(m.group(3), **SYMS)
    z_of = f.subs({x: xs, y: ys})
    what, where = m.group(4), m.group(5)
    if what == "dz/dt":
        t0 = int(re.fullmatch(r"t = (-?\d+)", where).group(1))
        return str(sympy.diff(z_of, t).subs(t, t0))
    if what == "d²z/dt²":
        t0 = int(re.fullmatch(r"t = (-?\d+)", where).group(1))
        return str(sympy.diff(z_of, t, 2).subs(t, t0))
    m2 = re.fullmatch(r"u = (-?\d+) and v = (-?\d+)", where)
    u0, v0 = int(m2.group(1)), int(m2.group(2))
    wrt = u if what.endswith("u") else v
    return str(sympy.diff(z_of, wrt).subs({u: u0, v: v0}))


def test_wiring():
    T.assert_wiring(SK, TOPIC, [EXAM])


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(150) if oracle(T.single(TOPIC, tier, s)) != T.single(TOPIC, tier, s).answer_text]
    assert bad == []


def test_each_tier_uses_its_shapes():
    def shape(tier, s):
        text = T.single(TOPIC, tier, s).text
        return "second" if "d²z" in text else "partial" if "∂z" in text else "path"

    assert {shape("easy", s) for s in range(60)} == {"path"}
    assert {shape("medium", s) for s in range(60)} == {"path", "partial"}
    assert {shape("hard", s) for s in range(60)} == {"partial", "second"}


def test_easy_paths_are_straight_and_hard_changes_are_curved():
    for s in range(60):
        text = T.single(TOPIC, "easy", s).text
        assert "t²" not in text.split("where")[1]
    for s in range(60):
        q = T.single(TOPIC, "hard", s)
        if "∂z" in q.text:
            assert re.search(r"[uv]²|uv", q.text.split("where")[1]), q.text


@pytest.mark.parametrize("tier", TIERS)
def test_grading_and_diagnosis(tier):
    T.assert_single_grading(SK, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_clean_text_variety_and_determinism(tier):
    T.assert_clean_variety_and_determinism(TOPIC, tier, distinct_answers=30)


# --- the exam chain --------------------------------------------------------------------------------------------


def exam_oracle(q):
    m = re.fullmatch(
        r"A thin metal plate has temperature T\(x, y\) = (.*?) degrees Celsius at the point \(x, y\), where x and y are in "
        r"centimetres\. A sensor moves across the plate so that at time t seconds it is at x = (.*?), y = (.*?)\.",
        q.stem,
    )
    f = T.parse_plain(m.group(1), **SYMS)
    xs, ys = T.parse_plain(m.group(2), **SYMS), T.parse_plain(m.group(3), **SYMS)
    t0 = int(re.search(r"when t = (\d+)\.", q.parts[0].prompt).group(1))
    dt = sympy.Rational(re.search(r"in the ([\d.]+) seconds", q.parts[3].prompt).group(1))
    rate = sympy.diff(f.subs({x: xs, y: ys}), t).subs(t, t0)
    pt = {x: xs.subs(t, t0), y: ys.subs(t, t0)}
    dp = int(re.search(r"correct to (\d) decimal", q.parts[3].prompt).group(1))
    return dict(a=str(sympy.diff(f, x).subs(pt)), b=str(sympy.diff(f, y).subs(pt)), c=str(rate), d=T.rnd(rate * dt, dp))


def test_exam_shape():
    T.assert_exam_shape(EXAM, ["a", "b", "c", "d"], [1, 1, 2, 2], [False, False, True, True])


def test_exam_keys_match_the_independent_derivation():
    for seed in range(200):
        q = T.exam(EXAM, seed)
        assert {p.label: p.answer_text for p in q.parts} == exam_oracle(q), seed


def test_exam_grading_distractors_and_notes():
    T.assert_exam_grading(SK, EXAM)


def test_follow_through_from_a_wrong_partial_and_a_wrong_rate():
    T.assert_follow_through_credited(SK, EXAM, minimum=150)


def test_exam_answer_space():
    T.assert_exam_variety(EXAM, distinct=100, repeat=6)
