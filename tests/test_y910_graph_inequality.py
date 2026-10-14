"""Proofs for `f10-graphical-inequalities-systems`.

The oracle reads the drawn lines from the figure spec (two through-points each) and the points or the stated
line from the text, then decides with SymPy relationals and `sympy.solve`, which is not how the generator keys.
"""

from __future__ import annotations

import re
from fractions import Fraction as F

import pytest
import sympy

from mathgen import figures, make
from mathgen.engine import grade as grade_skill
from mathgen.grading import grade
from mathgen.skills import load_skills

SKILL_IDS = ["f10-graphical-inequality-regions", "f10-graphical-systems-of-lines", "f10-match-inequality-to-line"]
SEEDS = range(80)
_SKILLS = {s.id: s for s in load_skills()[1] if s.id in SKILL_IDS}
_SKILL = _SKILLS["f10-graphical-systems-of-lines"]  # the skill that carries the exam chain used below
SRC = {(src.topic, src.difficulty): src for sk in _SKILLS.values() for src in sk.sources}
X, Y = sympy.symbols("x y")


def _make(topic, tier, seed):
    src = SRC[(topic, tier)]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _line(fig, i):
    (x1, y1), (x2, y2) = fig["lines"][i]["through"]
    m = sympy.Rational(y2 - y1, x2 - x1)
    return m, sympy.Rational(y1) - m * x1


_REL = {
    "strictly above": lambda lhs, rhs: lhs > rhs,
    "on or above": lambda lhs, rhs: lhs >= rhs,
    "strictly below": lambda lhs, rhs: lhs < rhs,
    "on or below": lambda lhs, rhs: lhs <= rhs,
}


def _region_oracle(q) -> list[str]:
    fig, text = q.figure, q.text
    lines = [_line(fig, i) for i in range(len(fig["lines"]))]
    pts = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"([P-U]) \((-?\d+), (-?\d+)\)", text)}
    if not pts:
        pts = {p["label"]: tuple(p["at"]) for p in fig["points"]}
    words = re.findall(r"(strictly above|on or above|strictly below|on or below)", text)
    if not words:
        sign = re.search(r"inequality y ([<>≤≥]) ", text).group(1)
        words = [{">": "strictly above", "≥": "on or above", "<": "strictly below", "≤": "on or below"}[sign]]
        # the stated equation must be the drawn line
        stated = re.search(r"graph of y = ([^.]*)\.", text).group(1)
        assert _parse_rhs(stated) == lines[0]
    out = []
    for label, (px, py) in pts.items():
        ok = all(bool(_REL[w](sympy.Integer(py), m * px + c)) for w, (m, c) in zip(words, lines))
        if ok:
            out.append(label)
    return out


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_region_key_matches_the_oracle(tier):
    for seed in SEEDS:
        q = _make("y910_graph_inequality_region", tier, seed)
        assert q.answer == _region_oracle(q), (tier, seed, q.text)
        assert 1 <= len(q.answer) <= 4


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_region_grader_and_distractors(tier):
    for seed in SEEDS:
        q = _make("y910_graph_inequality_region", tier, seed)
        assert grade("set", q, q.answer_text).correct
        assert len(q.distractors) == 3
        for d in q.distractors:
            assert not grade("set", q, d).correct, (tier, seed, d)
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])


def test_easy_has_a_point_on_the_boundary_and_hard_has_two_lines():
    for seed in SEEDS:
        q = _make("y910_graph_inequality_region", "easy", seed)
        m, c = _line(q.figure, 0)
        assert sum(1 for p in q.figure["points"] if p["at"][1] == m * p["at"][0] + c) == 1
        assert len(q.figure["lines"]) == 1 and len(q.figure["points"]) == 5
        assert len(_make("y910_graph_inequality_region", "hard", seed).figure["lines"]) == 2


def test_medium_has_points_off_the_grid():
    for seed in SEEDS:
        q = _make("y910_graph_inequality_region", "medium", seed)
        (x0, x1), (y0, y1) = q.figure["x_range"], q.figure["y_range"]
        pts = [(int(a), int(b)) for _l, a, b in re.findall(r"([P-U]) \((-?\d+), (-?\d+)\)", q.text)]
        assert sum(1 for a, b in pts if not (x0 <= a <= x1 and y0 <= b <= y1)) >= 2


def _system_oracle(q) -> tuple[sympy.Rational, sympy.Rational]:
    fig = q.figure
    ma, ca = _line(fig, 0)
    if len(fig["lines"]) == 2:
        mb, cb = _line(fig, 1)
    else:
        mb, p, qq = (
            int(g)
            for g in re.search(r"gradient (-?\d+) and passes through the point \((-?\d+), (-?\d+)\)", q.text).groups()
        )
        cb = qq - mb * p
    sol = sympy.solve([sympy.Eq(Y, ma * X + ca), sympy.Eq(Y, mb * X + cb)], [X, Y], dict=True)[0]
    return sol[X], sol[Y]


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_system_key_matches_the_oracle(tier):
    for seed in SEEDS:
        q = _make("y910_graph_system_solution", tier, seed)
        x, y = _system_oracle(q)
        assert (F(str(x)), F(str(y))) == (F(str(q.answer["x"])), F(str(q.answer["y"]))), (tier, seed)
        assert grade("pair", q, q.answer_text).correct
        for d in q.distractors:
            assert not grade("pair", q, d).correct, (tier, seed, d)
        assert len(q.distractors) == 3
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])


def test_system_tiers_differ():
    for seed in SEEDS:
        easy = _make("y910_graph_system_solution", "easy", seed)
        medium = _make("y910_graph_system_solution", "medium", seed)
        hard = _make("y910_graph_system_solution", "hard", seed)
        assert F(str(easy.answer["x"])).denominator == 1 and F(str(easy.answer["y"])).denominator == 1
        assert F(str(medium.answer["x"])).denominator == 2 or F(str(medium.answer["y"])).denominator == 2
        (x0, x1) = hard.figure["x_range"]
        assert len(hard.figure["lines"]) == 1 and not (x0 <= hard.answer["x"] <= x1)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_match_options_and_key(tier):
    for seed in SEEDS:
        q = _make("y910_graph_inequality_match", tier, seed)
        m, c = _line(q.figure, 0)
        px, py = (int(g) for g in re.search(r"point \((-?\d+), (-?\d+)\)", q.text).groups())
        above = py > m * px + c
        included = "is included" in q.text
        sign = ("≥" if included else ">") if above else ("≤" if included else "<")
        assert q.answer_text.startswith(f"y {sign} ")
        stated = q.answer_text.split(f"y {sign} ", 1)[1]
        assert _parse_rhs(stated) == (m, c)
        assert len(q.distractors) == 5 and len(set(q.distractors) | {q.answer_text}) == 6
        assert grade("mcq", q, "0") is not None


def _parse_rhs(text: str):
    """(m, c) from text like '-x/2 + 3', '2x/3 - 1', '3x', 'x - 2'."""
    text = text.replace(" ", "")
    m = re.match(r"^(-?)(\d*)x(?:/(\d+))?", text)
    sign = -1 if m.group(1) else 1
    num = int(m.group(2)) if m.group(2) else 1
    den = int(m.group(3)) if m.group(3) else 1
    rest = text[m.end() :]
    return sympy.Rational(sign * num, den), sympy.Integer(int(rest)) if rest else sympy.Integer(0)


def test_exam_chain_is_consistent_and_follows_through():
    src = SRC[("exd_x_y910_graph_system", "exam")]
    checked = 0
    for seed in range(40):
        q = make(src.topic, seed=seed, **(src.params or {}))
        a, b, c, d = q.parts
        ma, ca = _line(q.figure, 0)
        mb, cb = _line(q.figure, 1)
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert str(ma) == a.answer_text and str(ca) == b.answer_text
        x = (cb - ca) / (ma - mb)
        assert F(c.answer_text) == F(str(x)) and F(d.answer_text) == F(str(ma * x + ca))
        full = {p.label: p.answer_text for p in q.parts}
        res = grade_skill(_SKILL, q, full)
        assert res.correct
        for part in q.parts:
            for dist in part.distractors:
                assert grade_skill(_SKILL, q, {**full, part.label: dist}).parts[part.label].marks == 0
        # a wrong gradient (one of the tagged errors) carried through to the x-coordinate
        for dist in a.distractors:
            wrong_a = sympy.Rational(dist)
            if wrong_a in (0, mb) or wrong_a == ma:
                continue
            x_ft = (cb - ca) / (wrong_a - mb)
            if x_ft.q not in (1, 2) or x_ft in (
                x,
                ca,
                wrong_a,
            ):  # writing an earlier reply again is not following through
                continue
            reply = {
                "a": dist,
                "b": b.answer_text,
                "c": str(float(x_ft)) if x_ft.q == 2 else str(int(x_ft)),
                "d": d.answer_text,
            }
            r = grade_skill(_SKILL, q, reply)
            assert r.parts["a"].marks == 0 and r.parts["c"].marks == c.marks and r.parts["c"].follow_through, (
                seed,
                reply,
            )
            checked += 1
            break
    assert checked >= 3


def test_skills_have_one_answer_kind_each_and_all_tiers():
    assert {k: v.answer for k, v in _SKILLS.items()} == {
        "f10-graphical-inequality-regions": "set",
        "f10-graphical-systems-of-lines": "pair",
        "f10-match-inequality-to-line": "mcq",
    }
    for sk in _SKILLS.values():
        assert [s.difficulty for s in sk.sources] == ["easy", "medium", "hard", "exam"]


def test_determinism_and_variety():
    for topic, tier in SRC:
        if tier == "exam":
            continue
        assert _make(topic, tier, 3).to_json() == _make(topic, tier, 3).to_json()
        assert len({_make(topic, tier, s).answer_text for s in SEEDS}) >= 15
