"""Tests for `f10-conditional-relative-frequency`: every key is re-derived from the table printed in the stem."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.engine import grade
from mathgen.generators.fr2d_conditional_frequency_build import _CONTEXTS
from tests.fr2d_kit import SEEDS, check_determinism, check_exam, check_options, check_text, check_tiers, question, skill

SKILL = "f10-conditional-relative-frequency"


def _context(stem: str):
    return next(c for c in _CONTEXTS if c["rows"][0] in stem and c["cols"][0] in stem and c["row_title"] in stem)


def _grid(stem: str):
    """The four cells [[A and B, A and not B], [not A and B, not A and not B]] and the text of the lost cell."""
    body = re.search(r"Table \(.*?\): (.*?)\.(?: |$)", stem).group(1)
    rows = [r.split(" — ") for r in body.split("; ")]
    cells = [[None if c == "?" else int(c) for c in row[1:3]] for row in rows[:2]]
    return cells


def _event(ctx, clause: str):
    for axis, key in ((0, "row_ev"), (1, "col_ev")):
        if clause in ctx[key]:
            return axis, ctx[key].index(clause)
    raise AssertionError(clause)


def _records(cells):
    return [(r, c) for r in (0, 1) for c in (0, 1) for _ in range(cells[r][c])]


def _p(cells, x, y=None):
    recs = _records(cells)
    if y is not None:
        recs = [r for r in recs if r[y[0]] == y[1]]
    return Fraction(sum(1 for r in recs if r[x[0]] == x[1]), len(recs))


_ASKS = (
    re.compile(r"probability that (?P<x>.+), (?:given|knowing) that (?P<y>.+)\. Give your"),
    re.compile(r"Given that (?P<y>.+), estimate the probability that (?P<x>.+)\. Give your"),
)


def oracle_cond(stem: str) -> Fraction:
    ctx = _context(stem)
    cells = _grid(stem)
    for pattern in _ASKS:
        if m := pattern.search(stem):
            return _p(cells, _event(ctx, m["x"]), _event(ctx, m["y"]))
    raise AssertionError(stem)


def oracle_hard(stem: str) -> Fraction:
    ctx = _context(stem)
    cells = _grid(stem)
    m = re.search(r"chance that (?P<x>.+), given that (?P<y>.+), than the chance", stem)
    x, y = _event(ctx, m["x"]), _event(ctx, m["y"])
    return _p(cells, x, y) - _p(cells, x)


@pytest.mark.parametrize("tier,oracle", [("easy", oracle_cond), ("medium", oracle_cond), ("hard", oracle_hard)])
def test_key_matches_the_printed_table(tier, oracle):
    for seed in SEEDS:
        q = question(SKILL, tier, seed)
        assert q.answer_kind == "fraction"
        assert Fraction(q.answer["p"], q.answer["q"]) == oracle(q.text), (seed, q.text)
        assert grade(skill(SKILL), q, q.answer_text).correct


def test_easy_shows_totals_and_medium_does_not():
    assert all("Total" in question(SKILL, "easy", s).text for s in SEEDS)
    assert not any("Total" in question(SKILL, "medium", s).text for s in SEEDS)


def test_medium_conditions_on_opposites_and_in_both_directions():
    targets, conditions = set(), set()
    for seed in SEEDS:
        stem = question(SKILL, "medium", seed).text
        ctx = _context(stem)
        m = next(p.search(stem) for p in _ASKS if p.search(stem))
        x, y = _event(ctx, m["x"]), _event(ctx, m["y"])
        targets.add(x)
        conditions.add(y)
    assert {axis for axis, _ in targets} == {0, 1}
    assert {1} <= {i for _, i in conditions} and {0} <= {i for _, i in conditions}


def test_hard_difference_is_positive():
    for seed in SEEDS:
        assert oracle_hard(question(SKILL, "hard", seed).text) > 0


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_options_text_and_determinism(tier):
    check_options(SKILL, tier, min_options=3)
    check_text(SKILL, tier)
    check_determinism(SKILL, tier)


def test_tiers_differ():
    check_tiers(SKILL, exam=True)


# -- exam chain ---------------------------------------------------------------------------------------------------


def _lost_table(stem: str):
    total = int(re.search(r"(\d+) (?:students|days|customers|practice shots|seeds|times)", stem).group(1))
    cells = _grid(stem)
    known = sum(c for row in cells for c in row if c is not None)
    cells[0][0] = total - known
    return cells


def test_exam_chain_matches_the_table():
    sk = skill(SKILL)
    conclusions = set()
    for seed in SEEDS:
        q = question(SKILL, "exam", seed)
        cells = _lost_table(q.text)
        a, b, c, d = q.parts
        assert a.answer == cells[0][0]
        x, y, ny = (0, 0), (1, 0), (1, 1)
        assert Fraction(b.answer["p"], b.answer["q"]) == _p(cells, x, y)
        assert Fraction(c.answer["p"], c.answer["q"]) == _p(cells, x, ny)
        pb, pc = _p(cells, x, y), _p(cells, x, ny)
        expected = "equals" if pb == pc else ("greater" if pb > pc else "less")
        assert expected in d.answer_text
        conclusions.add(expected)
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "fraction", "mcq"]
        assert len(d.distractors) == 3
        assert grade(sk, q, {"a": str(a.answer), "b": b.answer_text, "c": c.answer_text, "d": d.answer_text}).correct
    assert conclusions == {"equals", "greater", "less"}


def test_exam_chain_grades_every_part_and_follows_through():
    check_exam(SKILL)
    sk = skill(SKILL)
    for seed in SEEDS:
        q = question(SKILL, "exam", seed)
        a, b, c, d = q.parts
        wrong_a = a.answer + 1
        cells = _lost_table(q.text)
        nab = cells[1][0]
        carried = Fraction(wrong_a, wrong_a + nab)
        reply = {
            "a": str(wrong_a),
            "b": f"{carried.numerator}/{carried.denominator}",
            "c": c.answer_text,
            "d": d.answer_text,
        }
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == b.marks and result.parts["b"].follow_through
        assert not result.correct
