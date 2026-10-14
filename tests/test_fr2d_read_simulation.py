"""Tests for `f10-read-a-simulation`: every key is re-derived from the digits or results printed in the stem."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.engine import grade
from tests.fr2d_kit import SEEDS, check_determinism, check_exam, check_options, check_text, check_tiers, question, skill

SKILL = "f10-read-a-simulation"


def _rule(stem: str) -> range:
    if m := re.search(r"digits (\d) and (\d)", stem):
        return range(int(m.group(1)), int(m.group(2)) + 1)
    m = re.search(r"digits (\d) to (\d)", stem)
    return range(int(m.group(1)), int(m.group(2)) + 1)


def _listed(stem: str) -> list[str]:
    body = re.search(r"The (\d+) (?:random digits|trials|results) (?:were|was): (.*?)\. ", stem)
    tokens = body.group(2).split()
    assert len(tokens) == int(body.group(1))
    return tokens


def _event(stem: str, a: bool, b: bool) -> bool:
    ask = stem.split("Two digits make one trial.")[-1]
    if "both" in ask:
        return a and b
    if "at least one" in ask:
        return a or b
    if "exactly one" in ask:
        return a != b
    raise AssertionError(ask)


def oracle_easy(stem: str) -> Fraction:
    rule, tokens = _rule(stem), _listed(stem)
    return Fraction(sum(int(t) in rule for t in tokens), len(tokens))


def oracle_medium(stem: str) -> Fraction:
    rule, tokens = _rule(stem), _listed(stem)
    return Fraction(sum(_event(stem, int(t[0]) in rule, int(t[1]) in rule) for t in tokens), len(tokens))


_LETTER = {"red": "R", "blue": "B", "a girl": "G", "a boy": "B"}


def oracle_hard(stem: str) -> Fraction:
    tokens = _listed(stem)
    m = re.search(r"probability that (?P<x>.+?)(?:, given that | when )(?P<y>.+?)\. Give your", stem)

    def read(clause):
        pos = 0 if ("first counter" in clause or "captain is" in clause and "vice" not in clause) else 1
        letter = _LETTER[re.search(r"is (red|blue|a girl|a boy)$", clause).group(1)]
        return pos, letter

    (px, lx), (py, ly) = read(m["x"]), read(m["y"])
    assert px != py
    cond = [t for t in tokens if t[py] == ly]
    return Fraction(sum(t[px] == lx for t in cond), len(cond))


@pytest.mark.parametrize("tier,oracle", [("easy", oracle_easy), ("medium", oracle_medium), ("hard", oracle_hard)])
def test_key_matches_the_printed_results(tier, oracle):
    for seed in SEEDS:
        q = question(SKILL, tier, seed)
        assert q.answer_kind == "fraction"
        assert Fraction(q.answer["p"], q.answer["q"]) == oracle(q.text), (seed, q.text)
        assert grade(skill(SKILL), q, q.answer_text).correct


def test_medium_asks_all_three_events():
    seen = set()
    for seed in SEEDS:
        ask = question(SKILL, "medium", seed).text.split("Two digits make one trial.")[-1]
        seen |= {w for w in ("both", "at least one", "exactly one") if w in ask}
    assert seen == {"both", "at least one", "exactly one"}


def test_hard_asks_in_both_directions():
    directions = set()
    for seed in SEEDS:
        stem = question(SKILL, "hard", seed).text
        m = re.search(r"probability that (?P<x>.+?)(?:, given that | when )(?P<y>.+?)\. Give your", stem)
        directions.add(
            "first-given"
            if re.search(r"(first counter|captain is)", m["y"]) and "vice" not in m["y"]
            else "second-given"
        )
    assert directions == {"first-given", "second-given"}


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_options_text_and_determinism(tier):
    check_options(SKILL, tier, min_options=3)
    check_text(SKILL, tier)
    check_determinism(SKILL, tier)


def test_tiers_differ():
    check_tiers(SKILL, exam=True)


# -- exam chain ---------------------------------------------------------------------------------------------------


def _theory(stem: str, k: int) -> Fraction:
    p = Fraction(k, 10)
    if "both" in stem.split("event of interest is that")[-1]:
        return p * p
    if "at least one" in stem.split("event of interest is that")[-1]:
        return 1 - (1 - p) ** 2
    return 2 * p * (1 - p)


def test_exam_chain_matches_the_printed_trials():
    sk = skill(SKILL)
    for seed in SEEDS:
        q = question(SKILL, "exam", seed)
        rule, tokens = _rule(q.text), _listed(q.text)
        tail = q.text.split("event of interest is that")[-1]
        hits = 0
        for t in tokens:
            a, b = int(t[0]) in rule, int(t[1]) in rule
            hits += (a and b) if "both" in tail else (a or b) if "at least one" in tail else (a != b)
        a_part, b_part, c_part, d_part = q.parts
        assert a_part.answer == hits
        assert Fraction(b_part.answer["p"], b_part.answer["q"]) == Fraction(hits, len(tokens))
        theory = _theory(q.text, len(rule))
        assert Fraction(c_part.answer["p"], c_part.answer["q"]) == theory
        expected = len(tokens) * theory
        assert expected.denominator == 1
        assert d_part.answer == hits - int(expected)
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "fraction", "number"]


def test_exam_chain_grades_every_part_and_follows_through():
    check_exam(SKILL)
    sk = skill(SKILL)
    for seed in SEEDS:
        q = question(SKILL, "exam", seed)
        a, b, c, d = q.parts
        n = len(_listed(q.text))
        wrong_a = a.answer + 1
        carried = Fraction(wrong_a, n)
        reply = {
            "a": str(wrong_a),
            "b": f"{carried.numerator}/{carried.denominator}" if carried.denominator != 1 else str(carried.numerator),
            "c": c.answer_text,
            "d": str(wrong_a - n * Fraction(c.answer["p"], c.answer["q"])),
        }
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == b.marks and result.parts["b"].follow_through
        assert result.parts["d"].marks == d.marks and result.parts["d"].follow_through
