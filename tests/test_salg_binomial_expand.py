"""Tests `senior-expand-x-y-n-for-small-positive-integers-n` (topic `salg_binomial_expand`) and its exam chain
`salg_exam_binomial_expand`. The oracle reads the binomial and its power back out of the question text and expands it
with SymPy."""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy as sp

import mathgen.generators.salg_binomial_expand_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from salg_support import check_exam, check_follow_through, check_single, parse, skill

TOPIC = "salg_binomial_expand"
EXAM = "salg_exam_binomial_expand"
SKILL = skill("senior-expand-x-y-n-for-small-positive-integers-n")
TIERS = ("easy", "medium", "hard")
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_POWER_RE = re.compile(r"\(([^()]*)\)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)")


def _power(text):
    """The binomial and exponent in the stem, as a SymPy power."""
    body, sup = _POWER_RE.search(text).groups()
    n = int("".join(_SUP[c] for c in sup))
    return parse(body.replace(" ", "")) ** n, n, body


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def test_wiring():
    assert SKILL.answer == "expression"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_sympy_expansion(tier):
    for seed in range(100):
        q = _q(tier, seed)
        power, n, _body = _power(q.text)
        assert sp.expand(power - parse(q.answer_text)) == 0, (seed, q.text)
        assert q.answer_form == "expanded"


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier, seeds=80, min_distractors=3)


def test_unexpanded_power_is_not_accepted():
    for tier in TIERS:
        for seed in range(30):
            q = _q(tier, seed)
            _power_expr, n, body = _power(q.text)
            assert not grade(SKILL, q, f"({body})^{n}").correct, (tier, seed, q.text)


def test_tiers_differ_in_shape():
    shapes = {}
    for tier in TIERS:
        rows = [_power(_q(tier, s).text) for s in range(80)]
        shapes[tier] = rows
    # easy: both terms have coefficient 1; medium: the second term is a number; hard: a leading coefficient too.
    assert all(re.fullmatch(r"[a-z] \+ [a-z]", body) for _p, _n, body in shapes["easy"])
    assert all(re.fullmatch(r"[a-z] [+-] \d+", body) for _p, _n, body in shapes["medium"])
    assert all(re.match(r"[23][a-z] ", body) for _p, _n, body in shapes["hard"])
    assert {n for _p, n, _b in shapes["easy"]} == {3, 4, 5, 6}


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(120)]
    assert len(set(answers)) >= (15 if tier == "easy" else 25), len(set(answers))
    assert max(Counter(answers).values()) <= 12


def test_negative_numbers_alternate_signs():
    seen = 0
    for seed in range(80):
        q = _q("medium", seed)
        if " - " in q.text:
            terms = re.findall(r"[+-] ", q.answer_text)
            assert terms == [("- ", "+ ")[k % 2] for k in range(len(terms))], q.answer_text
            seen += 1
    assert seen >= 20


# -- exam chain ---------------------------------------------------------------------------------------------------


def _exam_setup(q):
    """The variable, the shift and the two powers, read back out of the question text."""
    letter, sign, num = re.search(r"binomial ([a-z]) ([+-]) (\d+)\.", q.text).groups()
    a = int(num) * (1 if sign == "+" else -1)
    m = _power(q.parts[0].prompt)[1]
    n = _power(q.parts[1].prompt)[1]
    return sp.Symbol(letter), a, m, n


def test_exam_shape_and_marks():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["expression", "expression", "number", "number"]
        assert [bool(p.follow_through) for p in q.parts] == [False, True, False, True]


def test_exam_varies_letter_and_power():
    seen = set()
    for seed in range(60):
        v, a, m, n = _exam_setup(make(EXAM, seed=seed, difficulty="exam"))
        assert n == m + 1
        seen.add((str(v), m))
    assert len({v for v, _m in seen}) >= 3 and {m for _v, m in seen} == {3, 4}


def test_exam_keys_match_independent_derivation():
    for seed in range(120):
        q = make(EXAM, seed=seed, difficulty="exam")
        v, a, m, n = _exam_setup(q)
        lower = sp.expand((v + a) ** m)
        higher = sp.expand((v + a) ** n)
        assert sp.expand(parse(q.parts[0].answer_text) - lower) == 0
        assert sp.expand(parse(q.parts[1].answer_text) - higher) == 0
        assert int(q.parts[2].answer) == sp.Poly(higher, v).coeff_monomial(v**2)
        assert int(q.parts[3].answer) == sum(sp.Poly(higher, v).all_coeffs())


def test_exam_key_accepted_and_every_distractor_rejected():
    check_exam(SKILL, EXAM, seeds=60)


def test_exam_follow_through_from_a_wrong_power():
    def wrong(q):
        # (a) wrong: the middle terms dropped; (b) multiplied from it; (d) then summed at v = 1.
        v, a, _m, _n = _exam_setup(q)
        wrong_a = q.parts[0].distractors[1]
        base = parse(_power(q.parts[1].prompt)[2].replace(" ", ""))
        b = sp.expand(parse(wrong_a) * base)
        return wrong_a, {"b": str(b).replace("**", "^"), "d": str(b.subs(v, 1))}

    check_follow_through(SKILL, EXAM, "a", wrong, ["b", "d"])
