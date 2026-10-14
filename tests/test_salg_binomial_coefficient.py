"""Tests `senior-recognise-the-numbers-as-binomial-coefficients` (topic `salg_binomial_coefficient`) and its exam chain
`salg_exam_binomial_coefficient`. The oracle reads the binomial and the power asked for back out of the question text
and takes the coefficient from SymPy's expansion."""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy as sp

import mathgen.generators.salg_binomial_coefficient_build  # noqa: F401  (registers the topics)
from mathgen import make
from salg_support import check_exam, check_follow_through, check_single, parse, skill

TOPIC = "salg_binomial_coefficient"
EXAM = "salg_exam_binomial_coefficient"
SKILL = skill("senior-recognise-the-numbers-as-binomial-coefficients")
TIERS = ("easy", "medium", "hard")
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_X, _Y = sp.symbols("x y")


def _sup(text):
    return int("".join(_SUP[c] for c in text))


def _binomial(text):
    """(binomial as SymPy, n) for the first `(...)ⁿ` in `text`."""
    body, power = re.search(r"\(([^()]*)\)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", text).groups()
    return parse(body.replace(" ", "")), _sup(power)


def _asked_power(text):
    """The exponent of x in 'coefficient of x…' (1 for a bare x), or None for the 'independent of x' wording."""
    m = re.search(r"coefficient of (xy|x)([⁰¹²³⁴⁵⁶⁷⁸⁹]*)", text)
    return None if m is None else (_sup(m.group(2)) if m.group(2) else 1)


def _oracle(q):
    base, n = _binomial(q.text)
    expanded = sp.expand(base**n)
    if "independent of x" in q.text or "constant term" in q.text:
        return int(expanded.coeff(_X, 0))
    term = re.search(r"coefficient of ([xy⁰¹²³⁴⁵⁶⁷⁸⁹]+)", q.text).group(1)
    if "y" in term:  # x^r y^(n-r) in (x + y)^n
        m = re.fullmatch(r"x([⁰¹²³⁴⁵⁶⁷⁸⁹]*)y([⁰¹²³⁴⁵⁶⁷⁸⁹]*)", term)
        r = _sup(m.group(1)) if m.group(1) else 1
        return int(sp.Poly(expanded, _X, _Y).coeff_monomial(_X**r * _Y ** (n - r)))
    power = _sup(term[1:]) if len(term) > 1 else 1
    return int(expanded.coeff(_X, power))


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def test_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]
    assert SKILL.prerequisites == ["senior-expand-x-y-n-for-small-positive-integers-n"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_sympy(tier):
    for seed in range(150):
        q = _q(tier, seed)
        assert int(q.answer) == _oracle(q), (tier, seed, q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier, seeds=100)


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(60)]
    assert all("(x + y)" in t for t in easy)
    medium = [_q("medium", s).text for s in range(60)]
    assert all(re.search(r"\((x [+-] \d|1 [+-] \dx)\)", t) for t in medium)
    hard = [_q("hard", s).text for s in range(120)]
    assert any("independent of x" in t or "constant term" in t for t in hard)
    assert any(re.search(r"\([23]x [+-] \d\)", t) for t in hard)
    assert all(not re.search(r"\([23]x", t) for t in medium)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(120)]
    assert len(set(answers)) >= (8 if tier == "easy" else 25), len(set(answers))
    assert max(Counter(answers).values()) <= (40 if tier == "easy" else 25)


def test_negative_numbers_give_negative_odd_powers():
    seen = 0
    for seed in range(120):
        q = _q("medium", seed)
        base, n = _binomial(q.text)
        if "- " in q.text and int(q.answer) < 0:
            seen += 1
    assert seen >= 10


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [bool(p.follow_through) for p in q.parts] == [False, True, False, True]
        assert q.parts[0].sensible_range == (0, None)


def test_exam_keys_match_independent_derivation():
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        base, n = _binomial(q.text)
        expanded = sp.expand(base**n)
        r = _sup(re.search(r"coefficient of x([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", q.parts[1].prompt).group(1))
        assert int(q.parts[0].answer) == sp.binomial(n, r)
        assert int(q.parts[1].answer) == expanded.coeff(_X, r)
        assert int(q.parts[2].answer) == expanded.coeff(_X, r + 1)
        assert int(q.parts[3].answer) == sp.expand(base ** (n + 1)).coeff(_X, r + 1)


def test_exam_key_accepted_and_every_distractor_rejected():
    check_exam(SKILL, EXAM, seeds=60)


def test_exam_follow_through_from_a_wrong_binomial_coefficient():
    def wrong(q):
        base, n = _binomial(q.text)
        p, c = sp.Poly(base, _X).all_coeffs()
        r = _sup(re.search(r"coefficient of x([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", q.parts[1].prompt).group(1))
        a_wrong = int(q.parts[0].answer) + 3
        b = a_wrong * p**r * c ** (n - r)
        d = p * b + c * int(q.parts[2].answer)
        return str(a_wrong), {"b": str(b), "d": str(d)}

    check_follow_through(SKILL, EXAM, "a", wrong, ["b", "d"])
