"""Tests `senior-use-de-moivre-s-theorem-for-integral-powers` (topic `salg_de_moivre_power`) and its exam chain
`salg_exam_de_moivre_power`. The oracle reads the number and the power back out of the question text, raises the
Cartesian complex number to the power with `cmath` (and SymPy for the exam chain's exact answer) and reads the modulus
and argument off the result."""

from __future__ import annotations

import cmath
import math
import re
from collections import Counter

import pytest
import sympy as sp

import mathgen.generators.salg_de_moivre_power_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from salg_support import check_exam, check_follow_through, check_single, parse, skill

TOPIC = "salg_de_moivre_power"
EXAM = "salg_exam_de_moivre_power"
SKILL = skill("senior-use-de-moivre-s-theorem-for-integral-powers")
TIERS = ("easy", "medium", "hard")
_POLAR_STEM = re.compile(r"z = [^.]*cis")
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}


def _power(text):
    return int("".join(_SUP[c] for c in re.search(r"z([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", text).group(1)))


def _number(text):
    """The complex number z in the stem, as an exact SymPy number."""
    polar = re.search(r"z = (\d*) ?cis (\d+)°", text)
    if polar:
        return sp.nsimplify(int(polar.group(1) or 1) * sp.exp(sp.I * sp.pi * int(polar.group(2)) / 180))
    body = re.search(r"z = ([^.,]*?)(?:\.|, find| Find)", text).group(1)
    body = body.replace("√3", "sqrt(3)").replace(" ", "")
    body = re.sub(r"i(?=sqrt)", "I*", body)
    return sp.sympify(body.replace("i", "I"))


def _expected(text):
    z = complex(_number(text))
    value = z ** _power(text)
    return abs(value), math.degrees(cmath.phase(value)) % 360


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def test_wiring():
    assert SKILL.answer == "pair"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_cartesian_arithmetic(tier):
    for seed in range(150):
        q = _q(tier, seed)
        r, theta = _expected(q.text)
        assert abs(q.answer["r"] - r) < 1e-6 * max(1, r), (tier, seed, q.text, q.answer_text)
        got = q.answer["theta"]
        assert 0 <= got < 360 and min(abs(got - theta), 360 - abs(got - theta)) < 1e-6, (tier, seed, q.text, got, theta)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier, seeds=100)


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(60)]
    assert all(_POLAR_STEM.search(t) and _power(t) in (2, 3) for t in easy)
    for t in easy:
        a = int(re.search(r"cis (\d+)°", t).group(1))
        assert _power(t) * a < 360
    medium = [_q("medium", s).text for s in range(120)]
    assert all(_POLAR_STEM.search(t) for t in medium)
    assert any(_power(t) < 0 for t in medium) and any(_power(t) >= 5 for t in medium)
    assert all(_power(t) * int(re.search(r"cis (\d+)°", t).group(1)) not in range(0, 360) for t in medium)
    hard = [_q("hard", s).text for s in range(60)]
    assert not any(_POLAR_STEM.search(t) for t in hard)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(150)]
    assert len(set(answers)) >= 20, len(set(answers))
    assert max(Counter(answers).values()) <= 16


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 3]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "expression"]
        assert [bool(p.follow_through) for p in q.parts] == [False, False, True, True]
        assert [p.sensible_range for p in q.parts][:3] == [(0, None), (0, 360), (0, None)]


def test_exam_keys_match_independent_derivation():
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        z = _number(q.text.replace("Let z", "z"))
        n = _power(q.parts[2].prompt)
        value = sp.expand(z**n)
        assert float(q.parts[0].answer) == pytest.approx(abs(complex(z)))
        assert float(q.parts[1].answer) == pytest.approx(math.degrees(cmath.phase(complex(z))) % 360)
        assert float(q.parts[2].answer) == pytest.approx(abs(complex(value)))
        got = complex(parse(q.parts[3].answer_text))
        assert abs(got - complex(value)) < 1e-9, (seed, q.text, q.parts[3].answer_text)


def test_exam_key_accepted_and_every_distractor_rejected():
    check_exam(SKILL, EXAM, seeds=60)


def test_exam_cartesian_answer_accepts_other_spellings():
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        canon = {p.label: p.answer_text for p in q.parts}
        z = _number(q.text.replace("Let z", "z"))
        n = _power(q.parts[2].prompt)
        value = sp.expand(z**n)
        spelled = str(value).replace("I", "i").replace("**", "^")
        assert grade(SKILL, q, {**canon, "d": spelled}).parts["d"].marks == 3, (seed, spelled)


def test_exam_follow_through_from_a_wrong_argument():
    def wrong(q):
        z = _number(q.text.replace("Let z", "z"))
        n = _power(q.parts[2].prompt)
        ref = int(q.parts[1].answer)
        bad = (ref + 90) % 360  # a wrong argument, still inside 0..360
        if (n * bad) % 360 == (n * ref) % 360:
            return None
        value = sp.nsimplify(2**n * (sp.cos(sp.pi * n * bad / 180) + sp.I * sp.sin(sp.pi * n * bad / 180)))
        text = str(sp.expand(value)).replace("**", "^").replace("I", "i")
        return str(bad), {"d": text}

    check_follow_through(SKILL, EXAM, "b", wrong, ["d"], seeds=40, min_checked=15)
