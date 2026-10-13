"""Tests `senior-define-and-use-multiplication-and-division-of-complex-numbers-in-polar-form` (topic
`salg_polar_multiply_divide`) and its exam chain `salg_exam_polar_multiply_divide`. The oracle reads the numbers back out
of the question text, multiplies or divides them as Cartesian complex numbers with `cmath` and reads the modulus and
argument off the result."""

from __future__ import annotations

import cmath
import math
import re
from collections import Counter

import pytest

import mathgen.generators.salg_polar_multiply_divide_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from salg_support import check_exam, check_follow_through, check_single, skill

TOPIC = "salg_polar_multiply_divide"
EXAM = "salg_exam_polar_multiply_divide"
SKILL = skill("senior-define-and-use-multiplication-and-division-of-complex-numbers-in-polar-form")
TIERS = ("easy", "medium", "hard")
_NUMBER = re.compile(r"([zwu]) = (\d+) cis (\d+)°")


def _numbers(text):
    return {name: cmath.rect(int(r), math.radians(int(t))) for name, r, t in _NUMBER.findall(text)}


def _expected(text):
    z = _numbers(text)
    if "zw/u" in text:
        value = z["z"] * z["w"] / z["u"]
    elif "z/w" in text:
        value = z["z"] / z["w"]
    else:
        value = z["z"] * z["w"]
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
        assert abs(q.answer["r"] - r) < 1e-9, (tier, seed, q.text)
        got = q.answer["theta"]
        assert 0 <= got < 360 and min(abs(got - theta), 360 - abs(got - theta)) < 1e-6, (tier, seed, q.text, got, theta)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier, seeds=100)


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(60)]
    assert all("find zw." in t.lower() for t in easy)
    for t in easy:
        angles = [int(a) for _n, _r, a in _NUMBER.findall(t)]
        assert sum(angles) < 360
    medium = [_q("medium", s).text for s in range(60)]
    assert all("z/w" in t for t in medium)
    for t in medium:
        (_, _, a1), (_, _, a2) = _NUMBER.findall(t)
        assert int(a1) > int(a2)
    hard = [_q("hard", s).text for s in range(150)]
    assert any("zw/u" in t for t in hard)
    wrapped = 0
    for t in hard:
        found = _NUMBER.findall(t)
        if "zw/u" in t:
            continue
        a1, a2 = int(found[0][2]), int(found[1][2])
        wrapped += (a1 + a2 >= 360) if "find zw." in t else (a1 < a2)
    assert wrapped >= 40


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(150)]
    assert len(set(answers)) >= 40, len(set(answers))
    assert max(Counter(answers).values()) <= 10


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [bool(p.follow_through) for p in q.parts] == [False, False, True, True]
        assert [p.sensible_range for p in q.parts] == [(0, None), (0, 360), (0, None), (0, 360)]


def test_exam_keys_match_independent_derivation():
    for seed in range(150):
        q = make(EXAM, seed=seed, difficulty="exam")
        z = _numbers(q.text)
        zw, zwu = z["z"] * z["w"], z["z"] * z["w"] / z["u"]
        want = [abs(zw), math.degrees(cmath.phase(zw)) % 360, abs(zwu), math.degrees(cmath.phase(zwu)) % 360]
        for part, w in zip(q.parts, want):
            assert abs(float(part.answer) - w) < 1e-6, (seed, part.label, q.text)
        # (b) needs reducing: the arguments of z and w add to more than a full turn
        angles = [int(a) for _n, _r, a in _NUMBER.findall(q.text)]
        assert angles[0] + angles[1] >= 360


def test_exam_key_accepted_and_every_distractor_rejected():
    check_exam(SKILL, EXAM, seeds=60)


def test_exam_no_follow_through_from_an_unreduced_argument():
    """An argument of 360 or more is outside (b)'s sensible range, so (d) earns nothing from it."""
    for seed in range(20):
        q = make(EXAM, seed=seed, difficulty="exam")
        canon = {p.label: p.answer_text for p in q.parts}
        angles = [int(a) for _n, _r, a in _NUMBER.findall(q.text)]
        raw = angles[0] + angles[1]
        result = grade(SKILL, q, {**canon, "b": str(raw), "d": str(raw - angles[2])})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == 0


def test_exam_follow_through_from_wrong_moduli():
    def wrong(q):
        z = _numbers(q.text)
        r3 = round(abs(z["u"]))
        bad = round(abs(z["z"])) + round(abs(z["w"]))  # moduli added
        if bad % r3:
            return None
        return str(bad), {"c": str(bad // r3)}

    check_follow_through(SKILL, EXAM, "a", wrong, ["c"], seeds=80, min_checked=20)
