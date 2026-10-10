"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p19f_integral_additivity_build` --
every stated integral value, scalar and interval is parsed back out of the
rendered prose with regular expressions, then combined with plain Python
integer arithmetic written fresh here (no sympy needed -- every value is a
scalar integer, never a symbolic expression).
"""

import re

import pytest

from mathgen import make

SEEDS = range(200)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p19f_integral_additivity_linearity"


# Non-anchored and tolerant of connective wording ("is"/"equals"/"Given
# that ... find") -- the easy tier draws one of three phrasings
# (`_EASY_TEMPLATES` in the generator) that vary only the wording around the
# same three numeric clauses, never the clause order itself.
_EASY_RE = re.compile(
    r"from (?P<a>-?\d+) to (?P<b>-?\d+)\D*?(?P<i1>-?\d+)\D*?"
    r"from (?P=b) to (?P<c>-?\d+)\D*?(?P<i2>-?\d+)\D*?"
    r"f\(x\) from (?P=a) to (?P=c)\.\s*$"
)
_HARD_INV_RE = re.compile(
    r"^The integral of f\(x\) from (?P<a>-?\d+) to (?P<c>-?\d+) is (?P<i1>-?\d+), and "
    r"the integral of f\(x\) from (?P=a) to (?P<b>-?\d+) is (?P<i2>-?\d+)\. "
    r"Find the integral of f\(x\) from (?P=b) to (?P=c)\.$"
)
_LIN_HEAD_RE = re.compile(
    r"from (?P<a>-?\d+) to (?P<b>-?\d+)\D*?(?P<if_>-?\d+)\D*?"
    r"g\(x\) from (?P=a) to (?P=b)\D*?(?P<ig>-?\d+)\D*?"
    r"integral of (?P<combo>.+?) from (?P=a) to (?P=b)\.\s*$"
)
_EXAM_RE = re.compile(
    r"^The integral of f\(x\) from (?P<a>-?\d+) to (?P<b>-?\d+) is (?P<iabf>-?\d+), "
    r"the integral of f\(x\) from (?P=b) to (?P<c>-?\d+) is (?P<ibcf>-?\d+), and "
    r"the integral of g\(x\) from (?P=a) to (?P=c) is (?P<iacg>-?\d+)\. "
    r"Find the integral of (?P<combo>.+?) from (?P=a) to (?P=c)\.$"
)
_COMBO_RE = re.compile(r"^(?P<kterm>-?\d*)f\(x\) (?P<sign>[+-]) (?P<mterm>\d*)g\(x\)$")


def _parse_combo(combo: str) -> tuple[int, int]:
    """ "{k}f(x) {+/-} {m}g(x)" (an empty coefficient means 1, a bare "-"
    means -1) -> (k, m), the two scalar multipliers."""
    m = _COMBO_RE.match(combo)
    assert m is not None, f"could not parse combo: {combo!r}"
    kterm = m.group("kterm")
    k = 1 if kterm == "" else (-1 if kterm == "-" else int(kterm))
    m_abs = 1 if m.group("mterm") == "" else int(m.group("mterm"))
    mval = m_abs if m.group("sign") == "+" else -m_abs
    return k, mval


def _oracle(question) -> int:
    text = question.text
    m = _EASY_RE.search(text)
    if m is not None:
        return int(m.group("i1")) + int(m.group("i2"))
    m = _HARD_INV_RE.match(text)
    if m is not None:
        return int(m.group("i1")) - int(m.group("i2"))
    m = _LIN_HEAD_RE.search(text)
    if m is not None:
        k, mval = _parse_combo(m.group("combo"))
        return k * int(m.group("if_")) + mval * int(m.group("ig"))
    m = _EXAM_RE.match(text)
    if m is not None:
        k, mval = _parse_combo(m.group("combo"))
        i_ab_f = int(m.group("iabf"))
        i_bc_f = int(m.group("ibcf"))
        i_ac_g = int(m.group("iacg"))
        return k * (i_ab_f + i_bc_f) + mval * i_ac_g
    raise AssertionError(f"no oracle pattern matched: {text!r}")


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_answer = _oracle(q)
        assert str(oracle_answer) == q.answer_text, (
            f"seed={seed}/{tier}: oracle {oracle_answer} != rendered answer {q.answer_text}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_answer = _oracle(q)
        for d in q.distractors:
            assert int(d) != oracle_answer, f"seed={seed}/{tier}: distractor {d!r} equals oracle answer\ntext: {q.text}"
