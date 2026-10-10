"""Independent oracle for pythagoras_truss_chain.

Parses ``question.text`` alone (regex out every stated leg/hyp value and the
chain relationship), re-derives h1, h2 (if L=3) and cL with `math.isqrt`
from scratch, and asserts the result matches `int(question.answer_text)` for
the actual asked-for target parsed from the question's final sentence.

Must not import anything from ``mathgen.generators.t3_pythagoras_truss_chain_build``.
"""

import math
import re

import mathgen
import pytest

TOPIC = "pythagoras_truss_chain"
TIERS = ["easy", "medium", "hard"]

_STAGE1_RE = re.compile(r"one leg is (\d+) m and the hypotenuse is (\d+) m")
_STAGE2_HYP_RE = re.compile(r"its own hypotenuse is (\d+) m")
_CLOSING_LEG_RE = re.compile(r"its own other leg is (\d+) m")


def parse_chain(text: str) -> dict:
    """Every (leg, leg, hyp) triple stated or implied by ``text``, parsed
    without any knowledge of how the generator built it."""
    m1 = _STAGE1_RE.search(text)
    assert m1, f"could not find stage 1 in {text!r}"
    a1, c1 = int(m1.group(1)), int(m1.group(2))

    length = 3 if "Triangle 3:" in text else 2

    diff1 = c1**2 - a1**2
    assert diff1 >= 0
    h1 = math.isqrt(diff1)
    assert h1 * h1 == diff1

    c2 = h2 = None
    shared_prev = h1
    if length == 3:
        m2 = _STAGE2_HYP_RE.search(text)
        assert m2, f"could not find stage 2 hypotenuse in {text!r}"
        c2 = int(m2.group(1))
        diff2 = c2**2 - h1**2
        assert diff2 >= 0
        h2 = math.isqrt(diff2)
        assert h2 * h2 == diff2
        shared_prev = h2

    m_last = _CLOSING_LEG_RE.search(text)
    assert m_last, f"could not find closing stage's own other leg in {text!r}"
    a_last = int(m_last.group(1))
    c_last = math.isqrt(a_last**2 + shared_prev**2)
    assert c_last * c_last == a_last**2 + shared_prev**2

    return dict(a1=a1, c1=c1, h1=h1, c2=c2, h2=h2, a_last=a_last, c_last=c_last, length=length, shared_prev=shared_prev)


def parse_target(text: str) -> str:
    """Which quantity the question's own final sentence actually asks for."""
    if re.search(r"Find the length of h1\b", text):
        return "h1"
    if re.search(r"Find the length of h2\b", text):
        return "h2"
    assert re.search(r"Find its hypotenuse, in m\.$", text), f"could not find a target sentence in {text!r}"
    return "cL"


def oracle_answer(text: str) -> int:
    chain = parse_chain(text)
    target = parse_target(text)
    if target == "h1":
        return chain["h1"]
    if target == "h2":
        assert chain["h2"] is not None, "h2 target on a length-2 chain"
        return chain["h2"]
    return chain["c_last"]


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("seed", range(60))
def test_oracle_matches_generator(tier, seed):
    q = mathgen.make(TOPIC, seed=seed, difficulty=tier)
    assert oracle_answer(q.text) == int(q.answer_text), (
        f"{tier} seed {seed}: oracle disagrees with generator (question: {q.text!r})"
    )


def test_easy_is_always_length_two():
    for seed in range(60):
        q = mathgen.make(TOPIC, seed=seed, difficulty="easy")
        assert parse_chain(q.text)["length"] == 2
        assert parse_target(q.text) == "cL"


def test_medium_is_always_length_three_targeting_final():
    for seed in range(60):
        q = mathgen.make(TOPIC, seed=seed, difficulty="medium")
        assert parse_chain(q.text)["length"] == 3
        assert parse_target(q.text) == "cL"


def test_hard_targets_vary_over_all_three_quantities():
    targets = {parse_target(mathgen.make(TOPIC, seed=seed, difficulty="hard").text) for seed in range(60)}
    assert targets == {"h1", "h2", "cL"}, f"only saw targets {targets} over 60 seeds"
