"""Independent oracle for `pv0a_geometric_threshold`, the first term of a geometric sequence to pass a threshold.

Independent oracle for pv0a_geometric_threshold
(senior-use-geometric-sequences-in-contexts-involving-geometric-growth).

Parses ``question.text``/``question.answer_text`` only and imports NOTHING
from ``mathgen.generators.pv0a_geometric_threshold_build``. Every route below
iterates period by period in exact ``Fraction`` arithmetic (never the
generator's own log-space construction) to re-derive the first period past
the stated threshold, and separately re-asserts the log-space margin
boundary with ``math.log`` on the parsed values.
"""

import functools
import math
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS_1000 = range(1000)
SKILL_ID = "senior-use-geometric-sequences-in-contexts-involving-geometric-growth"

_MONEY_RE = re.compile(r"\$(\d+)")
_PCT_RE = re.compile(r"(\d+)%")


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _iterate(first_value, factor: Fraction, threshold: Fraction, growth: bool) -> int:
    k = 0
    val = Fraction(first_value)
    while True:
        if growth:
            if val > threshold:
                return k
        else:
            if val < threshold:
                return k
        val *= factor
        k += 1
        if k > 500:
            raise AssertionError("oracle: threshold never crossed")


def _margin(threshold, first_value, factor) -> float:
    return (math.log(float(threshold) / float(first_value)) / math.log(float(factor))) % 1.0


# =============================================================================
# easy: growth, period-0, absolute threshold
# =============================================================================


def _oracle_easy(text: str) -> tuple[int, float]:
    amounts = [int(a) for a in _MONEY_RE.findall(text)]
    V, T = amounts[0], amounts[1]
    rate_pct = int(_PCT_RE.search(text).group(1))
    factor = Fraction(100 + rate_pct, 100)
    n = _iterate(V, factor, T, growth=True)
    return n, _margin(T, V, factor)


@pytest.mark.parametrize("difficulty", ["easy"])
def test_oracle_easy(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        n, margin = _oracle_easy(q.text)
        assert str(n) == q.answer_text, (seed, q.text, q.answer_text, n)
        assert 0.05 <= margin <= 0.95, (seed, q.text, margin)
        checked += 1
    assert checked == 1000


# =============================================================================
# medium: decay, period-1 -- reported period is k + 1
# =============================================================================


def _oracle_medium(text: str) -> tuple[int, float]:
    amounts = [int(a) for a in _MONEY_RE.findall(text)]
    V, T = amounts[0], amounts[1]
    rate_pct = int(_PCT_RE.search(text).group(1))
    factor = Fraction(100 - rate_pct, 100)
    k = _iterate(V, factor, T, growth=False)
    return k + 1, _margin(T, V, factor)


@pytest.mark.parametrize("difficulty", ["medium"])
def test_oracle_medium(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        n, margin = _oracle_medium(q.text)
        assert str(n) == q.answer_text, (seed, q.text, q.answer_text, n)
        assert 0.05 <= margin <= 0.95, (seed, q.text, margin)
        checked += 1
    assert checked == 1000


# =============================================================================
# hard: relative threshold (percent of the stated first value, which cancels)
# =============================================================================


def _oracle_hard(text: str) -> tuple[int, float]:
    pcts = [int(p) for p in _PCT_RE.findall(text)]
    rate_pct, pct_threshold = pcts[0], pcts[1]
    growth = "exceed" in text
    period1 = ("In its first hour" in text) or ("In its first hour of decay" in text)
    factor = Fraction(100 + rate_pct, 100) if growth else Fraction(100 - rate_pct, 100)
    k = _iterate(100, factor, pct_threshold, growth)
    n = k + 1 if period1 else k
    return n, _margin(pct_threshold, 100, factor)


@pytest.mark.parametrize("difficulty", ["hard"])
def test_oracle_hard(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        n, margin = _oracle_hard(q.text)
        assert str(n) == q.answer_text, (seed, q.text, q.answer_text, n)
        assert 0.05 <= margin <= 0.95, (seed, q.text, margin)
        checked += 1
    assert checked == 1000


# =============================================================================
# exam: two-stage rate
# =============================================================================

_EXAM_M_RE = re.compile(r"first (\d+) (?:years|weeks)")
_EXAM_RATES_RE = re.compile(r"(\d+)% (?:p\.a\.|each (?:year|week))")


def _oracle_exam(text: str) -> tuple[int, float]:
    amounts = [int(a) for a in _MONEY_RE.findall(text)]
    if amounts:
        V, T = amounts[0], amounts[1]
    else:
        pop_re = re.compile(r"population of (\d+)|colony of (\d+)")
        m = pop_re.search(text)
        V = int(next(g for g in m.groups() if g))
        target_re = re.compile(r"exceed (\d+)")
        T = int(target_re.search(text).group(1))
    m_periods = int(_EXAM_M_RE.search(text).group(1))
    rates = [int(r) for r in _EXAM_RATES_RE.findall(text)]
    r1_pct, r2_pct = rates[0], rates[1]
    r1 = Fraction(100 + r1_pct, 100)
    r2 = Fraction(100 + r2_pct, 100)

    k, val = 0, Fraction(V)
    while True:
        if val > T:
            break
        val *= r1 if k < m_periods else r2
        k += 1
        if k > 500:
            raise AssertionError("oracle: exam threshold never crossed")
    n = k

    if n <= m_periods:
        margin = _margin(T, V, r1)
    else:
        v1 = Fraction(V) * r1**m_periods
        margin = _margin(T, v1, r2)
    return n, margin


@pytest.mark.parametrize("difficulty", ["exam"])
def test_oracle_exam(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        n, margin = _oracle_exam(q.text)
        assert str(n) == q.answer_text, (seed, q.text, q.answer_text, n)
        assert 0.05 <= margin <= 0.95, (seed, q.text, margin)
        checked += 1
    assert checked == 1000


# =============================================================================
# Red-first proof: a deliberately corrupted expectation (+1 on the re-derived
# key, and a flipped growth/decay direction) must FAIL against the real
# generator's output.
# =============================================================================


def test_oracle_fails_red_first_on_a_corrupted_expectation():
    q = _make_from_sources("easy", 0)
    n, _margin = _oracle_easy(q.text)
    corrupted = n + 1
    assert str(corrupted) != q.answer_text, "corrupted +1 expectation must not match -- test is not RED"

    q2 = _make_from_sources("medium", 0)
    amounts = [int(a) for a in _MONEY_RE.findall(q2.text)]
    V, T = amounts[0], amounts[1]
    rate_pct = int(_PCT_RE.search(q2.text).group(1))
    factor = Fraction(100 - rate_pct, 100)
    # Flipped convention: reporting k instead of k + 1.
    k = _iterate(V, factor, T, growth=False)
    assert str(k) != q2.answer_text, "flipped-convention expectation must not match -- test is not RED"

    # Flipped direction: treating the decay context's condition as growth
    # instead resolves immediately (k=0, since the starting value already
    # sits above a decay threshold) -- a visibly wrong answer, not the
    # generator's own reported period.
    flipped_k = _iterate(V, factor, T, growth=True)
    assert str(flipped_k) != q2.answer_text, "flipped-direction expectation must not match -- test is not RED"
