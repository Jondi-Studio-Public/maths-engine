"""Independent oracle for the annuity-duration skill (ACMGM099).

Independent oracle for
``senior-use-a-recurrence-relation-to-model-an-annuity`` (ACMGM099).

Parses the RENDERED question text only (``question.text``/``question.answer_text``) and
imports NOTHING from ``mathgen.generators.pv0a_annuity_duration_build``. Every route below
re-derives the answer from the stated text with ``Fraction`` arithmetic, using reading (a) --
a withdrawal is paid whenever ``A(k)(1 + r) >= d`` -- as the primary route, and separately
re-derives reading (b) -- count while the post-withdrawal balance itself stays ``>= d`` -- from
the same parsed numbers, asserting the two agree (the boundary rule:
construction keeps every draw clear of the point where the three readings in the
contract could disagree).
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS_1000 = range(1000)
SKILL_ID = "senior-use-a-recurrence-relation-to-model-an-annuity"


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _iterate_a(a0, r, d, limit=600):
    """Reading (a)/(c): count full withdrawals until the next one cannot be
    paid; returns (n, A(n-1), A(n)(1+r))."""
    n = 0
    b = Fraction(a0)
    prev_b = None
    grown = b * (1 + r)
    while grown >= d:
        prev_b = b
        b = grown - d
        n += 1
        assert n < limit, "runaway oracle iteration: never depletes"
        grown = b * (1 + r)
    return n, prev_b, grown


def _iterate_b(a0, r, d, limit=600):
    """Reading (b): count while the post-withdrawal balance itself is still
    >= d -- a genuinely different loop condition from reading (a) above."""
    n = 0
    b = Fraction(a0)
    while b >= d:
        b = b * (1 + r) - d
        n += 1
        assert n < limit, "runaway oracle iteration (reading b): never depletes"
    return n


# =============================================================================
# easy -- the recurrence is stated with its numbers substituted
# =============================================================================

_EASY_FORMULA_RE = re.compile(r"A\(n\+1\) = A\(n\)\(([\d.]+)\) - (\d+)")
_MONEY_RE = re.compile(r"\$(\d+)")


def _easy_oracle(text: str):
    gf_s, d_s = _EASY_FORMULA_RE.search(text).groups()
    r = Fraction(gf_s) - 1
    d = Fraction(int(d_s))
    money_vals = {int(m) for m in _MONEY_RE.findall(text)}
    a0_candidates = money_vals - {int(d)}
    assert len(a0_candidates) == 1, (text, money_vals, d)
    a0 = Fraction(a0_candidates.pop())
    return a0, r, d


# =============================================================================
# medium -- annual rate compounded monthly/quarterly, no formula given
# =============================================================================

_MEDIUM_RE = re.compile(
    r"\$(\d+).*?([\d.]+)% per annum.*?compounded (monthly|quarterly).*?fixed \$(\d+) is withdrawn",
    re.S,
)


def _medium_oracle(text: str):
    a0_s, annual_s, freq_word, d_s = _MEDIUM_RE.search(text).groups()
    freq = 12 if freq_word == "monthly" else 4
    r = (Fraction(annual_s) / 100) / freq
    return Fraction(int(a0_s)), r, Fraction(int(d_s))


# =============================================================================
# hard -- the EFFECT of one changed parameter (amount / rate / withdrawal)
# =============================================================================

_HARD_PCT_RE = re.compile(r"([\d.]+)% per annum")


def _hard_oracle(text: str):
    freq = 12 if "monthly" in text else 4
    percents = _HARD_PCT_RE.findall(text)
    moneys = [int(m) for m in _MONEY_RE.findall(text)]
    if len(percents) == 2:
        # "rate" branch: a0, d, annual1, annual2 (both compounded at `freq`).
        a0, d = Fraction(moneys[0]), Fraction(moneys[1])
        annual1, annual2 = Fraction(percents[0]), Fraction(percents[1])
        r1, r2 = (annual1 / 100) / freq, (annual2 / 100) / freq
        n1, _, _ = _iterate_a(a0, r1, d)
        n2, _, _ = _iterate_a(a0, r2, d)
    elif "invested" in text:
        # "amount" branch: annual, d, a0_1, a0_2.
        annual = Fraction(percents[0])
        r = (annual / 100) / freq
        d = Fraction(moneys[0])
        a0_1, a0_2 = Fraction(moneys[1]), Fraction(moneys[2])
        n1, _, _ = _iterate_a(a0_1, r, d)
        n2, _, _ = _iterate_a(a0_2, r, d)
    else:
        # "withdrawal" branch: annual, a0, d1, d2.
        annual = Fraction(percents[0])
        r = (annual / 100) / freq
        a0 = Fraction(moneys[0])
        d1, d2 = Fraction(moneys[1]), Fraction(moneys[2])
        n1, _, _ = _iterate_a(a0, r, d1)
        n2, _, _ = _iterate_a(a0, r, d2)
    return abs(n1 - n2)


# =============================================================================
# exam -- a mid-life change carried forward from the balance actually reached
# =============================================================================

_K_RE = re.compile(r"(\d+)(?:st|nd|rd|th) withdrawal")


def _exam_oracle(text: str):
    freq = 12 if "monthly" in text else 4
    percents = _HARD_PCT_RE.findall(text)
    moneys = [int(m) for m in _MONEY_RE.findall(text)]
    k = int(_K_RE.search(text).group(1))
    if len(percents) == 2:
        # "rate" branch: a0, d, annual1, annual2.
        a0, d = Fraction(moneys[0]), Fraction(moneys[1])
        r1 = (Fraction(percents[0]) / 100) / freq
        r2 = (Fraction(percents[1]) / 100) / freq
        b = a0
        for _ in range(k):
            b = b * (1 + r1) - d
        n2, _, _ = _iterate_a(b, r2, d)
    else:
        # "withdrawal" branch: annual, a0, d1, d2.
        annual = Fraction(percents[0])
        r = (annual / 100) / freq
        a0, d1, d2 = Fraction(moneys[0]), Fraction(moneys[1]), Fraction(moneys[2])
        b = a0
        for _ in range(k):
            b = b * (1 + r) - d1
        n2, _, _ = _iterate_a(b, r, d2)
    return k + n2


# =============================================================================
# The sweep, plus the boundary/reading-agreement assertion
# =============================================================================


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        if difficulty == "easy":
            a0, r, d = _easy_oracle(q.text)
            n, _, _ = _iterate_a(a0, r, d)
            assert str(n) == q.answer_text, (difficulty, seed, q.text, q.answer_text)
        elif difficulty == "medium":
            a0, r, d = _medium_oracle(q.text)
            n, _, _ = _iterate_a(a0, r, d)
            assert str(n) == q.answer_text, (difficulty, seed, q.text, q.answer_text)
        elif difficulty == "hard":
            diff = _hard_oracle(q.text)
            assert str(diff) == q.answer_text, (difficulty, seed, q.text, q.answer_text)
        else:
            total = _exam_oracle(q.text)
            assert str(total) == q.answer_text, (difficulty, seed, q.text, q.answer_text)
        checked += 1
    assert checked == 1000


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_oracle_boundary_readings_a_and_b_agree(difficulty):
    """The named boundary rule: reading (a) (grown balance vs `d`) and
    reading (b) (post-withdrawal balance vs `d`) must give the SAME count for
    every question this skill renders -- proving construction kept every
    draw clear of the point where the three contract readings could diverge.
    Also asserts the count is never sitting exactly on either margin (an
    equality would be the boundary case the contract forbids)."""
    checked = 0
    for seed in range(300):
        q = _make_from_sources(difficulty, seed)
        if difficulty == "easy":
            a0, r, d = _easy_oracle(q.text)
        else:
            a0, r, d = _medium_oracle(q.text)
        n_a, prev_b, fail_grown = _iterate_a(a0, r, d)
        n_b = _iterate_b(a0, r, d)
        assert n_a == n_b, (difficulty, seed, q.text, n_a, n_b)
        assert str(n_a) == q.answer_text, (difficulty, seed, q.text, q.answer_text)
        # No key sits exactly on either boundary: a genuine positive gap on
        # both the success side (A(n-1) vs d) and the failure side
        # (A(n)(1+r) vs d).
        assert prev_b is not None and prev_b > d, (difficulty, seed, q.text)
        assert fail_grown < d, (difficulty, seed, q.text)
        checked += 1
    assert checked == 300
