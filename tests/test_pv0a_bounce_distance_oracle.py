"""Independent oracle for `pv0a_bounce_distance`, the total distance of a bouncing ball.

Independent oracle for pv0a_bounce_distance
(senior-use-geometric-sequences-to-model-and-analyse-numerically).

Parses ``question.text``/``question.answer_text`` only and imports NOTHING from
``mathgen.generators.pv0a_bounce_distance_build``. Every route below SIMULATES the ball
stroke by stroke -- building an explicit list of every individual up/down distance and
summing it -- rather than the generator's own running-series-sum accumulation, so a slip
in either route is caught. Every re-derived exact total is also checked against the 2dp
rounding-boundary rule with ``rounds_unambiguously``.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import round_half_up, rounds_unambiguously

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS_1000 = range(1000)
SKILL_ID = "senior-use-geometric-sequences-to-model-and-analyse-numerically"

_H_RE = re.compile(r"from (?:a height of )?(\d+(?:\.\d+)?) m")
_R_RE = re.compile(r"(\d+(?:\.\d+)?)% of")
_K_RE = re.compile(r"(\d+)(?:st|nd|rd|th) (?:time|impact|rebound|bounce)")
_ORD_PAIR_RE = re.compile(
    r"(\d+)(?:st|nd|rd|th) rebound (?:reaches|rises to) (\d+(?:\.\d+)?) m and its (\d+)(?:st|nd|rd|th) "
    r"rebound (?:reaches|rises to) (\d+(?:\.\d+)?) m"
)
# Distinct from `_K_RE` above: the hard tier's TWO stated rebound heights are
# also phrased as "Its Nth rebound reaches ... m", which `_K_RE` would
# wrongly match first. The stated ENDPOINT is phrased only as "for the Nth
# time"/"Nth impact" (impact wording) or "top of its Nth rebound" (top
# wording) -- never as a bare "Its Nth rebound".
_K_HARD_RE = re.compile(
    r"(?:(?:for the|by the) (?P<k_impact>\d+)(?:st|nd|rd|th) (?:time|impact)"
    r"|top of its (?P<k_top>\d+)(?:st|nd|rd|th) rebound)",
    re.IGNORECASE,
)
_X_RE = re.compile(r"(?:less than|lower than) (\d+(?:\.\d+)?) m", re.IGNORECASE)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _h_from(text: str) -> Fraction:
    return Fraction(_H_RE.search(text).group(1))


def _r_from(text: str) -> Fraction:
    raw = _R_RE.search(text).group(1)
    return Fraction(raw) / 100


def _k_from(text: str) -> int:
    return int(_K_RE.search(text).group(1))


def _strokes_to_impact(h: Fraction, r: Fraction, k: int) -> Fraction:
    """Explicit list of every stroke up to and including the k-th impact:
    the initial drop, then EVERY earlier rebound's rise and fall listed
    separately (not accumulated via a series-sum formula), summed at the end."""
    strokes = [Fraction(h)]  # the initial drop, once
    for i in range(1, k):
        rebound_height = Fraction(h) * r**i
        strokes.append(rebound_height)  # rise
        strokes.append(rebound_height)  # fall
    return sum(strokes)


def _strokes_to_top(h: Fraction, r: Fraction, k: int) -> Fraction:
    return _strokes_to_impact(h, r, k) + Fraction(h) * r**k


def _rounded(total: Fraction) -> Fraction:
    assert rounds_unambiguously(float(total), 2), ("exact total sits on the 2dp rounding boundary", total)
    return round_half_up(total, 2)


# =============================================================================
# easy: h, r, k stated; endpoint = the k-th impact
# =============================================================================


@pytest.mark.parametrize("difficulty", ["easy"])
def test_oracle_easy(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        h, r, k = _h_from(q.text), _r_from(q.text), _k_from(q.text)
        expected = _rounded(_strokes_to_impact(h, r, k))
        assert str(expected) == q.answer_text or f"{float(expected):.2f}" == q.answer_text, (
            seed,
            q.text,
            q.answer_text,
            expected,
        )
        checked += 1
    assert checked == 1000


# =============================================================================
# medium: h, r, k stated; endpoint = the TOP of the k-th rebound
# =============================================================================


@pytest.mark.parametrize("difficulty", ["medium"])
def test_oracle_medium(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        h, r, k = _h_from(q.text), _r_from(q.text), _k_from(q.text)
        expected = _rounded(_strokes_to_top(h, r, k))
        assert f"{float(expected):.2f}" == q.answer_text, (seed, q.text, q.answer_text, expected)
        checked += 1
    assert checked == 1000


# =============================================================================
# hard: h and r NOT stated -- two consecutive rebound heights given instead
# =============================================================================


@pytest.mark.parametrize("difficulty", ["hard"])
def test_oracle_hard(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        m = _ORD_PAIR_RE.search(q.text)
        assert m, (seed, q.text)
        _ord1, h1_s, _ord2, h2_s = m.groups()
        h1, h2 = Fraction(h1_s), Fraction(h2_s)
        r = h2 / h1
        start = int(_ord1_from(q.text))
        h = h1 / r**start
        km = _K_HARD_RE.search(q.text)
        to_top = km.group("k_top") is not None
        k = int(km.group("k_top") if to_top else km.group("k_impact"))
        expected = _rounded(_strokes_to_top(h, r, k) if to_top else _strokes_to_impact(h, r, k))
        assert f"{float(expected):.2f}" == q.answer_text, (seed, q.text, q.answer_text, expected)
        checked += 1
    assert checked == 1000


def _ord1_from(text: str) -> int:
    return int(_ORD_PAIR_RE.search(text).group(1))


# =============================================================================
# exam: a threshold decides the endpoint
# =============================================================================


@pytest.mark.parametrize("difficulty", ["exam"])
def test_oracle_exam(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        h, r, x = _h_from(q.text), _r_from(q.text), Fraction(_X_RE.search(q.text).group(1))
        i = 1
        while True:
            height = h * r**i
            # Boundary assertion: X sits > 5% (relative) away
            # from every candidate rebound height, so the first-below search
            # is never itself ambiguous.
            assert abs(x - height) / height > Fraction(1, 20), (seed, q.text, i, height, x)
            if height < x:
                k_first_below = i
                break
            i += 1
            assert i < 60, (seed, q.text, "threshold never crossed")
        expected = _rounded(_strokes_to_impact(h, r, k_first_below + 1))
        assert f"{float(expected):.2f}" == q.answer_text, (seed, q.text, q.answer_text, expected)
        checked += 1
    assert checked == 1000


# =============================================================================
# Red-first proof: a deliberately corrupted expectation must FAIL against the
# real generator's output.
# =============================================================================


def test_oracle_fails_red_first_on_a_corrupted_expectation():
    q = _make_from_sources("easy", 0)
    h, r, k = _h_from(q.text), _r_from(q.text), _k_from(q.text)
    correct = _rounded(_strokes_to_impact(h, r, k))
    assert f"{float(correct):.2f}" == q.answer_text  # sanity: the real route matches first

    # +1 on the re-derived key.
    corrupted = correct + 1
    assert f"{float(corrupted):.2f}" != q.answer_text, "corrupted +1 expectation must not match -- test is not RED"

    # Flipped convention: comparing the impact total against the TOP total.
    top_total = _rounded(_strokes_to_top(h, r, k))
    assert f"{float(top_total):.2f}" != q.answer_text, (
        "flipped impact/top expectation must not match -- test is not RED"
    )

    q2 = _make_from_sources("medium", 0)
    h2, r2, k2 = _h_from(q2.text), _r_from(q2.text), _k_from(q2.text)
    # Flipped convention the other way: comparing medium's TOP-tier text
    # against the plain IMPACT total (this deliberately-wrong route is not
    # itself guaranteed clear of the 2dp rounding boundary, so round it
    # directly rather than through `_rounded`'s boundary assertion).
    impact_total = round_half_up(_strokes_to_impact(h2, r2, k2), 2)
    assert f"{float(impact_total):.2f}" != q2.answer_text, (
        "flipped top/impact expectation must not match -- test is not RED"
    )
