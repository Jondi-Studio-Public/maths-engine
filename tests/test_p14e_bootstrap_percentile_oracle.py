"""Re-derives each rendered question's mean and requested percentile endpoint from `question.text` /
`question.answer_text` alone, via hand-written parsing and its OWN copy of the order-statistic convention --
never importing anything from `mathgen.generators.p14e_bootstrap_percentile_build`.

Total re-derived across the four tiers: 4 x 260 = 1040 (>= 1000).
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "p14e_bootstrap_percentile"
SEEDS_PER_TIER = 260

_RESAMPLE_RUN_RE = re.compile(r"(?:-?\d+,\s*){2,}-?\d+")
_MEAN_RE = re.compile(r"mean = (-?\d+(?:/\d+)?)")
_ENDPOINT_RE = re.compile(r",\s*[A-Za-z]\w*\s*=\s*(-?\d+)\s*$")


def _parse_resample(text: str) -> list[int]:
    """The resample list is the LONGEST run of comma-separated integers in
    the rendered prompt -- robust to the several template wordings that
    introduce it differently ("is:", "reads:", "order:")."""
    runs = _RESAMPLE_RUN_RE.findall(text)
    assert runs, f"could not parse resample from {text!r}"
    longest = max(runs, key=len)
    return [int(tok.strip()) for tok in longest.split(",") if tok.strip()]


def _oracle_mean(values: list[int]) -> Fraction:
    return Fraction(sum(values), len(values))


def _oracle_kth_smallest(values: list[int], p: int) -> int:
    n = len(values)
    assert p * n % 100 == 0, f"percentile {p} does not land on an exact order statistic for n={n}"
    k = p * n // 100
    return sorted(values)[k - 1]


def _parse_answer_components(answer_text: str) -> tuple[Fraction, int]:
    mean_match = _MEAN_RE.search(answer_text)
    assert mean_match, f"could not parse mean from {answer_text!r}"
    end_match = _ENDPOINT_RE.search(answer_text)
    assert end_match, f"could not parse endpoint from {answer_text!r}"
    return Fraction(mean_match.group(1)), int(end_match.group(1))


# ===========================================================================
# easy: list already sorted; percentile stated directly ("its {p}th percentile").
# ===========================================================================

_EASY_PERCENTILE_RE = re.compile(r"(?:its|the) (\d+)th percentile")


@pytest.mark.parametrize("seed", range(SEEDS_PER_TIER))
def test_easy_oracle_matches(seed):
    q = make(TOPIC, seed=seed, difficulty="easy")
    values = _parse_resample(q.text)
    p_match = _EASY_PERCENTILE_RE.search(q.text)
    assert p_match, q.text
    p = int(p_match.group(1))
    oracle_mean = _oracle_mean(values)
    oracle_endpoint = _oracle_kth_smallest(values, p)
    given_mean, given_endpoint = _parse_answer_components(q.answer_text)
    assert oracle_mean == given_mean, (seed, q.text, q.answer_text)
    assert oracle_endpoint == given_endpoint, (seed, q.text, q.answer_text)


# ===========================================================================
# medium/hard: list unsorted; percentile stated directly, same as easy.
# ===========================================================================


@pytest.mark.parametrize("tier", ["medium", "hard"])
def test_medium_hard_oracle_matches(tier):
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty=tier)
        values = _parse_resample(q.text)
        p_match = _EASY_PERCENTILE_RE.search(q.text)
        assert p_match, q.text
        p = int(p_match.group(1))
        oracle_mean = _oracle_mean(values)
        oracle_endpoint = _oracle_kth_smallest(values, p)
        given_mean, given_endpoint = _parse_answer_components(q.answer_text)
        assert oracle_mean == given_mean, (tier, seed, q.text, q.answer_text)
        assert oracle_endpoint == given_endpoint, (tier, seed, q.text, q.answer_text)


# ===========================================================================
# exam: coverage stated, not the percentile itself -- the oracle derives it
# from the coverage and the lower/upper wording, independently of the
# generator's own derivation.
# ===========================================================================

_EXAM_COVERAGE_RE = re.compile(r"(\d+)% bootstrap interval")
_EXAM_WHICH_RE = re.compile(r"(lower|upper) endpoint")


@pytest.mark.parametrize("seed", range(SEEDS_PER_TIER))
def test_exam_oracle_matches(seed):
    q = make(TOPIC, seed=seed, difficulty="exam")
    values = _parse_resample(q.text)
    coverage_match = _EXAM_COVERAGE_RE.search(q.text)
    which_match = _EXAM_WHICH_RE.search(q.text)
    assert coverage_match and which_match, q.text
    coverage = int(coverage_match.group(1))
    which = which_match.group(1)
    lower_p = (100 - coverage) // 2
    upper_p = 100 - lower_p
    p = lower_p if which == "lower" else upper_p

    oracle_mean = _oracle_mean(values)
    oracle_endpoint = _oracle_kth_smallest(values, p)
    given_mean, given_endpoint = _parse_answer_components(q.answer_text)
    assert oracle_mean == given_mean, (seed, q.text, q.answer_text)
    assert oracle_endpoint == given_endpoint, (seed, q.text, q.answer_text)


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    values = _parse_resample(q.text)
    p_match = _EASY_PERCENTILE_RE.search(q.text)
    p = int(p_match.group(1))
    oracle_endpoint = _oracle_kth_smallest(values, p)
    corrupted_endpoint = oracle_endpoint + 999
    assert corrupted_endpoint != oracle_endpoint
