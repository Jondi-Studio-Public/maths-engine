"""This file does **not** import that module -- confirmed by its own import list below: only `re`, `statistics`,
`fractions`, `mathgen.make`, and `mathgen.grading`.
"""

from __future__ import annotations

import re
import statistics
from fractions import Fraction

import pytest

from mathgen import make

FNS_TOPIC = "p12d_five_number_summary"
OEM_TOPIC = "p12d_outlier_effect_mean"
TIERS = ("easy", "medium", "hard", "exam")

# The 9-value comma-separated list, wherever it sits in the tier's own wording.
_FNS_VALUES_RE = re.compile(r"(-?\d+(?:,\s*-?\d+){8})")
_FNS_ANSWER_RE = re.compile(r"^min = (-?\d+), Q1 = (-?\d+), median = (-?\d+), Q3 = (-?\d+), max = (-?\d+)$")
# Distractors may carry a half-integer Q1/Q3 (the excludes-median-from-halves
# misconception averages two integers) -- a looser pattern for THOSE only.
_FNS_DISTRACTOR_RE = re.compile(
    r"^min = (-?\d+), Q1 = (-?\d+(?:\.\d+)?), median = (-?\d+), "
    r"Q3 = (-?\d+(?:\.\d+)?), max = (-?\d+)$"
)

_OEM_OUTLIER_RE = re.compile(r"stated outlier value is (-?\d+)")


def _fns_all(tier, seeds=range(60)):
    return [make(FNS_TOPIC, seed=s, difficulty=tier) for s in seeds]


def _oem_all(tier, seeds=range(60)):
    return [make(OEM_TOPIC, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# p12d_five_number_summary oracle
# ============================================================================


def _fns_oracle_answer(question_text: str) -> tuple[int, int, int, int, int]:
    """Re-derive the five-number summary from the rendered PRESENTED
    9-value list alone, via a genuinely different route from construction:
    `statistics.median` of the lower/upper halves (each including the
    overall median), rather than the literal `sorted[2]`/`sorted[4]`/
    `sorted[6]` indexing the generator's own construction uses."""
    m = _FNS_VALUES_RE.search(question_text)
    assert m is not None, f"oracle: no 9-value list found in {question_text!r}"
    values = [int(t) for t in m.group(1).split(",")]
    assert len(values) == 9, values
    ordered = sorted(values)
    lower, upper = ordered[:5], ordered[4:]
    mn = ordered[0]
    q1 = statistics.median(lower)
    md = ordered[4]
    q3 = statistics.median(upper)
    mx = ordered[8]
    return mn, q1, md, q3, mx


@pytest.mark.parametrize("tier", TIERS)
def test_fns_oracle_matches_answer_text_over_60_seeds(tier):
    mismatches = []
    for q in _fns_all(tier):
        oracle = _fns_oracle_answer(q.text)
        m = _FNS_ANSWER_RE.match(q.answer_text)
        assert m is not None, q.answer_text
        actual = tuple(int(v) for v in m.groups())
        if oracle != actual:
            mismatches.append((q.seed, q.text, oracle, actual))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_fns_oracle_answer_shape_is_recognised(tier):
    for q in _fns_all(tier, seeds=range(20)):
        assert _FNS_ANSWER_RE.match(q.answer_text), q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_fns_oracle_every_distractor_disagrees_with_the_oracle(tier):
    """Every offered distractor's five-tuple must differ from the
    oracle's own independently-derived correct tuple -- confirms the
    oracle would in fact challenge a wrong option, not merely agree with
    whatever the generator emits."""
    for q in _fns_all(tier, seeds=range(20)):
        oracle = _fns_oracle_answer(q.text)
        oracle_frac = tuple(Fraction(v) for v in oracle)
        for d in q.distractors:
            m = _FNS_DISTRACTOR_RE.match(d)
            assert m is not None, d
            actual = tuple(Fraction(v) for v in m.groups())
            assert actual != oracle_frac, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"


# ============================================================================
# p12d_outlier_effect_mean oracle
# ============================================================================

# The n-value comma-separated list -- easy(5)/medium(7)/hard,exam(9); tried
# longest-first so a 9-value list is never mistaken for a shorter run inside it.
_OEM_VALUES_RES = [
    re.compile(r"(-?\d+(?:,\s*-?\d+){8})"),  # 9 values
    re.compile(r"(-?\d+(?:,\s*-?\d+){6})"),  # 7 values
    re.compile(r"(-?\d+(?:,\s*-?\d+){4})"),  # 5 values
]


def _oem_parse_values(text: str) -> list[int]:
    for pattern in _OEM_VALUES_RES:
        m = pattern.search(text)
        if m:
            return [int(t) for t in m.group(1).split(",")]
    raise AssertionError(f"oracle: no value list found in {text!r}")


def _oem_oracle_diff(question_text: str) -> Fraction:
    """Re-derive `mean_with - mean_without` from the rendered prompt alone,
    via the closed-form route `(x - mean_without) / n` -- algebraically
    identical to, but computed independently of, the generator's own
    direct-subtraction construction (see the generator module's docstring
    for the derivation proof)."""
    values = _oem_parse_values(question_text)
    n = len(values)
    m = _OEM_OUTLIER_RE.search(question_text)
    assert m is not None, f"oracle: no stated outlier marker in {question_text!r}"
    x = int(m.group(1))
    assert x in values, f"oracle: stated outlier {x} not present in {values}"
    rest = list(values)
    rest.remove(x)
    mean_without = Fraction(sum(rest), n - 1)
    return (Fraction(x) - mean_without) / n


@pytest.mark.parametrize("tier", TIERS)
def test_oem_oracle_matches_answer_text_over_60_seeds(tier):
    mismatches = []
    for q in _oem_all(tier):
        oracle = _oem_oracle_diff(q.text)
        actual = Fraction(q.answer_text)
        if oracle != actual:
            mismatches.append((q.seed, q.text, str(oracle), q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_oem_oracle_every_distractor_disagrees_with_the_oracle(tier):
    for q in _oem_all(tier, seeds=range(20)):
        oracle = _oem_oracle_diff(q.text)
        for d in q.distractors:
            assert Fraction(d) != oracle, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"
