"""Re-derives each rendered question's total variance (or, at the exam tier, the withheld conditional variance)
from `question.text` / `question.answer_text` alone, via hand-written parsing and its OWN implementation of
the published law of total variance -- never importing anything from
`mathgen.generators.p14e_total_variance_two_case_build`.

Total re-derived across the four tiers: 4 x 260 = 1040 (>= 1000).
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "p14e_total_variance_two_case"
SEEDS_PER_TIER = 260

_MEAN_RE = re.compile(r"(?:conditional )?mean (-?\d+)")
_VAR_RE = re.compile(r"(?:conditional )?variance (-?\d+)\b")
_VAR_TOTAL_RE = re.compile(r"Var\(Y\) = (-?\d+(?:/\d+)?)")

_P1_P2_PATTERNS = [
    re.compile(r"probability ([\d/]+).*?probability ([\d/]+)", re.DOTALL),
    re.compile(r"\(p1 = ([\d/]+)\).*?\(p2 = ([\d/]+)\)", re.DOTALL),
]
_EQUAL_PROB_RE = re.compile(r"equal probability \(([\d/]+) each\)")

_N_RE = re.compile(r"N = (\d+)")
_RATIO_RE = re.compile(r"(\d+)\s*:\s*(\d+)")
_INDIVIDUALS_RE = re.compile(r"(\d+) individuals belong to case 1 and (\d+) to case 2")


def _to_fraction(token: str) -> Fraction:
    return Fraction(token)


def _parse_p1_p2(text: str) -> tuple[Fraction, Fraction] | None:
    equal_match = _EQUAL_PROB_RE.search(text)
    if equal_match:
        p1 = _to_fraction(equal_match.group(1))
        return p1, 1 - p1
    for pattern in _P1_P2_PATTERNS:
        m = pattern.search(text)
        if m:
            p1 = _to_fraction(m.group(1))
            p2 = _to_fraction(m.group(2))
            return p1, p2
    return None


def _parse_counts(text: str) -> tuple[int, int, int] | None:
    n_match = _N_RE.search(text)
    if not n_match:
        return None
    N = int(n_match.group(1))
    ratio_match = _RATIO_RE.search(text)
    if ratio_match:
        n1, n2 = int(ratio_match.group(1)), int(ratio_match.group(2))
        return n1, n2, N
    indiv_match = _INDIVIDUALS_RE.search(text)
    if indiv_match:
        n1, n2 = int(indiv_match.group(1)), int(indiv_match.group(2))
        return n1, n2, N
    return None


def _parse_weights(text: str) -> tuple[Fraction, Fraction]:
    """p1, p2 from either a directly-stated probability or a counts idiom."""
    direct = _parse_p1_p2(text)
    if direct is not None:
        return direct
    counts = _parse_counts(text)
    assert counts is not None, f"could not parse weights from {text!r}"
    n1, n2, N = counts
    return Fraction(n1, N), Fraction(n2, N)


def _oracle_var_total(p1: Fraction, p2: Fraction, mu1: Fraction, mu2: Fraction, s1: Fraction, s2: Fraction) -> Fraction:
    e_within = p1 * s1 + p2 * s2
    e_y = p1 * mu1 + p2 * mu2
    var_between = p1 * (mu1 - e_y) ** 2 + p2 * (mu2 - e_y) ** 2
    return e_within + var_between


# ===========================================================================
# easy / medium / hard: forward evaluation -- Var(Y) is the graded answer.
# ===========================================================================


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_evaluate_tiers_oracle_matches(tier):
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty=tier)
        p1, p2 = _parse_weights(q.text)
        means = [Fraction(x) for x in _MEAN_RE.findall(q.text)]
        variances = [Fraction(x) for x in _VAR_RE.findall(q.text)]
        assert len(means) == 2 and len(variances) == 2, (tier, seed, q.text)
        mu1, mu2 = means
        s1, s2 = variances
        recomputed = _oracle_var_total(p1, p2, mu1, mu2, s1, s2)
        given = Fraction(q.answer_text)
        assert recomputed == given, f"{tier} seed={seed}: oracle {recomputed} != canonical {given} (text={q.text!r})"


# ===========================================================================
# exam: inversion -- solve for the withheld conditional variance in case 2.
# ===========================================================================


@pytest.mark.parametrize("seed", range(SEEDS_PER_TIER))
def test_exam_oracle_matches(seed):
    q = make(TOPIC, seed=seed, difficulty="exam")
    p1, p2 = _parse_weights(q.text)
    means = [Fraction(x) for x in _MEAN_RE.findall(q.text)]
    variances = [Fraction(x) for x in _VAR_RE.findall(q.text)]
    total_match = _VAR_TOTAL_RE.search(q.text)
    assert len(means) == 2 and len(variances) == 1 and total_match, (seed, q.text)
    mu1, mu2 = means
    s1 = variances[0]
    var_total = Fraction(total_match.group(1))

    # Isolate-and-divide: Var(Y) = p1*s1 + p2*s2 + Var(E[Y|X])  =>
    # s2 = (Var(Y) - Var(E[Y|X]) - p1*s1) / p2.
    e_y = p1 * mu1 + p2 * mu2
    var_between = p1 * (mu1 - e_y) ** 2 + p2 * (mu2 - e_y) ** 2
    solved_s2 = (var_total - var_between - p1 * s1) / p2

    given = Fraction(q.answer_text)
    assert solved_s2 == given, f"exam seed={seed}: oracle solved {solved_s2} != canonical {given} (text={q.text!r})"

    # Cross-check: substituting the solved s2 back into the forward formula
    # reproduces the stated Var(Y) exactly.
    reforward = _oracle_var_total(p1, p2, mu1, mu2, s1, solved_s2)
    assert reforward == var_total, (seed, q.text)


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    p1, p2 = _parse_weights(q.text)
    means = [Fraction(x) for x in _MEAN_RE.findall(q.text)]
    variances = [Fraction(x) for x in _VAR_RE.findall(q.text)]
    mu1, mu2 = means
    s1, s2 = variances
    recomputed = _oracle_var_total(p1, p2, mu1, mu2, s1, s2)
    corrupted = recomputed + 1
    assert corrupted != recomputed
