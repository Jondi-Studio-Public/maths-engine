"""The polynomial/quadratic string is extracted from the raw prompt text with a generic "longest run of
digits/x/^/+/-//space that contains an x" regex -- independent of which of the twelve prompt templates
produced it -- and handed to sympy's OWN parser/`expand`/`Poly`, a genuinely different implementation from
this module's `join_terms`-based construction and `Poly.degree`/`Poly.LC` in-generator check.

Seed count is `N_SEEDS` (default 250 per tier, >= 1000 questions over the
four tiers), overridable by `P19E_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest
from sympy import Poly, Symbol, expand, sympify

from mathgen import make

DEG_TOPIC = "p19e_poly_degree_leading"
DISC_TOPIC = "p19e_discriminant_zero_count"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P19E_ORACLE_SEEDS", "250"))

X = Symbol("x")

_POLY_RUN_RE = re.compile(r"[0-9x^+\-() ]*x[0-9x^+\-() ]*")


def _extract_poly(text: str) -> str:
    """The longest run of `[0-9x^+-() ]` containing an `x`, from anywhere in
    `text` -- independent of which prompt template produced it (no template
    ever puts another such run, since every other word contains a letter
    outside this charset). A trailing unmatched `(` can survive when the
    polynomial is immediately followed by parenthesised prose ("... x^2
    (terms not written in order of degree)"); trimmed back to the last
    balanced point."""
    candidates = [c.strip() for c in _POLY_RUN_RE.findall(text) if c.strip()]
    assert candidates, f"oracle: no polynomial-shaped run found in {text!r}"
    best = max(candidates, key=len)
    while best.count("(") > best.count(")"):
        best = best[: best.rfind("(")].strip()
    assert best, f"oracle: trimmed to nothing from {text!r}"
    return best


def _to_sympy_str(s: str) -> str:
    """`^` -> `**`, and implicit multiplication (`2x`, `x(`, `2(`, `)(`)
    made explicit -- the minimum needed for sympy's own parser to read the
    generator's plain-text convention, never touching the generator's code."""
    s = s.replace("^", "**")
    s = re.sub(r"(\d)(x)", r"\1*\2", s)
    s = re.sub(r"(x)(\()", r"\1*\2", s)
    s = re.sub(r"(\d)(\()", r"\1*\2", s)
    s = re.sub(r"(\))(\()", r"\1*\2", s)
    return s


def _oracle_degree_lc(text: str) -> tuple[int, int]:
    poly = _extract_poly(text)
    expr = sympify(_to_sympy_str(poly), locals={"x": X})
    p = Poly(expand(expr), X)
    return p.degree(), int(p.LC())


def _oracle_disc_count(text: str) -> tuple[int, int]:
    poly = _extract_poly(text)
    expr = sympify(_to_sympy_str(poly), locals={"x": X})
    p = Poly(expand(expr), X, domain="QQ")
    a = p.coeff_monomial(X**2)
    b = p.coeff_monomial(X)
    c = p.coeff_monomial(1)
    disc = int(b * b - 4 * a * c)
    if disc > 0:
        count = 2
    elif disc == 0:
        count = 1
    else:
        count = 0
    return disc, count


def _all(topic, tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(topic, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# Red-first proof: the oracle helpers themselves catch a corrupted value.
# ============================================================================


def test_oracle_degree_lc_would_reject_a_wrong_pair():
    """Prove the oracle is not vacuously true: check it against a KNOWN
    hand-computed case, and confirm a deliberately wrong pair disagrees."""
    text = "Consider the polynomial P(x) = 3x^3 - x^2 - 7x + 3. State its degree."
    degree, lc = _oracle_degree_lc(text)
    assert (degree, lc) == (3, 3)
    assert (degree, lc) != (3, -3)  # a corrupted leading coefficient
    assert (degree, lc) != (4, 3)  # a corrupted degree


def test_oracle_disc_count_would_reject_a_wrong_pair():
    text = "Consider the quadratic 2x^2 + 9x - 3 = 0. Compute its discriminant."
    disc, count = _oracle_disc_count(text)
    assert (disc, count) == (105, 2)
    assert (disc, count) != (105, 1)  # a corrupted count
    assert (disc, count) != (104, 2)  # a corrupted discriminant


# ============================================================================
# p19e_poly_degree_leading
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_deg_oracle_matches_answer(tier):
    mismatches = []
    for q in _all(DEG_TOPIC, tier):
        oracle = _oracle_degree_lc(q.text)
        actual = (q.answer["degree"], q.answer["coefficient"])
        if oracle != actual:
            mismatches.append((q.seed, q.text, oracle, actual))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_deg_oracle_every_distractor_disagrees(tier):
    for q in _all(DEG_TOPIC, tier, seeds=range(60)):
        oracle = _oracle_degree_lc(q.text)
        for d in q.distractors:
            m = re.match(r"degree = (-?\d+), coefficient = (-?\d+)", d)
            assert m, d
            actual = (int(m.group(1)), int(m.group(2)))
            assert actual != oracle, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"


# ============================================================================
# p19e_discriminant_zero_count
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_disc_oracle_matches_answer(tier):
    mismatches = []
    for q in _all(DISC_TOPIC, tier):
        oracle = _oracle_disc_count(q.text)
        actual = (q.answer["D"], q.answer["zeros"])
        if oracle != actual:
            mismatches.append((q.seed, q.text, oracle, actual))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_disc_oracle_every_distractor_disagrees(tier):
    for q in _all(DISC_TOPIC, tier, seeds=range(60)):
        oracle = _oracle_disc_count(q.text)
        for d in q.distractors:
            m = re.match(r"D = (-?\d+), zeros = (-?\d+)", d)
            assert m, d
            actual = (int(m.group(1)), int(m.group(2)))
            assert actual != oracle, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"
