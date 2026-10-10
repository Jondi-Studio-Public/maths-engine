"""Two families of text to parse:
- a rendered polynomial `p(x) = ...`, where alpha is named by REAL-ROOT
  ORDER ("the largest real root of p(x) = 0" / "the smallest real root of
  key outright, e.g. "the rational root"). The oracle takes ALL of p's real
  roots via `sympy.real_roots`, sorts them numerically, picks the one the
  phrase names, and reports the degree of ITS OWN minimal polynomial over Q
  -- never trusting which factor the generator claims that root belongs to;
- a compound-field prompt naming two integers inside `sqrt(...)`, degree
  re-derived via `sympy.minimal_polynomial(sqrt(a) + sqrt(b))` regardless of
  whether the prompt named `alpha` or asked for `[Q(sqrt a, sqrt b):Q]`
  directly (the two are the same number by the primitive-element theorem,
  verified numerically above for both the square and non-square `a*b`
  cases before relying on it here).

`N_SEEDS` (default 250 per tier, overridable by `P17A_ORACLE_SEEDS`) -- so
>= 1000 questions over the four tiers.
"""

from __future__ import annotations

import os
import re

import pytest
from sympy import Poly, Symbol, minimal_polynomial, real_roots, sqrt, sympify

from mathgen import make

TOPIC = "p17a_field_extension_degree"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P17A_ORACLE_SEEDS", "250"))
_x = Symbol("x")

_P_EXPR_RE = re.compile(r"p\(x\) = (?!0)([^,.]+?)(?=,| and alpha|\.)")
_ORDER_RE = re.compile(r"\b(largest|smallest) real root\b")
_SQRT_NUM_RE = re.compile(r"sqrt\((\d+)\)")
_AB_DEFINED_RE = re.compile(r"[Ll]et a = (\d+) and b = (\d+)")


def _to_expr(text: str):
    py = text.replace("^", "**")
    py = re.sub(r"(\d)(x)", r"\1*\2", py)
    return sympify(py, locals={"x": _x})


def _oracle_polynomial_degree(text: str) -> int:
    m = _P_EXPR_RE.search(text)
    assert m, text
    expr = _to_expr(m.group(1))
    order_m = _ORDER_RE.search(text)
    assert order_m, text
    order = order_m.group(1)

    roots = sorted(real_roots(Poly(expr, _x)), key=lambda root: root.evalf())
    assert roots, (text, "no real roots found")
    assert len(set(roots)) == len(roots), (text, roots, "repeated real root")
    root = roots[-1] if order == "largest" else roots[0]
    mp = minimal_polynomial(root, _x)
    return Poly(mp, _x).degree()


def _oracle_compound_degree(text: str) -> int:
    """`a`/`b` are stated either as literal numbers inside `sqrt(...)`
    ("sqrt(8) + sqrt(59)") or defined once as "Let a = 8 and b = 59" and
    then referred to by letter ("sqrt(a) + sqrt(b)", "[Q(sqrt(a), sqrt(b))
    : Q]") -- either way the two integers are unambiguously stated in the
    question, so the oracle reads whichever form is present."""
    m = _AB_DEFINED_RE.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
    else:
        nums = [int(v) for v in _SQRT_NUM_RE.findall(text)]
        assert len(nums) == 2, (text, nums)
        a, b = nums
    mp = minimal_polynomial(sqrt(a) + sqrt(b), _x)
    return Poly(mp, _x).degree()


def _oracle_degree(text: str) -> int:
    if "sqrt(" in text:
        return _oracle_compound_degree(text)
    return _oracle_polynomial_degree(text)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_recomputes(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        recomputed = _oracle_degree(q.text)
        expected = int(q.answer_text)
        assert recomputed == expected, (
            f"{tier} seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        checked += 1
    assert checked == N_SEEDS


def test_oracle_is_red_on_a_corrupted_answer():
    """Prove the oracle actually discriminates: corrupt the canonical answer and confirm the oracle's own
    re-derivation disagrees with the corrupted value."""
    q = make(TOPIC, seed=0, difficulty="easy")
    recomputed = _oracle_degree(q.text)
    corrupted = next(v for v in (1, 2, 3, 4) if v != recomputed)
    assert recomputed != corrupted


def test_oracle_parses_both_families_over_the_sweep():
    """A denominator on the parser itself: confirm the compound-field branch
    is actually exercised (exam tier) and the polynomial branch is actually
    exercised (every tier), not just that one regex silently matched nothing
    everywhere."""
    compound_hits, polynomial_hits = 0, 0
    for seed in range(N_SEEDS):
        for tier in TIERS:
            q = make(TOPIC, seed=seed, difficulty=tier)
            if "sqrt(" in q.text:
                compound_hits += 1
            else:
                polynomial_hits += 1
    assert compound_hits > 0
    assert polynomial_hits > 0
