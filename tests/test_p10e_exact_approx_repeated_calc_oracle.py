"""Independent oracle for f10-exact-approximate-repeated-calculation. Parses ONLY what a learner is given
(`question.text`, `question.answer_text`, `question.distractors`) and never imports anything from
`mathgen.generators.p10e_exact_approx_repeated_calc_build`.

Rebuilds both pathways from the parsed operations list using sympy for the
exact pathway (`sympify(...).evalf(30)`, exact Fraction literals, never a
decimal literal) and a fresh Fraction/round-half-up loop for the approximate
pathway, then asserts the rendered `answer_text` is their discrepancy.
"""

import math
import re
from decimal import Decimal
from fractions import Fraction

import sympy as sp

from mathgen import make
from mathgen.core import round_half_up, stated_precision_tolerance

TOPIC = "p10e_exact_approx_repeated_calc"

_HEADER_RE = re.compile(r"Start with (\S+)\. (.+), in that order\. Work out")
_D_RE = re.compile(r"rounding only the FINAL result to (\d) decimal places?")
_MUL_RE = re.compile(r"multiply by (\S+)$", re.I)
_ADD_RE = re.compile(r"add (\S+)$", re.I)
_PI_RE = re.compile(r"multiply by pi", re.I)


def _parse_ops(clauses):
    ops = []
    for clause in clauses:
        clause = clause.strip().rstrip(".")
        if _PI_RE.search(clause):
            ops.append(("mul_pi", None))
            continue
        m = _MUL_RE.search(clause)
        if m:
            ops.append(("mul", Fraction(m.group(1))))
            continue
        m = _ADD_RE.search(clause)
        if m:
            ops.append(("add", Fraction(m.group(1))))
            continue
        raise AssertionError(f"unrecognised operation clause: {clause!r}")
    return ops


def _oracle_exact_final(v0: Fraction, ops, d: int) -> Fraction:
    s = f"({v0.numerator}/{v0.denominator})"
    for kind, k in ops:
        if kind == "mul":
            s = f"({s})*({k.numerator}/{k.denominator})"
        elif kind == "add":
            s = f"({s})+({k.numerator}/{k.denominator})"
        else:
            s = f"({s})*pi"
    val = sp.sympify(s).evalf(30)
    return round_half_up(Fraction(Decimal(str(val))), d)


def _oracle_approx_final(v0: Fraction, ops, d: int) -> Fraction:
    val = v0
    for kind, k in ops:
        if kind == "mul":
            val = val * k
        elif kind == "add":
            val = val + k
        else:
            val = Fraction(float(val) * math.pi)
        val = round_half_up(val, d)
    return val


def _re_derive(question):
    m = _HEADER_RE.match(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    v0 = Fraction(m.group(1))
    ops_text = m.group(2)
    clauses = [c.strip() for c in ops_text.split("; then ")]
    ops = _parse_ops(clauses)
    d_match = _D_RE.search(question.text)
    assert d_match
    d = int(d_match.group(1))

    exact_final = _oracle_exact_final(v0, ops, d)
    approx_final = _oracle_approx_final(v0, ops, d)
    discrepancy = abs(exact_final - approx_final)
    return discrepancy, d


def test_oracle_rederives_answer_text_every_tier():
    """Recomputes both pathways from the parsed question text alone, never
    importing the generator's own construction code, over 60 seeds per tier
    (180 questions total)."""
    n = 0
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(60):
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            discrepancy, d = _re_derive(question)
            rendered = Fraction(question.answer_text)
            assert abs(discrepancy - rendered) <= stated_precision_tolerance(d), (
                f"seed {seed} {difficulty}: oracle discrepancy {discrepancy} != "
                f"rendered answer {rendered} (question: {question.text!r})"
            )
            n += 1
    assert n == 180


def test_oracle_rejects_every_distractor():
    n = 0
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(60):
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            discrepancy, d = _re_derive(question)
            for distractor in question.distractors:
                assert abs(discrepancy - Fraction(distractor)) > stated_precision_tolerance(d), (
                    f"seed {seed} {difficulty}: distractor {distractor!r} matches the "
                    f"oracle's own recomputed discrepancy {discrepancy}"
                )
                n += 1
    assert n > 0
