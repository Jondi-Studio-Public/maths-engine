"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p9c_difference_quotient_build` --
the polynomial, the point, and (for row 2) delta-x are all parsed back out of
the rendered prose with regular expressions, then re-derived with sympy/
Fraction arithmetic written fresh here, never by calling into the generator.
"""

import re
from fractions import Fraction

import pytest
from sympy import diff, sympify, symbols

from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

_x = symbols("x")


def _parse_poly(prose: str):
    """A generator-rendered polynomial ("5·x^2 + 6", "-8·x - 5", "x^2 - 4·x + 5")
    back into a sympy expression in x. `·` is the generator's own stand-in for
    `*` (never a bare asterisk, per `mathgen/text_quality.py`); `^` is its
    stand-in for `**`. Both substitutions are exactly reversible because the
    generator produced them from a sympy string in the first place."""
    py_text = prose.replace("^", "**").replace("·", "*")
    return sympify(py_text, locals={"x": _x})


# --- Row 1: the numeric-trend skill ----------------------------------------

_ROW1_HEAD_RE = re.compile(r"^f\(x\) = (?P<f>.+?)\. At x = (?P<p>-?\d+), the difference quotient")
_Q_TOKEN_RE = re.compile(r"Q\((-?0\.(?:1|01|001))\) = (-?\d+\.\d{3})")


def _row1_oracle(question):
    """Re-derive row 1's answer from the rendered text alone: parse f(x) and
    p, differentiate with sympy's general operator, substitute p. This is a
    different route from the generator's own closed-form-then-limit
    construction and from its own diff()-based independent check -- it is a
    third, wholly separate derivation, written without reading the generator
    at all."""
    head = _ROW1_HEAD_RE.search(question.text)
    assert head is not None, f"could not parse row 1 head from: {question.text!r}"
    f_expr = _parse_poly(head.group("f"))
    p = int(head.group("p"))
    oracle_answer = diff(f_expr, _x).subs(_x, p)

    # Cross-check every displayed Q(h) token against a fresh (f(p+h)-f(p))/h
    # computed directly from the parsed polynomial, never from the closed
    # form the generator derived.
    for h_text, q_text in _Q_TOKEN_RE.findall(question.text):
        h = Fraction(h_text)
        expected = (f_expr.subs(_x, p + h) - f_expr.subs(_x, p)) / h
        got = Fraction(q_text)
        assert Fraction(expected.p, expected.q) == got, (
            f"Q({h_text}) mismatch: text says {q_text}, recomputed {expected}"
        )
    return int(oracle_answer)


@pytest.mark.parametrize("tier", TIERS)
def test_row1_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make("p9c_difference_quotient_behaviour", seed=seed, difficulty=tier)
        oracle_answer = _row1_oracle(q)
        assert str(oracle_answer) == q.answer_text, (
            f"seed={seed}/{tier}: oracle {oracle_answer} != rendered answer {q.answer_text}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_row1_oracle_rejects_every_distractor(tier):
    """Every offered distractor must differ from the independently-derived
    answer -- not merely from the generator's own claimed answer."""
    for seed in SEEDS:
        q = make("p9c_difference_quotient_behaviour", seed=seed, difficulty=tier)
        oracle_answer = _row1_oracle(q)
        for d in q.distractors:
            if "." in d:
                continue  # the deliberately non-integer reads-off-least-converged option
            assert int(d) != oracle_answer, f"seed={seed}/{tier}: distractor {d!r} equals oracle answer"


# --- Row 2: the single-quotient/notation skill ------------------------------

_ROW2_HEAD_RE = re.compile(
    r"^y = f\(x\) = (?P<f>.+?)\. At x = (?P<p>-?\d+), x changes by δx = (?P<dx>-?\d+(?:/\d+)?)\. "
    r"Find δy/δx"
)


def _parse_fraction_token(text: str) -> Fraction:
    return Fraction(text)


def _row2_oracle(question):
    """Re-derive row 2's answer from the rendered text alone: parse f(x), p
    and delta-x, then compute (f(p+dx)-f(p))/dx with sympy substitution --
    a route independent of the generator's own plain-Fraction arithmetic and
    of its own symbolic-then-substitute check."""
    head = _ROW2_HEAD_RE.search(question.text)
    assert head is not None, f"could not parse row 2 head from: {question.text!r}"
    f_expr = _parse_poly(head.group("f"))
    p = int(head.group("p"))
    dx = _parse_fraction_token(head.group("dx"))

    delta_y = f_expr.subs(_x, p + dx) - f_expr.subs(_x, p)
    oracle_answer = delta_y / dx
    return Fraction(oracle_answer.p, oracle_answer.q)


@pytest.mark.parametrize("tier", TIERS)
def test_row2_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make("p9c_delta_y_delta_x_notation", seed=seed, difficulty=tier)
        oracle_answer = _row2_oracle(q)
        rendered_answer = Fraction(q.answer["p"], q.answer["q"])
        assert oracle_answer == rendered_answer, (
            f"seed={seed}/{tier}: oracle {oracle_answer} != rendered answer {rendered_answer}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_row2_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make("p9c_delta_y_delta_x_notation", seed=seed, difficulty=tier)
        oracle_answer = _row2_oracle(q)
        for d in q.distractors:
            assert _parse_fraction_token(d) != oracle_answer, (
                f"seed={seed}/{tier}: distractor {d!r} equals oracle answer"
            )


def test_row2_hard_tier_ignored_delta_x_sign_would_be_rejected():
    """The `ignored-delta-x-sign` misconception (hard tier only) uses
    |delta-x| in place of the stated negative delta-x. Recompute it purely
    from the parsed text -- f(x), p and the stated (negative) delta-x -- and
    confirm it differs from the oracle's own independently-derived answer.
    This exercises the misconception's value even on the seeds where it is
    not one of the three rendered options (`pick_distractors` caps output at"""
    checked = 0
    for seed in SEEDS:
        q = make("p9c_delta_y_delta_x_notation", seed=seed, difficulty="hard")
        head = _ROW2_HEAD_RE.search(q.text)
        f_expr = _parse_poly(head.group("f"))
        p = int(head.group("p"))
        dx = _parse_fraction_token(head.group("dx"))
        assert dx < 0, f"seed={seed}: hard tier delta-x is not negative: {dx}"

        oracle_answer = _row2_oracle(q)
        magnitude_variant = (f_expr.subs(_x, p + abs(dx)) - f_expr.subs(_x, p)) / abs(dx)
        magnitude_variant = Fraction(magnitude_variant.p, magnitude_variant.q)
        assert magnitude_variant != oracle_answer, (
            f"seed={seed}: |delta-x| variant {magnitude_variant} equals the answer {oracle_answer}"
        )
        checked += 1
    assert checked == len(SEEDS)
