"""Independent oracle for the Bernoulli-model / Bernoulli-sequence pair.

Parses ONLY `question.text`/`question.answer_text` and re-derives the
answer from scratch via a route that never touches
`mathgen.generators.p12d_bernoulli_build` -- not imported, not read. See
that module's docstring for what it constructs; this file must not know it.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(150)  # 8 (topic, tier) combinations x 150 seeds = 1200 questions

MODEL_TOPIC = "p12d_bernoulli_model"
SEQ_TOPIC = "p12d_bernoulli_sequence"


def _frac_text(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


# --- p12d_bernoulli_model (ACMMM143) ----------------------------------------

_MODEL_D_RE = re.compile(
    r"(\d+) (?:equal sectors|identical-looking raffle tickets|members|"
    r"marbles|cards is numbered 1 to \d+|parked cars)"
)
_MODEL_K_RE = re.compile(r"(\d+) of (?:which|whom|the cards)")
_MODEL_VAR_RE = re.compile(r"Var\(X\) = (\d+)/(\d+)")
_MODEL_LESS_MORE_RE = re.compile(r"\b(less|more)\b (?:common|often|frequent)")
_MODEL_FEWER_MORE_RE = re.compile(r"\b(more|fewer)\b than half")


def _oracle_model_easy(text: str) -> Fraction:
    d = int(_MODEL_D_RE.search(text).group(1))
    k = int(_MODEL_K_RE.search(text).group(1))
    return Fraction(k, d)


def _oracle_model_medium(text: str) -> Fraction:
    d = int(_MODEL_D_RE.search(text).group(1))
    k = int(_MODEL_K_RE.search(text).group(1))
    p = Fraction(k, d)
    return p * (1 - p)


def _quadratic_roots(v: Fraction) -> tuple[Fraction, Fraction]:
    """Roots of p^2 - p + v = 0 via the quadratic formula, kept exact: the
    discriminant 1 - 4v is a perfect square of a rational for every
    constructed question (by construction the two roots are p_true and
    1 - p_true, both rational), so an integer square root of the scaled
    numerator/denominator recovers it exactly without any float step."""
    disc = 1 - 4 * v
    num, den = disc.numerator, disc.denominator
    # disc = num/den; need sqrt(disc) = sqrt(num*den)/den (den made a perfect
    # square by multiplying through) -- simplest: scale to a common square
    # denominator by multiplying num and den by den, then take integer sqrt.
    scaled_num = num * den
    root = math.isqrt(scaled_num)
    assert root * root == scaled_num, f"discriminant {disc} is not a perfect square"
    sqrt_disc = Fraction(root, den)
    lower = (1 - sqrt_disc) / 2
    upper = (1 + sqrt_disc) / 2
    return lower, upper


def _oracle_model_hard(text: str) -> Fraction:
    vn, vd = _MODEL_VAR_RE.search(text).groups()
    v = Fraction(int(vn), int(vd))
    lower, upper = _quadratic_roots(v)
    comp = _MODEL_LESS_MORE_RE.search(text).group(1)
    return lower if comp == "less" else upper


def _oracle_model_exam(text: str) -> Fraction:
    vn, vd = _MODEL_VAR_RE.search(text).groups()
    v = Fraction(int(vn), int(vd))
    lower, upper = _quadratic_roots(v)
    freq = _MODEL_FEWER_MORE_RE.search(text).group(1)
    # "a failure on {freq} than half of all trials" -- freq == "more" means
    # P(failure) is the LARGER root; "fewer" means P(failure) is the smaller.
    # The question asks for P(success), the COMPLEMENT of P(failure).
    p_failure = upper if freq == "more" else lower
    p_success = lower if p_failure == upper else upper
    return p_success


_MODEL_ORACLES = {
    "easy": _oracle_model_easy,
    "medium": _oracle_model_medium,
    "hard": _oracle_model_hard,
    "exam": _oracle_model_exam,
}


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_model_oracle_agrees(tier):
    oracle_fn = _MODEL_ORACLES[tier]
    challenged = []
    for seed in SEEDS:
        q = make(MODEL_TOPIC, seed=seed, difficulty=tier)
        oracle_answer = oracle_fn(q.text)
        if oracle_answer != Fraction(q.answer_text):
            challenged.append((seed, q.text, q.answer_text, str(oracle_answer)))
    assert not challenged, f"{MODEL_TOPIC}/{tier}: oracle disagreed on {challenged}"


# --- p12d_bernoulli_sequence (ACMMM146) -------------------------------------

_SEQ_LABELS = {
    "win": "loss",
    "loss": "win",
    "heads": "tails",
    "tails": "heads",
    "faulty": "ok",
    "ok": "faulty",
}
_PROB_TOKEN = r"(\d+/\d+|0\.\d)"
_SEQ_STATED_PROB_RE = re.compile(
    r"lands on '(\w+)' with probability " + _PROB_TOKEN + r"|"
    r"lands (heads|tails) with probability " + _PROB_TOKEN + r"|"
    r"independently (faulty) with probability " + _PROB_TOKEN + r"|"
    r"independently free of faults \(ok\) with probability " + _PROB_TOKEN
)
_SEQ_SEQUENCE_RE = re.compile(r"sequence of outcomes, in order, is: (.+)\.$")


def _parse_prob(raw: str) -> Fraction:
    return Fraction(raw) if "/" in raw else Fraction(raw)


def _oracle_seq(text: str) -> Fraction:
    m = _SEQ_STATED_PROB_RE.search(text)
    assert m, f"sequence oracle: no stated probability found in {text!r}"
    groups = m.groups()
    if groups[0] is not None:
        stated_label, raw = groups[0], groups[1]
    elif groups[2] is not None:
        stated_label, raw = groups[2], groups[3]
    elif groups[4] is not None:
        stated_label, raw = groups[4], groups[5]
    else:
        stated_label, raw = "ok", groups[6]
    stated_p = _parse_prob(raw)

    seq_m = _SEQ_SEQUENCE_RE.search(text)
    assert seq_m, f"sequence oracle: no sequence found in {text!r}"
    outcomes = [o.strip() for o in seq_m.group(1).split(" then ")]

    product = Fraction(1)
    for outcome in outcomes:
        if outcome == stated_label:
            product *= stated_p
        else:
            assert outcome == _SEQ_LABELS[stated_label], (
                f"sequence oracle: outcome {outcome!r} does not pair with stated label {stated_label!r} in {text!r}"
            )
            product *= 1 - stated_p
    return product


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_seq_oracle_agrees(tier):
    challenged = []
    for seed in SEEDS:
        q = make(SEQ_TOPIC, seed=seed, difficulty=tier)
        oracle_answer = _oracle_seq(q.text)
        if oracle_answer != Fraction(q.answer_text):
            challenged.append((seed, q.text, q.answer_text, str(oracle_answer)))
    assert not challenged, f"{SEQ_TOPIC}/{tier}: oracle disagreed on {challenged}"
