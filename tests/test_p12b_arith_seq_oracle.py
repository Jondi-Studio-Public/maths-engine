"""Independent oracle for the arithmetic-sequence discrete-linear-model pair.

Parses ONLY `question.text`/`question.answer_text` and re-derives the
answer via the closed-form arithmetic-sequence formula, directly from the
numbers stated in the rendered text. Never imports
`mathgen.generators.p12b_arith_seq_build` -- see that module's docstring
for what it constructs; this file must not know it.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make

SEEDS = range(150)  # 8 (topic, tier) combinations x 150 seeds = 1200 questions

METHODS_TOPIC = "p12b_arith_seq_methods"
GENERAL_TOPIC = "p12b_arith_seq_general"


def _num(s: str) -> int:
    return int(s.replace("$", ""))


# --- Methods (ACMMM070): stated algebraic model -----------------------------

_METHODS_EASY_RE = re.compile(
    r"is (\$?\d+)(?: \S+)? in (\w+) 1\. Each \2 after that, it (increases|decreases) "
    r"by (\$?\d+)(?: \S+)?, .*\. Find .* in \2 (\d+)\.$"
)
_METHODS_MEDIUM_RE = re.compile(
    r"is (\$?\d+)(?: \S+)? in (\w+) 1, and it (increases|decreases) by (\$?\d+)(?: \S+)? "
    r"in every \2 after that, .*\. Find the total .* over the first (\d+) \2s\.$"
)
_METHODS_HARD_RE = re.compile(
    r"is (\$?\d+)(?: \S+)? in (\w+) 1\. Each \2 after that, it (increases|decreases) "
    r"by (\$?\d+)(?: \S+)?, .*\. In which \2 does .* first (?:reach|fall to) (\$?\d+)(?: \S+)?\?$"
)
_METHODS_EXAM_RE = re.compile(
    r"is (\$?\d+)(?: \S+)? in (\w+) (\d+) and (\$?\d+)(?: \S+)? in \2 (\d+)\. Assuming it "
    r"changes by a constant amount each \2, find the total .* over the first (\d+) \2s\.$"
)


def _oracle_methods_easy(text: str) -> int:
    m = _METHODS_EASY_RE.search(text)
    assert m, f"methods-easy oracle: no match for {text!r}"
    t1, _period, verb, d_mag, n = m.groups()
    t1, d_mag, n = _num(t1), _num(d_mag), int(n)
    d = d_mag if verb == "increases" else -d_mag
    return t1 + (n - 1) * d


def _oracle_methods_medium(text: str) -> int:
    m = _METHODS_MEDIUM_RE.search(text)
    assert m, f"methods-medium oracle: no match for {text!r}"
    a1, _period, verb, d_mag, n = m.groups()
    a1, d_mag, n = _num(a1), _num(d_mag), int(n)
    d = d_mag if verb == "increases" else -d_mag
    return n * (2 * a1 + (n - 1) * d) // 2


def _oracle_methods_hard(text: str) -> int:
    m = _METHODS_HARD_RE.search(text)
    assert m, f"methods-hard oracle: no match for {text!r}"
    t1, _period, verb, d_mag, target = m.groups()
    t1, d_mag, target = _num(t1), _num(d_mag), _num(target)
    d = d_mag if verb == "increases" else -d_mag
    quotient, remainder = divmod(target - t1, d)
    assert remainder == 0, f"methods-hard oracle: target not on the sequence for {text!r}"
    return quotient + 1


def _oracle_methods_exam(text: str) -> int:
    m = _METHODS_EXAM_RE.search(text)
    assert m, f"methods-exam oracle: no match for {text!r}"
    value_p, _period, p, value_q, q, n = m.groups()
    value_p, p, value_q, q, n = _num(value_p), int(p), _num(value_q), int(q), int(n)
    d, remainder = divmod(value_q - value_p, q - p)
    assert remainder == 0, f"methods-exam oracle: gap does not divide evenly for {text!r}"
    t1 = value_p - (p - 1) * d
    return n * (2 * t1 + (n - 1) * d) // 2


_METHODS_ORACLES = {
    "easy": _oracle_methods_easy,
    "medium": _oracle_methods_medium,
    "hard": _oracle_methods_hard,
    "exam": _oracle_methods_exam,
}


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_methods_oracle_agrees(tier):
    oracle_fn = _METHODS_ORACLES[tier]
    challenged = []
    for seed in SEEDS:
        q = make(METHODS_TOPIC, seed=seed, difficulty=tier)
        oracle_answer = oracle_fn(q.text)
        if oracle_answer != int(q.answer_text):
            challenged.append((seed, q.text, q.answer_text, oracle_answer))
    assert not challenged, f"{METHODS_TOPIC}/{tier}: oracle disagreed on {challenged}"


# --- General (ACMGM070): read off a practical table -------------------------

_GENERAL_PREDICT_RE = re.compile(
    r"is (\$?\d+)(?: \S+)? in (\w+) (\d+) and (\$?\d+)(?: \S+)? in \2 (\d+)\. Assuming it "
    r"changes by the same amount every \2, find .* in \2 (\d+)\.$"
)
_GENERAL_EXAM_RE = re.compile(
    r"is (\$?\d+)(?: \S+)? in (\w+) (\d+) and (\$?\d+)(?: \S+)? in \2 (\d+)\. Assuming it "
    r"changes by the same amount every \2, find the first \2 in which .* "
    r"(exceeds|falls below) (\$?\d+)(?: \S+)?\.$"
)


def _oracle_general_predict(text: str) -> int:
    m = _GENERAL_PREDICT_RE.search(text)
    assert m, f"general-predict oracle: no match for {text!r}"
    v1, _period, p1, v2, p2, p3 = m.groups()
    v1, p1, v2, p2, p3 = _num(v1), int(p1), _num(v2), int(p2), int(p3)
    d, remainder = divmod(v2 - v1, p2 - p1)
    assert remainder == 0, f"general-predict oracle: gap does not divide evenly for {text!r}"
    return v1 + (p3 - p1) * d


def _oracle_general_exam(text: str) -> int:
    m = _GENERAL_EXAM_RE.search(text)
    assert m, f"general-exam oracle: no match for {text!r}"
    v1, _period, p1, v2, p2, cross_word, target = m.groups()
    v1, p1, v2, p2, target = _num(v1), int(p1), _num(v2), int(p2), _num(target)
    d, remainder = divmod(v2 - v1, p2 - p1)
    assert remainder == 0, f"general-exam oracle: gap does not divide evenly for {text!r}"
    period = p1
    value = v1
    for _ in range(1000):
        if (cross_word == "exceeds" and value > target) or (cross_word == "falls below" and value < target):
            return period
        value += d
        period += 1
    raise AssertionError(f"general-exam oracle: never crossed the threshold for {text!r}")


_GENERAL_ORACLES = {
    "easy": _oracle_general_predict,
    "medium": _oracle_general_predict,
    "hard": _oracle_general_predict,
    "exam": _oracle_general_exam,
}


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_general_oracle_agrees(tier):
    oracle_fn = _GENERAL_ORACLES[tier]
    challenged = []
    for seed in SEEDS:
        q = make(GENERAL_TOPIC, seed=seed, difficulty=tier)
        oracle_answer = oracle_fn(q.text)
        if oracle_answer != int(q.answer_text):
            challenged.append((seed, q.text, q.answer_text, oracle_answer))
    assert not challenged, f"{GENERAL_TOPIC}/{tier}: oracle disagreed on {challenged}"
