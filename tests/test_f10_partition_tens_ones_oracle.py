"""Independent oracle for `f10-partition-one-two-digit-numbers`.

Reads only `question.text` and the public `set` grader. It reads the number (or the shown form, or the starting
number and the change) and any limit out of the wording, finds every partition by brute force over both parts, and
compares the result with the question's key. It does not import the generator module.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.grading.typed import grade_set

TOPIC = "f10_partition_tens_ones"
TIERS = ("easy", "medium", "hard")
N_SEEDS = 300


def _tens_ones(total):
    return {(t, o) for t in range(total + 1) for o in range(total + 1) if 10 * t + o == total}


def _expected(text):
    """The set of pairs the wording asks for."""
    m = re.match(r"Split (\d+) into two parts", text)
    if m:
        n = int(m.group(1))
        return {(a, b) for a in range(n + 1) for b in range(n + 1) if a + b == n and a >= b}
    m = re.match(r"Show (\d+) as tens and ones in every way where (.+?)\. Write", text)
    if m:
        n, cond = int(m.group(1)), m.group(2)
        pairs = _tens_ones(n)
        lim = re.fullmatch(r"the number of ones is less than (\d+)", cond)
        if lim:
            return {p for p in pairs if p[1] < int(lim.group(1))}
        low = re.fullmatch(r"there are at least (\d+) tens?", cond)
        assert low, cond
        return {p for p in pairs if p[0] >= int(low.group(1))}
    m = re.match(r"Show (\d+) as tens and ones in every way\.", text)
    if m:
        return _tens_ones(int(m.group(1)))
    m = re.match(r"A number is shown as (\d+) tens? and (\d+) ones\. List every other way", text)
    if m:
        t, o = int(m.group(1)), int(m.group(2))
        return _tens_ones(10 * t + o) - {(t, o)}
    m = re.match(r"A number is (5|10) (more|less) than (\d+) tens? and (\d+) ones?\. Show the number", text)
    assert m, text
    delta = int(m.group(1)) * (1 if m.group(2) == "more" else -1)
    return _tens_ones(10 * int(m.group(3)) + int(m.group(4)) + delta)


def _asked(text):
    tail = re.search(r"Write each way as \((.+?)\)", text).group(1)
    return tail


@pytest.mark.parametrize("tier", TIERS)
def test_the_key_is_every_partition_the_wording_allows(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        expected = _expected(q.text)
        got = {tuple(item) for item in q.answer}
        assert got == expected, (q.text, sorted(got), sorted(expected))
        assert len(q.answer) == len(expected)


@pytest.mark.parametrize("tier", TIERS)
def test_every_listed_partition_is_valid_for_its_wording(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        kind = _asked(q.text)
        for a, b in (tuple(i) for i in q.answer):
            if kind == "number of tens, number of ones":
                total = _total(q.text)
                assert 10 * a + b == total, (q.text, a, b)
            else:
                assert kind == "larger part, smaller part"
                assert a + b == int(re.match(r"Split (\d+)", q.text).group(1)) and a >= b >= 0


def _total(text):
    m = re.match(r"Show (\d+) as tens", text)
    if m:
        return int(m.group(1))
    m = re.match(r"A number is shown as (\d+) tens? and (\d+) ones", text)
    if m:
        return 10 * int(m.group(1)) + int(m.group(2))
    m = re.match(r"A number is (5|10) (more|less) than (\d+) tens? and (\d+) ones?", text)
    return 10 * int(m.group(3)) + int(m.group(4)) + int(m.group(1)) * (1 if m.group(2) == "more" else -1)


@pytest.mark.parametrize("tier", TIERS)
def test_the_grader_accepts_a_reply_built_from_the_oracle_and_rejects_one_off_replies(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        expected = sorted(_expected(q.text))
        reply = ", ".join(f"({a}, {b})" for a, b in reversed(expected))
        assert grade_set(q, "{" + reply + "}").correct, q.text
        short = ", ".join(f"({a}, {b})" for a, b in expected[1:])
        assert not grade_set(q, short).correct, q.text
        swapped = ", ".join(f"({b}, {a})" for a, b in expected)
        if {(b, a) for a, b in expected} != set(expected):
            assert not grade_set(q, swapped).correct, q.text
