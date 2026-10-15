"""Tests for the multi-part exam source of ucv1-finite-automaton-acceptance: the end state of a run, the number of
accepted words of length 4 and 5 and an acceptance probability. Keys are re-derived by running every word."""

from __future__ import annotations

import re
from fractions import Fraction
from itertools import product

import mathgen.generators.exd_x_r2g_dfa_counting  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import follow_through_check, standard_checks

TOPIC = "exd_x_r2g_dfa_counting"
SKILL = "ucv1-finite-automaton-acceptance"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 2, 1]


def _text(v) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _facts(q):
    delta = {
        int(s): {"a": int(a), "b": int(b)}
        for s, a, b in re.findall(r"from q(\d), a goes to q(\d) and b goes to q(\d)", q.stem)
    }
    accept_text = re.search(r"(?:accepting|accepts in) (?:state|states) (q\d(?: and q\d)?)", q.stem).group(1)
    accepting = {int(x) for x in re.findall(r"q(\d)", accept_text)}
    word = re.search(r'word "([ab]+)"', q.text).group(1)
    return delta, accepting, word


def _run(delta, word):
    state = 0
    for ch in word:
        state = delta[state][ch]
    return state


def _accepted(delta, accepting, length):
    return sum(1 for w in product("ab", repeat=length) if _run(delta, "".join(w)) in accepting)


def oracle(q):
    delta, accepting, word = _facts(q)
    n5 = _accepted(delta, accepting, 5)
    return {
        "a": str(_run(delta, word)),
        "b": str(_accepted(delta, accepting, 4)),
        "c": str(n5),
        "d": _text(Fraction(n5, 32)),
    }


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"d"}, "p15d_automaton_acceptance")


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_keys_match_independent_derivation():
    CHECKS["oracle_agrees"]()


def test_composed_answer_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_rejected_and_tagged():
    CHECKS["distractors_rejected"]()


def test_follow_through_from_wrong_count():
    """A count of accepted words that is two out is carried into the probability."""

    def wrong_chain(q):
        delta, accepting, word = _facts(q)
        n5 = _accepted(delta, accepting, 5)
        wrong = n5 + 2
        replies = {
            "a": str(_run(delta, word)),
            "b": str(_accepted(delta, accepting, 4)),
            "c": str(wrong),
            "d": _text(Fraction(wrong, 32)),
        }
        return "c", replies, 1 + 2 + 1

    follow_through_check(TOPIC, SKILL, wrong_chain)
