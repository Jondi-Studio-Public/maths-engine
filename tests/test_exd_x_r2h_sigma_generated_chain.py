"""Tests for the multi-part exam source `exd_x_r2h_sigma_generated_chain` (skill `ucv1-sigma-algebra-closure`).

The oracle reads the universe, the family, the weights and the set expression out of the rendered text, finds the
atoms from the membership pattern of each point, and evaluates the set expression with a small parser.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks

TOPIC = "exd_x_r2h_sigma_generated_chain"
SKILL_ID = "ucv1-sigma-algebra-closure"
HARD_TOPIC = "p15c_sigma_algebra_closure"
LABELS = list("abcd")
MARKS = [2, 1, 1, 2]
FT_LABELS = ["b", "c"]


def _set(text):
    return frozenset(int(v) for v in text.split(","))


def parse(q):
    universe = _set(re.search(r"U = \{([\d, ]+)\}", q.text).group(1))
    family = {name: _set(body) for name, body in re.findall(r"\b([ABC]) = \{([\d, ]+)\}", q.text)}
    weights = {int(i): int(w) for i, w in re.findall(r"w_(\d+) = (\d+)", q.text)}
    expr = re.search(r"E = (.+?)(?: belongs|\.$)", q.parts[3].prompt).group(1)
    return universe, family, weights, expr


def evaluate(expr, family, universe):
    """Left-to-right evaluation of names, parentheses, a postfix ^c and the binary operators of the stems."""
    tokens = re.findall(r"\^c|[ABC()∪∩−]", expr)
    pos = 0

    def atom():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            value = formula()
            pos += 1
        else:
            value = family[tok]
        while pos < len(tokens) and tokens[pos] == "^c":
            value = universe - value
            pos += 1
        return value

    def formula():
        nonlocal pos
        value = atom()
        while pos < len(tokens) and tokens[pos] in "∪∩−":
            op = tokens[pos]
            pos += 1
            other = atom()
            value = value | other if op == "∪" else value & other if op == "∩" else value - other
        return value

    return formula()


def oracle(q):
    universe, family, weights, expr = parse(q)
    patterns = {}
    for point in universe:
        patterns.setdefault(tuple(point in s for s in family.values()), set()).add(point)
    atoms = len(patterns)
    e_set = evaluate(expr, family, universe)
    return {
        "a": str(atoms),
        "b": str(2**atoms),
        "c": str(2**atoms - len(family)),
        "d": str(sum(weights[i] for i in e_set)),
    }


def test_follow_through_from_wrong_atoms():
    def wrong_chain(q):
        key_b, key_c = int(q.parts[1].answer_text), int(q.parts[2].answer_text)
        n_family = key_b - key_c
        wrong = int(q.parts[0].distractors[0])
        replies = {
            "a": str(wrong),
            "b": str(2**wrong),
            "c": str(2**wrong - n_family),
            "d": q.parts[3].answer_text,
        }
        return "a", replies, 4

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_follow_through_from_wrong_count():
    def wrong_chain(q):
        key_b, key_c = int(q.parts[1].answer_text), int(q.parts[2].answer_text)
        n_family = key_b - key_c
        wrong = int(q.parts[1].distractors[0])
        replies = {
            "a": q.parts[0].answer_text,
            "b": str(wrong),
            "c": str(wrong - n_family),
            "d": q.parts[3].answer_text,
        }
        return "b", replies, 5

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        assert grade(skill, q, {"a": "1009", "b": "211", "c": "307", "d": "401"}).marks == 0


CHECKS = standard_checks(TOPIC, SKILL_ID, oracle, LABELS, MARKS, FT_LABELS, HARD_TOPIC)


def test_wired():
    CHECKS["wired"]()


def test_shape():
    CHECKS["shape"]()


def test_prompt_shapes():
    CHECKS["shapes"]()


def test_oracle_agrees_over_300_seeds():
    CHECKS["oracle_agrees"]()


def test_composed_answer_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()
