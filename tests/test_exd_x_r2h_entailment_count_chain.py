"""Tests for the multi-part exam source `exd_x_r2h_entailment_count_chain` (skill `ucv1-syntactic-semantic-separation`).

The oracle parses the two premises and the conclusion with its own recursive-descent parser for the connectives
(negation, conjunction, disjunction, implication; binary formulas fully parenthesised) and counts the satisfying
interpretations by enumeration.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks
from itertools import product

TOPIC = "exd_x_r2h_entailment_count_chain"
SKILL_ID = "ucv1-syntactic-semantic-separation"
HARD_TOPIC = "p15b_syntactic_semantic_separation"
LABELS = list("abcd")
MARKS = [2, 2, 1, 1]
FT_LABELS = ["c", "d"]


def parse_formula(text):
    tokens = re.findall(r"[¬∧∨→()]|[a-z]", text)
    pos = 0

    def formula():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "¬":
            inner = formula()
            return lambda env: not inner(env)
        if tok == "(":
            left = formula()
            op = tokens[pos]
            pos += 1
            right = formula()
            pos += 1
            return {
                "∧": lambda env: left(env) and right(env),
                "∨": lambda env: left(env) or right(env),
                "→": lambda env: (not left(env)) or right(env),
            }[op]
        return lambda env: env[tok]

    result = formula()
    assert pos == len(tokens), text
    return result


_STEMS = (
    r"Premise 1: (.+?)\. Premise 2: (.+?)\. A student concludes (.+?) from these two premises\. "
    r"Consider all (\d+) interpretations that assign True or False to each of ([a-z, ]+)\.",
    r"The premises are (.+?) and (.+?), and the student's conclusion is (.+?)\. There are (\d+) interpretations, "
    r"assigning True or False to each of ([a-z, ]+)\.",
    r"From the premises (.+?) and (.+?), a student draws the conclusion (.+?)\. Look at all (\d+) "
    r"interpretations of ([a-z, ]+), each",
)


def parse(q):
    m = next(m for m in (re.search(pattern, q.text) for pattern in _STEMS) if m)
    p1, p2, concl = (parse_formula(m.group(i)) for i in (1, 2, 3))
    atoms = re.findall(r"[a-z]", m.group(5))
    assert 2 ** len(atoms) == int(m.group(4))
    return p1, p2, concl, atoms


def oracle(q):
    p1, p2, concl, atoms = parse(q)
    both = both_and_concl = 0
    for values in product((False, True), repeat=len(atoms)):
        env = dict(zip(atoms, values))
        if p1(env) and p2(env):
            both += 1
            both_and_concl += bool(concl(env))
    return {
        "a": str(both),
        "b": str(both_and_concl),
        "c": str(both - both_and_concl),
        "d": str(Fraction(both_and_concl, both)),
    }


def test_follow_through_from_wrong_premise_count():
    def wrong_chain(q):
        key_b = int(q.parts[1].answer_text)
        wrong = int(q.parts[0].distractors[0])
        replies = {
            "a": str(wrong),
            "b": q.parts[1].answer_text,
            "c": str(wrong - key_b),
            "d": str(Fraction(key_b, wrong)),
        }
        return "a", replies, 4

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_follow_through_from_wrong_joint_count():
    def wrong_chain(q):
        key_a = int(q.parts[0].answer_text)
        wrong = int(q.parts[1].distractors[0])
        replies = {
            "a": q.parts[0].answer_text,
            "b": str(wrong),
            "c": str(key_a - wrong),
            "d": str(Fraction(wrong, key_a)),
        }
        return "b", replies, 4

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        assert grade(skill, q, {"a": "1009", "b": "211", "c": "307", "d": "401/1013"}).marks == 0


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
