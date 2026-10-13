"""Tests for the multi-part exam source `exd_x_r2h_central_force_chain` (skill `ucv1-symmetry-conservation`).

The oracle reads the mass, the potential strength, the initial state and the later radius out of the rendered text and
recomputes the conserved angular momentum and energy exactly.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks

TOPIC = "exd_x_r2h_central_force_chain"
SKILL_ID = "ucv1-symmetry-conservation"
HARD_TOPIC = "p15c_symmetry_conservation"
LABELS = list("abcd")
MARKS = [2, 2, 1, 2]
FT_LABELS = ["c", "d"]


def parse(q):
    m = int(re.search(r"mass (\d+) kg", q.text).group(1))
    c = int(re.search(r"V\(x, y\) = (\d+)\(x\^2", q.text).group(1))
    (x, y), (vx, vy) = [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", q.text)]
    r1 = int(re.search(r"(\d+) m from the origin", q.parts[2].prompt).group(1))
    return m, c, x, y, vx, vy, r1


def oracle(q):
    m, c, x, y, vx, vy, r1 = parse(q)
    lz = m * (x * vy - y * vx)
    energy = Fraction(m, 2) * (vx**2 + vy**2) + c * (x**2 + y**2)
    v_perp = Fraction(lz, m * r1)
    radial = energy - c * r1**2 - Fraction(m, 2) * v_perp**2
    return {"a": str(lz), "b": str(energy), "c": str(v_perp), "d": str(radial)}


def test_follow_through_from_wrong_angular_momentum():
    def wrong_chain(q):
        m, c, *_, r1 = parse(q)
        wrong = Fraction(q.parts[0].distractors[0])
        v_perp = wrong / (m * r1)
        energy = Fraction(q.parts[1].answer_text)
        replies = {
            "a": str(wrong),
            "b": q.parts[1].answer_text,
            "c": str(v_perp),
            "d": str(energy - c * r1**2 - Fraction(m, 2) * v_perp**2),
        }
        return "a", replies, 5

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_follow_through_from_wrong_energy():
    def wrong_chain(q):
        m, c, *_, r1 = parse(q)
        wrong = Fraction(q.parts[1].distractors[0])
        v_perp = Fraction(q.parts[2].answer_text)
        replies = {
            "a": q.parts[0].answer_text,
            "b": str(wrong),
            "c": q.parts[2].answer_text,
            "d": str(wrong - c * r1**2 - Fraction(m, 2) * v_perp**2),
        }
        return "b", replies, 5

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        assert grade(skill, q, {"a": "1009", "b": "211/1013", "c": "-307/1019", "d": "401/1031"}).marks == 0


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
