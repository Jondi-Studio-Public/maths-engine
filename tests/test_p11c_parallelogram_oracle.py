"""Independent oracle for the `p11c_parallelogram` questions: it re-derives each answer from the rendered
question text.

Parses ONLY the rendered question (``question.text``, ``question.answer_text``, ``question.distractors``) and
re-derives, from scratch, both things a learner is actually asked to trust: the numeric answer, and whether
the implied figure is a plausible parallelogram. Imports nothing from
``mathgen.generators.p11c_parallelogram_build`` -- not the quadruple table, not the plausibility gate, not a
formatting helper.

Two independent checks:

1. **Value oracle**: parse the three stated lengths (converting any
   metres/millimetres to a common base of centimetres using ONLY the parsed
   numbers), independently apply the parallelogram diagonal law
   ``d1^2+d2^2=2(a^2+b^2)`` to derive the missing length, and assert it
   equals the rendered ``answer_text`` (converted back to whatever unit the
   question actually asked for).
2. **Plausibility oracle**: independently re-derive the angle implied by the
   parsed side/diagonal numbers from the law of cosines, and the
   diagonal/side ratio, and assert both sit inside a freshly-authored bound
   -- mirroring ``tests/test_p10b_estimate_lengths_oracle.py``'s pattern (an
   independently-authored plausibility check, never imported from the
   generator's own table). This is the check that would catch an
   arithmetically correct but implausible figure.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make

TOPIC = "p11c_parallelogram_diagonal_law"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

# Easy and medium each gained 2 more sentence templates (identical question type, different
# wording -- see the generator module's own `_EASY_TEMPLATES`/`_MEDIUM_TEMPLATES` comment), and the
# hard tier gained a third unit-conversion variant (the GIVEN diagonal, not a side or the answer,
# stated in millimetres). Every template's capture-group ORDER is unchanged (side1, side2, given for
# easy; diag1, diag2, known for medium), so one tuple of regexes per tier is tried in order and the
# first match wins.
_EASY_RES = (
    re.compile(
        r"^A parallelogram has sides of length (\d+) cm and (\d+) cm\. "
        r"One of its diagonals is (\d+) cm long\. "
        r"What is the length of the other diagonal, in cm\?$"
    ),
    re.compile(
        r"^In a parallelogram, one side is (\d+) cm and the adjacent side is (\d+) cm\. "
        r"One diagonal is (\d+) cm long\. "
        r"Find the length of the other diagonal, in cm\.$"
    ),
    re.compile(
        r"^A parallelogram's two sides measure (\d+) cm and (\d+) cm\. "
        r"Given that one diagonal is (\d+) cm, calculate the length of the other diagonal, "
        r"in cm\.$"
    ),
)
_MEDIUM_RES = (
    re.compile(
        r"^A parallelogram has diagonals of length (\d+) cm and (\d+) cm\. "
        r"One of its sides is (\d+) cm long\. "
        r"What is the length of the other side, in cm\?$"
    ),
    re.compile(
        r"^In a parallelogram, the diagonals measure (\d+) cm and (\d+) cm\. "
        r"One side is (\d+) cm long\. "
        r"Find the length of the other side, in cm\.$"
    ),
    re.compile(
        r"^A parallelogram has diagonal lengths (\d+) cm and (\d+) cm\. "
        r"Given that one side is (\d+) cm, calculate the length of the other side, in cm\.$"
    ),
)
_HARD_METRES_RE = re.compile(
    r"^A parallelogram has one side (\d+(?:\.\d+)?) m long and the adjacent side (\d+) cm long\. "
    r"One of its diagonals is (\d+) cm long\. "
    r"What is the length of the other diagonal, in cm\?$"
)
_HARD_MM_RE = re.compile(
    r"^A parallelogram has sides of length (\d+) cm and (\d+) cm\. "
    r"One of its diagonals is (\d+) cm long\. "
    r"What is the length of the other diagonal, in millimetres\?$"
)
_HARD_GIVEN_MM_RE = re.compile(
    r"^A parallelogram has sides of length (\d+) cm and (\d+) cm\. "
    r"One of its diagonals is (\d+) mm long\. "
    r"What is the length of the other diagonal, in cm\?$"
)


def _match_any(text, regexes):
    for rx in regexes:
        m = rx.match(text)
        if m:
            return m
    return None


# Fresh plausibility oracle -- re-implemented from the law of cosines, NEVER
# imported from the generator's own `is_plausible` or `_TABLE`.

_ANGLE_LO_DEG = 20.0
_ANGLE_HI_DEG = 160.0
_RATIO_THRESH = 0.5


def _oracle_angle_deg(a: int, b: int, given_diagonal: int) -> float:
    cos_theta = Fraction(given_diagonal * given_diagonal - a * a - b * b, 2 * a * b)
    cos_f = max(-1.0, min(1.0, float(cos_theta)))
    return math.degrees(math.acos(cos_f))


def _assert_plausible(a: int, b: int, d1: int, d2: int, where: str) -> None:
    theta = _oracle_angle_deg(a, b, d1)
    assert _ANGLE_LO_DEG <= theta <= _ANGLE_HI_DEG, (
        f"{where}: implied angle {theta:.2f} deg outside the independently-checked "
        f"[{_ANGLE_LO_DEG}, {_ANGLE_HI_DEG}] plausibility band for sides {a},{b} diag {d1},{d2}"
    )
    assert min(d1, d2) > _RATIO_THRESH * max(a, b), (
        f"{where}: diagonal/side ratio too thin for sides {a},{b} diag {d1},{d2}"
    )


def _other_diagonal(a: int, b: int, given: int) -> int:
    """Independent value oracle for the diagonal-unknown shape: solve
    ``2(a^2+b^2) - given^2`` and require an exact integer square root (the
    skill's own contract: never a surd needing rounding)."""
    disc = 2 * (a * a + b * b) - given * given
    root = math.isqrt(disc) if disc >= 0 else -1
    assert disc >= 0 and root * root == disc, f"no exact integer solution for a={a} b={b} given={given}"
    return root


def _other_side(d1: int, d2: int, known: int) -> int:
    """Independent value oracle for the side-unknown shape."""
    disc = (d1 * d1 + d2 * d2) // 2 - known * known
    assert (d1 * d1 + d2 * d2) % 2 == 0, f"d1^2+d2^2 not even for d1={d1} d2={d2}"
    root = math.isqrt(disc) if disc >= 0 else -1
    assert disc >= 0 and root * root == disc, f"no exact integer solution for d1={d1} d2={d2} known={known}"
    return root


def test_easy_tier_value_and_plausibility_oracle():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy", tier="easy")
        m = _match_any(q.text, _EASY_RES)
        assert m, f"unparseable easy question: {q.text!r}"
        side1, side2, given = (int(x) for x in m.groups())
        expected_other = _other_diagonal(side1, side2, given)
        assert str(expected_other) == q.answer_text, (
            f"easy seed={seed}: oracle expected {expected_other}, question answered {q.answer_text!r}"
        )
        _assert_plausible(side1, side2, given, expected_other, f"easy seed={seed}")
        for distractor in q.distractors:
            assert distractor != q.answer_text
            assert int(distractor) != expected_other
        checked += 1
    assert checked == len(SEEDS)


def test_medium_tier_value_and_plausibility_oracle():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium", tier="medium")
        m = _match_any(q.text, _MEDIUM_RES)
        assert m, f"unparseable medium question: {q.text!r}"
        diag1, diag2, known = (int(x) for x in m.groups())
        expected_other = _other_side(diag1, diag2, known)
        assert str(expected_other) == q.answer_text, (
            f"medium seed={seed}: oracle expected {expected_other}, question answered {q.answer_text!r}"
        )
        _assert_plausible(known, expected_other, diag1, diag2, f"medium seed={seed}")
        for distractor in q.distractors:
            assert int(distractor) != expected_other
        checked += 1
    assert checked == len(SEEDS)


def test_hard_tier_value_and_plausibility_oracle_all_variants():
    """Parses whichever of the three hard-tier variants was drawn, independently converts by a power of ten, and
    applies the same value oracle as the easy tier."""
    checked = 0
    seen_metres_variant = False
    seen_answer_mm_variant = False
    seen_given_mm_variant = False
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard", tier="hard")
        m_metres = _HARD_METRES_RE.match(q.text)
        m_mm = _HARD_MM_RE.match(q.text)
        m_given_mm = _HARD_GIVEN_MM_RE.match(q.text)
        assert m_metres or m_mm or m_given_mm, f"unparseable hard question: {q.text!r}"

        if m_metres:
            seen_metres_variant = True
            side1_m_str, side2_str, given_str = m_metres.groups()
            side1_cm = int(Fraction(side1_m_str) * 100)  # independent m->cm conversion
            assert Fraction(side1_m_str) * 100 == side1_cm, "metre-to-cm conversion did not land on an exact integer"
            side2, given = int(side2_str), int(given_str)
            expected_other_cm = _other_diagonal(side1_cm, side2, given)
            assert str(expected_other_cm) == q.answer_text, (
                f"hard(metres) seed={seed}: oracle expected {expected_other_cm}, answered {q.answer_text!r}"
            )
            _assert_plausible(side1_cm, side2, given, expected_other_cm, f"hard(metres) seed={seed}")
        elif m_mm:
            seen_answer_mm_variant = True
            side1, side2, given = (int(x) for x in m_mm.groups())
            expected_other_cm = _other_diagonal(side1, side2, given)
            expected_other_mm = expected_other_cm * 10  # independent cm->mm conversion
            assert str(expected_other_mm) == q.answer_text, (
                f"hard(mm) seed={seed}: oracle expected {expected_other_mm}, answered {q.answer_text!r}"
            )
            _assert_plausible(side1, side2, given, expected_other_cm, f"hard(mm) seed={seed}")
        else:
            seen_given_mm_variant = True
            side1, side2, given_mm_str = m_given_mm.groups()
            side1, side2 = int(side1), int(side2)
            given_mm = int(given_mm_str)
            assert given_mm % 10 == 0, "an exact mm-to-cm conversion must land on a multiple of 10"
            given_cm = given_mm // 10  # independent mm->cm conversion
            expected_other_cm = _other_diagonal(side1, side2, given_cm)
            assert str(expected_other_cm) == q.answer_text, (
                f"hard(given mm) seed={seed}: oracle expected {expected_other_cm}, answered {q.answer_text!r}"
            )
            _assert_plausible(side1, side2, given_cm, expected_other_cm, f"hard(given mm) seed={seed}")
        checked += 1
    assert checked == len(SEEDS)
    assert seen_metres_variant, "hard tier never drew the side-in-metres variant over 60 seeds"
    assert seen_answer_mm_variant, "hard tier never drew the answer-in-millimetres variant over 60 seeds"
    assert seen_given_mm_variant, "hard tier never drew the given-diagonal-in-millimetres variant over 60 seeds"


def test_oracle_regexes_cover_every_rendered_question_every_tier():
    """Sanity on the oracle's own coverage: every question over a 60-seed
    sweep of every tier must match exactly one of this file's own regexes --
    an oracle that silently skipped unparseable questions would prove
    nothing."""
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            all_regexes = _EASY_RES + _MEDIUM_RES + (_HARD_METRES_RE, _HARD_MM_RE, _HARD_GIVEN_MM_RE)
            matched = sum(bool(rx.match(q.text)) for rx in all_regexes)
            assert matched == 1, f"{tier} seed={seed}: question matched {matched} oracle patterns: {q.text!r}"
