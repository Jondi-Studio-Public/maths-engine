"""Tests `uc2-integral-theorem-verification`: verifying an integral theorem. Single-answer tiers and the four-part exam chain."""

from __future__ import annotations

from tests import s2a3_support as sup

SKILL = "uc2-integral-theorem-verification"


def test_single_tiers_grade_and_distractors():
    sup.check_single_tiers(SKILL)


def test_tiers_differ():
    sup.check_tiers_differ(SKILL)


def test_deterministic():
    sup.check_deterministic(SKILL)


def test_exam_shape():
    sup.check_exam_shape(SKILL, "abcd", "d")


def test_exam_canonical_answers_full_marks():
    sup.check_exam_canonical(SKILL)


def test_exam_distractors_rejected():
    sup.check_exam_distractors(SKILL)


def test_exam_follow_through():
    sup.check_exam_follow_through(SKILL)
