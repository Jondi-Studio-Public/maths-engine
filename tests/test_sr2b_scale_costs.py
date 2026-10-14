"""Tests the scale-drawing-costs skill: costs re-derived from the plan measurements in the stem."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from math import ceil

import pytest

from mathgen.engine import grade
from tests import sr2b_support as sup

SKILL = "scale-drawing-costs"
TOPIC = "sr2b_scale_costs"
EXAM = "sr2b_exam_scale_costs"


def _num(text: str) -> Fraction:
    return Fraction(text)


def _room(stem: str):
    scale = int(re.search(r"scale of 1 : (\d+)", stem).group(1))
    found = re.search(r"measures (\d+(?:\.\d+)?) cm by (\d+(?:\.\d+)?) cm", stem) or re.search(
        r"(\d+(?:\.\d+)?) cm long and (\d+(?:\.\d+)?) cm wide", stem
    )
    plan_l, plan_w = _num(found.group(1)), _num(found.group(2))
    return plan_l * scale / 100, plan_w * scale / 100


def _derive(stem: str) -> Fraction:
    length, width = _room(stem)
    area = length * width
    gate = re.search(r"gate opening (\d+(?:\.\d+)?) m wide", stem)
    if gate:
        fence, gate_cost = (
            int(v) for v in re.search(r"fencing costs \$(\d+) per metre and the gate costs \$(\d+)", stem).groups()
        )
        return (2 * (length + width) - _num(gate.group(1))) * fence + gate_cost
    waste = re.search(r"(\d+)% more", stem)
    factor = 1 + Fraction(int(waste.group(1)), 100) if waste else 1
    boxes = re.search(r"each cover (\d+(?:\.\d+)?) m² and cost \$(\d+(?:\.\d+)?) a box", stem)
    if boxes:
        return ceil(area * factor / _num(boxes.group(1))) * _num(boxes.group(2))
    price = int(re.search(r"\$(\d+) per square metre", stem).group(1))
    return area * factor * price


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "number",
        ["senior-obtain-a-scale-factor-and-use-it-to", "f10-direct-proportion-scale-model"],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_derivation(tier):
    for seed in range(150):
        q = sup.tier_question(TOPIC, tier, seed)
        assert _num(q.answer_text) == _derive(q.text), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=20, top=0.1)


def test_tiers_differ():
    asked = {t: " ".join(sup.tier_question(TOPIC, t, s).text for s in range(80)) for t in sup.TIERS}
    assert "waste" not in asked["easy"] and "boxes" not in asked["easy"]
    assert "waste" in asked["medium"] and "boxes" not in asked["medium"]
    assert "whole boxes" in asked["hard"] and "gate" in asked["hard"]
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_the_stem_reads_grammatically():
    for t in sup.TIERS:
        for s in range(80):
            text = sup.tier_question(TOPIC, t, s).text
            assert "tiles costs" not in text and " a office" not in text and " a hall costs" not in text


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number"] * 4, [1, 2, 1, 2], [False, True, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.05)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(120):
        q = sup.exam_question(EXAM, seed)
        length, width = _room(q.text)
        waste = int(re.search(r"(\d+)% more", q.text).group(1))
        cover, price = (_num(v) for v in re.search(r"each cover (\S+) m² and cost \$(\S+) a box", q.text).groups())
        need = length * width * (1 + Fraction(waste, 100))
        keys = [length, length * width, need, ceil(need / cover) * price]
        assert [_num(p.answer_text) for p in q.parts] == keys, (seed, q.text)


def test_exam_follow_through_from_a_wrong_length():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        width = _room(q.text)[1]
        waste = int(re.search(r"(\d+)% more", q.text).group(1))
        cover, price = (_num(v) for v in re.search(r"each cover (\S+) m² and cost \$(\S+) a box", q.text).groups())
        wrong_length = _num(canon["a"]) + 1
        area = wrong_length * width
        need = area * (1 + Fraction(waste, 100))
        cost = ceil(need / cover) * price

        def shown(v):
            exact = Decimal(v.numerator) / Decimal(v.denominator)
            return str(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        replies = {**canon, "a": shown(wrong_length), "b": shown(area), "c": shown(need), "d": str(int(cost))}
        result = grade(sk, q, replies)
        for label in "bcd":
            assert result.parts[label].marks == q.parts["abcd".index(label)].marks, (seed, label, replies)
