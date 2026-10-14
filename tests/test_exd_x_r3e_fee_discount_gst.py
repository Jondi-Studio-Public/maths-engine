"""Multi-part exam tier of `f10-rational-percentage-financial-model`: a fee that is a fraction of the price, the
amount after a percentage discount, the GST on it, and the total.

The oracle reads the price, the fraction and the discount out of the rendered stem and works in whole cents.
"""

from __future__ import annotations

import re
from fractions import Fraction

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "f10-rational-percentage-financial-model"
TOPIC = "exd_x_r3e_fee_discount_gst"


def oracle(q):
    price = int(re.search(r"\$(\d+)", q.stem).group(1))
    frac = Fraction(re.search(r"equal to (\d+/\d+) of", q.stem).group(1))
    pct = int(re.search(r"discount of (\d+)%", q.stem).group(1))
    cents_fee = price * 100 * frac
    cents_discount = price * pct
    cents_subtotal = price * 100 + cents_fee - cents_discount
    cents_gst = cents_subtotal / 10
    assert (
        all(v.denominator == 1 for v in (cents_fee, cents_subtotal, cents_gst))
        if isinstance(cents_fee, Fraction)
        else True
    )
    return [Fraction(int(v), 100) for v in (cents_fee, cents_subtotal, cents_gst, cents_subtotal + cents_gst)]


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abcd", ["number"] * 4, [1, 2, 1, 1])
    sup.assert_answer_space(TOPIC)


def test_every_amount_is_written_to_two_decimal_places_and_is_plausible():
    for seed in range(120):
        q = sup.exam_question(TOPIC, seed)
        for p in q.parts:
            assert re.fullmatch(r"\d+\.\d\d", p.answer_text), (seed, p.answer_text)
            for d in p.distractors:
                assert re.fullmatch(r"\d+\.\d\d", d), (seed, p.label, d)
        price = int(re.search(r"\$(\d+)", q.stem).group(1))
        assert 400 <= price <= 2000
        assert price / 2 < Fraction(q.parts[3].answer_text) < 3 * price


def test_key_matches_independent_derivation_from_the_rendered_stem():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        assert [Fraction(p.answer_text) for p in q.parts] == oracle(q), (seed, q.stem, q.answer_text)


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC)


def test_a_wrong_fee_carries_into_the_amount_gst_and_total():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        fee, sub, gst, total = oracle(q)
        price = int(re.search(r"\$(\d+)", q.stem).group(1))
        pct = int(re.search(r"discount of (\d+)%", q.stem).group(1))
        wfee = fee + 40
        wsub = price + wfee - Fraction(price * pct, 100)
        wgst = wsub / 10
        replies = sup.reply(
            q, a=f"{float(wfee):.2f}", b=f"{float(wsub):.2f}", c=f"{float(wgst):.2f}", d=f"{float(wsub + wgst):.2f}"
        )
        got = grade(skill, q, replies)
        assert got.parts["a"].marks == 0
        assert (got.parts["b"].marks, got.parts["c"].marks, got.parts["d"].marks) == (2, 1, 1), (seed, replies, got)


def test_a_wrong_gst_carries_into_the_total_only():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        fee, sub, gst, total = oracle(q)
        wgst = gst + 12
        got = grade(skill, q, sup.reply(q, c=f"{float(wgst):.2f}", d=f"{float(sub + wgst):.2f}"))
        assert got.parts["c"].marks == 0
        assert got.parts["d"].marks == 1 and got.parts["d"].follow_through is True, (seed, got)
        assert grade(skill, q, sup.reply(q, d=f"{float(total) + 1:.2f}")).parts["d"].marks == 0
