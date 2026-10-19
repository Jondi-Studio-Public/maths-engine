"""Multi-part exam tier of `senior-use-scales-to-find-distances-such-as-on`: a route in two sections on a map,
its real length, the time to cover it, and its length on a map of a different scale.

The oracle parses the scale, both map lengths, the speed and the second scale out of the rendered text and works
the answers with exact fractions.
"""

from __future__ import annotations

import re
from fractions import Fraction

import exd_x_r3e_support as sup
from mathgen.engine import grade
from mathgen.core import round_half_up

SKILL = "senior-use-scales-to-find-distances-such-as-on"
TOPIC = "exd_x_r3e_map_route"


def oracle(q):
    S = int(re.search(r"1 : (\d+)", q.stem).group(1))
    d1, d2 = (Fraction(v) for v in re.findall(r"(\d+\.\d) cm", q.stem))
    v = int(re.search(r"averages (\d+) km/h", q.parts[2].prompt).group(1))
    S2 = int(re.search(r"scale of 1 : (\d+)", q.parts[3].prompt).group(1))
    metres = d1 * S / 100
    km = (d1 + d2) * S / 100000
    minutes = km / v * 60
    return S, d1, d2, v, S2, metres, km, round_half_up(minutes, 0), round_half_up(km * 100000 / S2, 1)


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abcd", ["number"] * 4, [1, 2, 2, 2])
    sup.assert_answer_space(TOPIC)


def test_walkers_and_cyclists_both_appear_with_plausible_distances():
    seen = set()
    for seed in range(120):
        q = sup.exam_question(TOPIC, seed)
        S, d1, d2, v, S2, metres, km, minutes, second = oracle(q)
        seen.add("cycling" if "cycling" in q.stem else "walking")
        assert 1 <= km <= (60 if "cycling" in q.stem else 18)
        assert (v >= 12) == ("cycling" in q.stem)
    assert seen == {"cycling", "walking"}


def test_key_matches_independent_derivation_from_the_rendered_text():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        S, d1, d2, v, S2, metres, km, minutes, second = oracle(q)
        a, b, c, d = (Fraction(p.answer_text) for p in q.parts)
        assert (a, b, c, d) == (metres, km, minutes, second), (seed, q.text, q.answer_text)


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC)


def test_a_wrong_first_section_carries_into_the_total_time_and_second_map():
    skill = sup.skill_object(SKILL)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        S, d1, d2, v, S2, metres, km, minutes, second = oracle(q)
        wa = metres + 100
        wb = (wa + d2 * S / 100) / 1000
        if wb * 100 != int(wb * 100):
            continue
        wc = round_half_up(wb / v * 60, 0)
        wd = round_half_up(wb * 100000 / S2, 1)
        replies = sup.reply(q, a=str(wa), b=f"{float(wb):.2f}", c=str(int(wc)), d=f"{float(wd):.1f}")
        got = grade(skill, q, replies)
        assert got.parts["a"].marks == 0
        assert (got.parts["b"].marks, got.parts["c"].marks, got.parts["d"].marks) == (2, 2, 2), (seed, replies, got)
        checked += 1
    assert checked >= 30


def test_a_wrong_total_does_not_carry_when_the_later_answer_is_the_original():
    skill = sup.skill_object(SKILL)
    for seed in range(40):
        q = sup.exam_question(TOPIC, seed)
        S, d1, d2, v, S2, metres, km, minutes, second = oracle(q)
        got = grade(skill, q, sup.reply(q, b=f"{float(km) + 1:.2f}"))
        assert got.parts["b"].marks == 0 and got.parts["c"].marks == 2
