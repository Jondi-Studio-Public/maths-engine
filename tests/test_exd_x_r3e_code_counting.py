"""Multi-part exam tier of `multiplication-principle`: all codes, codes with no repeat, codes with a repeat, and
no-repeat codes with a restricted first symbol.

The oracle reads N, k and the restriction out of the rendered text and counts with exact integer arithmetic.
"""

from __future__ import annotations

import re

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "multiplication-principle"
TOPIC = "exd_x_r3e_code_counting"


def oracle(q):
    s = q.stem
    if "numbered 1 to" in s:
        N = int(re.search(r"numbered 1 to (\d+)", s).group(1))
        k = int(re.search(r"sequence of (\d+) numbers", s).group(1))
        step = int(re.search(r"multiple of (\d+)", q.parts[3].prompt).group(1))
        m = len([v for v in range(1, N + 1) if v % step == 0])
    elif "digits" in s:
        N, k = 10, int(re.search(r"sequence of (\d+) digits", s).group(1))
        m = len([v for v in range(10) if v % 2 == 1])
    else:
        N, k = 26, int(re.search(r"sequence of (\d+) letters", s).group(1))
        m = len([c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c in "AEIOU"])
    total = N**k
    distinct = 1
    for i in range(k):
        distinct *= N - i
    rest = 1
    for i in range(1, k):
        rest *= N - i
    return total, distinct, total - distinct, m * rest


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abcd", ["number"] * 4, [1, 2, 1, 2])
    sup.assert_answer_space(TOPIC)


def test_all_three_settings_appear():
    seen = set()
    for seed in range(150):
        s = sup.exam_question(TOPIC, seed).stem
        seen.add("dial" if "dial" in s else "digits" if "digits" in s else "letters")
    assert seen == {"dial", "digits", "letters"}


def test_key_matches_brute_force_style_recount_from_the_rendered_text():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        assert [int(p.answer_text) for p in q.parts] == list(oracle(q)), (seed, q.text, q.answer_text)


def test_small_cases_agree_with_enumeration():
    """The counting formulas, checked by listing every code, on dials small enough to enumerate."""
    from itertools import product

    for N in (5, 6):
        for k in (3, 4):
            codes = list(product(range(1, N + 1), repeat=k))
            distinct = [c for c in codes if len(set(c)) == k]
            first_ok = [c for c in distinct if c[0] % 2 == 0]
            m = len([v for v in range(1, N + 1) if v % 2 == 0])
            rest = 1
            for i in range(1, k):
                rest *= N - i
            assert len(codes) == N**k and len(first_ok) == m * rest
            assert len(codes) - len(distinct) == N**k - len(distinct)


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC)


def test_a_wrong_total_or_count_carries_into_the_repeat_count():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        total, distinct, repeat, restricted = oracle(q)
        wa, wb = total + 11, distinct - 7
        got = grade(skill, q, sup.reply(q, a=str(wa), b=str(wb), c=str(wa - wb)))
        assert got.parts["a"].marks == 0 and got.parts["b"].marks == 0
        assert got.parts["c"].marks == 1 and got.parts["c"].follow_through is True, (seed, got)
        assert grade(skill, q, sup.reply(q, c=str(repeat + 1))).parts["c"].marks == 0
        assert grade(skill, q, sup.reply(q, d=str(restricted + 1))).parts["d"].marks == 0
