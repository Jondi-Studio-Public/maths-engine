"""Tests `uc2-homomorphism-kernel-image`: kernels, images and preimages are listed element by element from the maps read
back out of the question TEXT, the typed grader on every key and distractor, and the four-part exam chain."""

from __future__ import annotations

import re
from math import gcd

import mathgen.generators.r4a_homomorphism_kernel_image_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4a_support as sup

TOPIC = "r4a_homomorphism_kernel_image"
SKILL_ID = "uc2-homomorphism-kernel-image"


def _is_homomorphism_from_cyclic(m, n, k):
    return all(((x + y) % m) * k % n == (k * x + k * y) % n for x in range(m) for y in range(m))


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "number")


def test_easy_key_is_the_kernel_order():
    for seed in range(100):
        question = sup.q(TOPIC, "easy", seed)
        m, n = map(int, re.search(r"Z_(\d+) → Z_(\d+)", question.text).groups())
        k = int(re.search(r"(\d+)x mod", question.text).group(1))
        assert _is_homomorphism_from_cyclic(m, n, k), (seed, question.text)
        kernel = [x for x in range(m) if (k * x) % n == 0]
        assert int(question.answer) == len(kernel) > 1
        assert len(kernel) < m


def test_medium_key_is_an_image_or_kernel_order_of_a_power_map():
    seen = set()
    for seed in range(120):
        question = sup.q(TOPIC, "medium", seed)
        n = int(re.search(r"Z_(\d+)", question.text).group(1))
        k = int(re.search(r"x\^(\d+)", question.text).group(1))
        units = [x for x in range(1, n) if gcd(x, n) == 1]
        assert all(pow(x * y % n, k, n) == pow(x, k, n) * pow(y, k, n) % n for x in units for y in units[:6])
        if "image" in question.text:
            assert int(question.answer) == len({pow(x, k, n) for x in units}), (seed, question.text)
            seen.add("image")
        else:
            assert "kernel" in question.text
            assert int(question.answer) == sum(1 for x in units if pow(x, k, n) == 1), (seed, question.text)
            seen.add("kernel")
    assert seen == {"image", "kernel"}


def test_hard_key_is_the_kernel_of_a_map_from_a_product():
    for seed in range(100):
        question = sup.q(TOPIC, "hard", seed)
        a, b, c = map(int, re.search(r"Z_(\d+) × Z_(\d+) → Z_(\d+)", question.text).groups())
        u, v = map(int, re.search(r"(\d+)x \+ (\d+)y", question.text).groups())
        assert (a * u) % c == 0 and (b * v) % c == 0  # well defined on the product
        kernel = sum(1 for x in range(a) for y in range(b) if (u * x + v * y) % c == 0)
        image = {(u * x + v * y) % c for x in range(a) for y in range(b)}
        assert int(question.answer) == kernel, (seed, question.text)
        assert len(image) < c  # not onto, so |kernel| is not |domain| / |codomain|


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "number")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 6, "medium": 8, "hard": 12})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (2, 2, 2, 3), ("number",) * 4, (0, 1, 1, 0))


def test_exam_keys_by_listing_elements():
    for seed in range(150):
        question = sup.exam(TOPIC, seed)
        a_mod, b_mod = map(int, re.search(r"Z_(\d+) → Z_(\d+)", question.stem).groups())
        k = int(re.search(r"(\d+)x mod", question.stem).group(1))
        assert _is_homomorphism_from_cyclic(a_mod, b_mod, k)
        a, b, c, d = question.parts
        image = {(k * x) % b_mod for x in range(a_mod)}
        kernel = [x for x in range(a_mod) if (k * x) % b_mod == 0]
        assert int(a.answer) == len(image)
        assert int(b.answer) == len(kernel)
        y1, y2 = map(int, re.search(r"φ\(x\) = (\d+) or φ\(x\) = (\d+)", c.prompt).groups())
        assert int(c.answer) == sum(1 for x in range(a_mod) if (k * x) % b_mod in (y1, y2))
        target = int(re.search(r"with φ\(x\) = (\d+)", d.prompt).group(1))
        assert int(d.answer) == min(x for x in range(a_mod) if (k * x) % b_mod == target)


def test_exam_follow_through_from_a_wrong_image_order():
    s = sup.skill(SKILL_ID)
    checked = 0
    for seed in range(150):
        question = sup.exam(TOPIC, seed)
        a_mod = int(re.search(r"Z_(\d+) →", question.stem).group(1))
        true_a = int(question.parts[0].answer)
        for wrong in (true_a * 2, true_a // 2 if true_a % 2 == 0 else None):
            if wrong is None or a_mod % wrong:
                continue
            kernel = a_mod // wrong
            sent = {p.label: p.answer_text for p in question.parts} | {
                "a": str(wrong),
                "b": str(kernel),
                "c": str(2 * kernel),
            }
            if (
                kernel == int(question.parts[1].answer)
                or str(2 * kernel) == str(wrong)
                or kernel == wrong
                or kernel == 1
            ):
                continue
            result = grade(s, question, sent)
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, sent)
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, sent)
            checked += 1
    assert checked >= 10, checked
