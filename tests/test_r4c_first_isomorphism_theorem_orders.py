"""Tests `uc2-first-isomorphism-theorem-orders`: kernels and images are recounted by running through every element of
the domain, with the maps read out of the question text."""

from __future__ import annotations

import re
from math import gcd

import mathgen.generators.r4c_first_isomorphism_theorem_orders_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4c_support as sup

TOPIC = "r4c_first_isomorphism_theorem_orders"
SKILL_ID = "uc2-first-isomorphism-theorem-orders"


def _product_map(text):
    m, n = (int(v) for v in re.search(r"Z_(\d+) × Z_(\d+)", text).groups())
    k = int(re.findall(r"Z_(\d+)", text)[-1])
    a, b = (int(v) for v in re.search(r"(\d+)x \+ (\d+)y", text).groups())
    return m, n, k, a, b


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "number")


def test_easy_kernel_by_brute_force():
    for seed in range(80):
        question = sup.q(TOPIC, "easy", seed)
        m, n = (int(v) for v in re.findall(r"Z_(\d+)", question.text)[:2])
        a = int(re.search(r"(\d+)x", question.text).group(1))
        assert (m * a) % n == 0, "the map must be well defined"
        kernel = [x for x in range(m) if (a * x) % n == 0]
        image = {(a * x) % n for x in range(m)}
        assert int(question.answer) == len(kernel) == m // len(image), (seed, question.text)
        assert len(kernel) > 1 and len(image) > 1


def test_medium_kernel_by_brute_force():
    for seed in range(60):
        question = sup.q(TOPIC, "medium", seed)
        m, n, k, a, b = _product_map(question.text)
        assert (m * a) % k == 0 and (n * b) % k == 0
        kernel = sum(1 for x in range(m) for y in range(n) if (a * x + b * y) % k == 0)
        assert int(question.answer) == kernel, (seed, question.text)


def test_hard_image_of_the_power_map():
    for seed in range(80):
        question = sup.q(TOPIC, "hard", seed)
        n = int(re.search(r"Z_(\d+)", question.text).group(1))
        e = int(re.search(r"x\^(\d+)", question.text).group(1))
        units = [x for x in range(n) if gcd(x, n) == 1]
        image = {(x**e) % n for x in units}
        kernel = [x for x in units if (x**e) % n == 1]
        assert int(question.answer) == len(image) == len(units) // len(kernel), (seed, question.text)
        assert len(image) != len(units) // gcd(e, len(units)), "must differ from the cyclic-group guess"


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "number")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 6, "medium": 8, "hard": 4})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (2, 1, 3, 2), ("number",) * 4, (0, 1, 0, 1))


def test_exam_keys_are_recomputed_from_the_text():
    for seed in range(60):
        question = sup.exam(TOPIC, seed)
        m, n, k, a, b = _product_map(question.stem)
        pa, pb, pc, pd = question.parts
        t = int(re.search(r"satisfy φ\(x, y\) = (\d+)", pd.prompt).group(1))
        values = [(a * x + b * y) % k for x in range(m) for y in range(n)]
        assert int(pa.answer) == len(set(values))
        assert int(pb.answer) == values.count(0)
        assert int(pd.answer) == values.count(t) > 0
        # (1, 0) + ker maps to a; its order in the quotient is the order of a in Z_k (an isomorphism onto the image)
        kernel = {(x, y) for x in range(m) for y in range(n) if (a * x + b * y) % k == 0}
        coset_order = next(j for j in range(1, m * n + 1) if (j % m, 0) in kernel)
        assert int(pc.answer) == coset_order


def test_exam_follow_through_chain():
    s = sup.skill(SKILL_ID)
    checked = 0
    for seed in range(40):
        question = sup.exam(TOPIC, seed)
        m, n = _product_map(question.stem)[:2]
        wrong = next((d for d in range(2, m * n) if (m * n) % d == 0 and d != int(question.parts[0].answer)), None)
        if wrong is None:
            continue
        kernel = m * n // wrong
        sent = {p.label: p.answer_text for p in question.parts} | {"a": str(wrong), "b": str(kernel), "d": str(kernel)}
        result = grade(s, question, sent)
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, sent)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, sent)
        checked += 1
    assert checked >= 8
