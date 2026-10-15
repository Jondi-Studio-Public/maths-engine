"""Exam tier of senior-use-recursion-to-generate-an-arithmetic-sequence.

The oracle re-derives the four generated terms from the rendered prompt alone with exact fractions and
never imports the generator.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f3_arithrec"
_GRADER = GRADERS["ordered_list"]


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_prompt_variety_and_new_shapes_over_hard():
    shapes = {prompt_shape(_q(s)) for s in range(60)}
    hard = {
        prompt_shape(make("arithmetic_recursion_generate", seed=s, difficulty="hard", variant="from_two_terms"))
        for s in range(60)
    } | {prompt_shape(make("arithmetic_recursion_generate", seed=s, difficulty="hard")) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert len(shapes - hard) >= 3


_SUM = re.compile(r"t(\d+) \+ t(\d+) = (-?\d+) and t(\d+) = (-?\d+)\.")
_FIRST = re.compile(r"sum of the first (\d+) terms .* is (-?\d+), and t(\d+) = (-?\d+)\.")
_PAR = re.compile(r"are (.+?), (.+?) and (.+?), where k")


def _lin(expr: str):
    expr = expr.replace(" ", "")
    coef = const = 0
    for sign, body in re.findall(r"([+-]?)([^+-]+)", expr):
        mult = -1 if sign == "-" else 1
        if body.endswith("k"):
            coef += mult * (int(body[:-1]) if body[:-1] else 1)
        else:
            const += mult * int(body)
    return Fraction(coef), Fraction(const)


def _oracle(text):
    m = _SUM.search(text)
    if m:
        p, q, s, r, tr = map(int, m.groups())
        a_c, d_c, rhs = 2, p + q - 2, s
    else:
        m = _FIRST.search(text)
        if m:
            n, s, r, tr = map(int, m.groups())
            a_c, d_c, rhs = n, n * (n - 1) // 2, s
        else:
            m = _PAR.search(text)
            assert m, text
            (c1, e1), (c2, e2), (c3, e3) = (_lin(g) for g in m.groups())
            k = (e1 + e3 - 2 * e2) / (2 * c2 - c1 - c3)
            t1 = c1 * k + e1
            d = c2 * k + e2 - t1
            return [int(t1 + n * d) for n in range(1, 5)]
    # a_c*t1 + d_c*d = rhs ; t1 + (r-1)*d = tr
    det = Fraction(a_c * (r - 1) - d_c)
    t1 = (rhs * (r - 1) - d_c * tr) / det
    d = (a_c * tr - rhs) / det
    return [int(t1 + n * d) for n in range(1, 5)]


def test_key_matches_independent_oracle_over_400_seeds():
    bad = [(s, _q(s).text) for s in range(400) if _oracle(_q(s).text) != _q(s).answer]
    assert bad == [], bad[:3]


def test_all_three_forms_occur():
    kinds = set()
    for s in range(60):
        t = _q(s).text
        kinds.add("sum" if _SUM.search(t) else "first" if _FIRST.search(t) else "par")
    assert kinds == {"sum", "first", "par"}


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for s in range(150):
        q = _q(s)
        assert len(q.distractors) == 3
        assert _GRADER(q, q.answer_text).correct, s
        bad += [(s, d) for d in q.distractors if _GRADER(q, d).correct]
    assert bad == []


def test_grader_rejects_fixed_wrong_values():
    q = _q(2)
    t = q.answer
    for wrong in (", ".join(map(str, t[::-1])), ", ".join(map(str, t[:-1])), ", ".join(str(v + 1) for v in t), ""):
        assert not _GRADER(q, wrong).correct, wrong
