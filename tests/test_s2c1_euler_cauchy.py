"""``uc2-euler-cauchy``: the generators grade their own answers right and every distractor wrong, tiers differ, exam parts chain."""

from __future__ import annotations

import pytest

from mathgen import make

from s2c1_support import second_route  # noqa: F401  (autouse fixture)
from s2c1_support import SEEDS, check_follow_through, check_grading, check_text, questions, tier_topic

NAME = "euler_cauchy"
ALL = ("easy", "medium", "hard", "exam")


@pytest.mark.parametrize("tier", ALL)
def test_answers_and_distractors_grade(tier):
    for q in questions(NAME, tier):
        check_grading(q)
        check_text(q)


@pytest.mark.parametrize("tier", ALL)
def test_variety_and_determinism(tier):
    qs = questions(NAME, tier, range(1, 25))
    texts = {q.text for q in qs}
    assert len(texts) >= 12, len(texts)
    again = make(tier_topic(NAME, tier), seed=7, difficulty=tier)
    assert again.to_dict() == make(tier_topic(NAME, tier), seed=7, difficulty=tier).to_dict()


def test_exam_chain_shape():
    for q in questions(NAME, "exam"):
        assert 3 <= len(q.parts) <= 4
        assert [p.label for p in q.parts] == list("abcd")[: len(q.parts)]
        assert any(p.follow_through for p in q.parts)
        check_follow_through(q)


def test_tiers_are_distinct():
    shapes = {t: {q.text for q in questions(NAME, t, range(1, 13))} for t in ("easy", "medium", "hard")}
    assert shapes["easy"] != shapes["medium"] != shapes["hard"]
    assert not (shapes["easy"] & shapes["hard"])


def test_single_part_tiers_have_no_parts():
    for t in ("easy", "medium", "hard"):
        assert all(not q.parts for q in questions(NAME, t, range(1, 6)))


def _read_equation(text):
    """(a, b, y0, v0) read back from the stem alone, independent of the generator's own variables."""
    import re

    eq = re.search(r"(x\^2 y''[^=«»]*?)\s*=\s*0", text).group(1)
    a = re.search(r"([+-]?\s*\d*)\s*x y'(?!')", eq)
    b = re.search(r"([+-]?\s*\d*)\s*y$", eq.strip())

    def num(m):
        if m is None:
            return 0
        s = m.group(1).replace(" ", "")
        return {"": 1, "+": 1, "-": -1}.get(s, None) if s in ("", "+", "-") else int(s)

    y0 = int(re.search(r"y\(1\) = (-?\d+)", text).group(1))
    v0 = int(re.search(r"y'\(1\) = (-?\d+)", text).group(1))
    return num(a), num(b), y0, v0


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_answer_solves_the_equation_in_the_stem(tier):
    import sympy

    from mathgen.grading.typed import _parse_expression

    x = sympy.Symbol("x", positive=True)
    for q in questions(NAME, tier):
        a, b, y0, v0 = _read_equation(q.text)
        sol = _parse_expression(q.answer_text).subs(sympy.Symbol("x"), x)
        residual = x**2 * sympy.diff(sol, x, 2) + a * x * sympy.diff(sol, x) + b * sol
        assert abs(float(residual.subs(x, 1.7))) < 1e-9, (q.text, q.answer_text)
        assert abs(float((sol - y0).subs(x, 1))) < 1e-9 and abs(float((sympy.diff(sol, x) - v0).subs(x, 1))) < 1e-9
