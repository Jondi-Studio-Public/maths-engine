"""``uc2-complex-log-powers-branches``: answers grade right, distractors wrong, each value is recomputed with cmath."""

from __future__ import annotations

import cmath
import re

import pytest
import sympy

from mathgen import make

from w5a_support import (
    ORACLE_SEEDS,
    answer_value,
    check_follow_through,
    check_grading,
    check_text,
    close,
    coverage,
    numeric,
    parse,
    questions,
    tier_topic,
)

NAME = "complex_log"
ALL = ("easy", "medium", "hard", "exam")
EPS = 1e-11


@pytest.mark.parametrize("tier", ALL)
def test_answers_and_distractors_grade(tier):
    for q in questions(NAME, tier):
        check_grading(q)
        check_text(q)


@pytest.mark.parametrize("tier", ALL)
def test_variety_and_determinism(tier):
    qs = questions(NAME, tier, range(1, 25))
    assert len({q.text for q in qs}) >= 12
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


# -- the oracle ----------------------------------------------------------------------------------------------------------


def _c(text: str) -> complex:
    return complex(sympy.N(parse(text), 20))


def _group_back(text: str, close_at: int) -> str:
    """The bracketed group that ends at text[close_at]."""
    depth, k = 0, close_at
    while k >= 0:
        depth += text[k] == ")"
        depth -= text[k] == "("
        if depth == 0:
            return text[k : close_at + 1]
        k -= 1
    raise AssertionError(text)


def _group_forward(text: str, open_at: int) -> str:
    depth, k = 0, open_at
    while k < len(text):
        depth += text[k] == "("
        depth -= text[k] == ")"
        if depth == 0:
            return text[open_at : k + 1]
        k += 1
    raise AssertionError(text)


def _principal_power(base: complex, w: complex) -> complex:
    return cmath.exp(w * cmath.log(base))


def _expected(text: str) -> complex | None:
    m = re.search(r"z1 = (.+?) and z2 = (.+?)(?:\.|,)\s", text)
    if m:
        e1, e2 = parse(m.group(1)), parse(m.group(2))
        z1, z2 = complex(sympy.N(e1, 20)), complex(sympy.N(e2, 20))
        product = complex(sympy.N(sympy.log(sympy.expand(e1 * e2)), 20))  # exact product: no stray imaginary part
        return product - cmath.log(z1) - cmath.log(z2)
    m = re.search(r"Log\(z\^\(?(-?\d+)\)?\)", text)
    if m:
        z_exact = parse(re.search(r"z = (.+?)(?= has modulus|,\s|\.\s|\.$)", text).group(1))
        n = int(m.group(1))
        # exact arithmetic for z^n: in floating point a negative real such as -64 can pick up a tiny imaginary part
        power = complex(sympy.N(sympy.log(sympy.expand(z_exact**n)), 20))
        return power - n * cmath.log(complex(sympy.N(z_exact, 20)))
    m = re.search(r"f\(z\) = Log\((.+?)\)(?: with| \(|\.|,)", text)
    if m and ("tends to" in text or "approach" in text or "→" in text):
        inner = parse(m.group(1))
        point = _c(re.search(r"(?:tends to|→|approaches) (-?\d+)", text).group(1))
        up = "Im z > 0" in text or "from above" in text
        zz = point + (1j if up else -1j) * EPS
        return cmath.log(complex(sympy.N(inner.subs(sympy.Symbol("z"), zz), 20)))
    m = re.search(r"z\^\((\d+)/(\d+)\)|\((\d+)/(\d+)\) Log z", text)
    if m and ("just above" in text or "f(-" in text or "jump" in text):
        p, q = (int(g) for g in (m.group(1), m.group(2))) if m.group(1) else (int(m.group(3)), int(m.group(4)))
        r = int(
            re.search(
                r"-(\d+)",
                text.split("jump")[-1] if "jump" in text else text.split("cut at")[-1] if "cut at" in text else text,
            ).group(1)
        )
        f = lambda w: cmath.exp(p / q * cmath.log(w))  # noqa: E731
        return f(complex(-r, EPS * 0.1)) - f(complex(-r, -EPS * 0.1))
    # a power: base^(exponent)
    j = text.find("^(")
    while j != -1:
        if text[max(0, j - 3) : j] == "Log" or text[j - 1] == "e":
            j = text.find("^(", j + 1)
            continue
        expo = _group_forward(text, j + 1)
        if text[j - 1] == ")":
            base = _group_back(text, j - 1)
        elif text[j - 1] == "z":
            base = "(" + re.search(r"z = (.+?)(?= has modulus|,\s|\.\s|\.$)", text).group(1) + ")"
        else:
            base = text[j - 1]
        if base == "(" + ")":
            break
        return _principal_power(_c(base), _c(expo))
    m = re.search(
        r"Log z for z = (.+?), in exact form|Evaluate Log\((.+?)\), where|logarithm of (.+?) as a \+ bi", text
    )
    if m:
        return cmath.log(_c(next(g for g in m.groups() if g)))
    return None


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_answer_is_recomputed_from_the_stem(tier):
    checked = 0
    qs = questions(NAME, tier, ORACLE_SEEDS)
    for q in qs:
        want = _expected(q.text)
        if want is None:
            continue
        got = answer_value(q)
        assert close(got, want, 1e-6), (q.text, q.answer_text, want)
        checked += 1
    coverage(checked, len(qs), 0.9)


def test_exam_parts_are_recomputed_from_the_stem():
    for q in questions(NAME, "exam", ORACLE_SEEDS):
        a = int(-sympy.sympify(parse(re.search(r"f\(z\) = Log\((.+?)\)", q.text).group(1)).subs(sympy.Symbol("z"), 0)))
        p = _c(re.search(r"Find f\((.+?)\) in exact form", q.parts[0].prompt).group(1))
        va = cmath.log(p - a)
        assert close(numeric(parse(q.parts[0].answer_text)), va, 1e-8), q.text
        assert q.parts[1].answer_text == f"x ≤ {a}", (q.text, q.parts[1].answer_text)
        below = _c(re.search(r"tends to (-?\d+)", q.parts[2].prompt).group(1))
        vc = cmath.log(complex(below - a, -EPS))
        assert close(numeric(parse(q.parts[2].answer_text)), vc, 1e-6), q.text
        assert close(numeric(parse(q.parts[3].answer_text)), va - vc, 1e-6), q.text
