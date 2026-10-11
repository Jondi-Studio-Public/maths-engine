"""Independent oracle for `f10-quadrilateral-properties`. It imports nothing from the generator: it reads the
rendered question (the stated property, the numbers and what is asked) and works the answer out again with its
own formulas, then compares with the keyed answer."""

import math
import re
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "f10c_quadrilateral_properties"
TIERS = ["easy", "medium", "hard"]


def _nums(text, pattern=r"(\d+)"):
    return [int(x) for x in re.findall(pattern, text)]


def oracle(text):
    body = re.sub(r" Give your answer in (cm|degrees), to 1 decimal place\.$", "", text)
    lengths = [int(v) for v in re.findall(r"[A-Z]{2} (?:=|is) (\d+) cm", body)]
    # -- algebra stems -------------------------------------------------------------
    m = re.search(r"\((\d+)x \+ (\d+)\) (cm|degrees) and .*?\((\d+)x - (\d+)\) (?:cm|degrees)", body)
    if m:
        p, q, r, s = int(m.group(1)), int(m.group(2)), int(m.group(4)), int(m.group(5))
        x = Fraction(q + s, r - p)
        assert x.denominator == 1 and 1 <= x
        value = p * x + q
        if "angle" in body:
            return Fraction(180) - value
        return 2 * value if "diagonal" in body else value
    # -- diagonals bisect -----------------------------------------------------------
    if "perimeter of triangle" in body:
        ac, bd, ab = lengths
        return Fraction(ab) + Fraction(ac + bd, 2)
    if re.search(r"[A-Z]M = \d+ cm and [A-Z]M = \d+ cm", body):
        halves = {a: int(v) for a, v in re.findall(r"([A-Z])M = (\d+) cm", body)}
        asked = re.findall(r"(?:diagonal |find )([A-Z]{2})\b", body)[-1]
        return Fraction(2 * halves[next(c for c in asked if c in halves)])
    if "bisect each other" in body and "parallelogram" in body and len(lengths) == 2:
        d = {}
        for a, b, v in re.findall(r"\b([A-Z])([A-Z]) = (\d+) cm", body):
            d[a + b] = int(v)
        asked = re.findall(r"(?:Find the length |to find )([A-Z])M", body)[0]
        whole = next(v for k, v in d.items() if asked in k)
        return Fraction(whole, 2)
    # -- rhombus / rectangle / kite lengths ---------------------------------------
    if "rhombus" in body and "perimeter of" in body and "Find the length of side" in body:
        return Fraction(int(re.search(r"perimeter of (\d+) cm", body).group(1)), 4)
    if "rhombus" in body and "Find the perimeter" in body:
        return Fraction(4 * lengths[0])
    if "perimeter" in body and "Opposite sides of a rectangle are equal" in body:
        p = int(re.search(r"perimeter (?:of|is) (\d+) cm", body).group(1))
        return Fraction(p, 2) - lengths[0]
    if "perimeter of the kite" in body:
        a, c = _nums(body, r"= (\d+) cm")
        return Fraction(2 * a + 2 * c)
    if "rhombus" in body and "diagonals" in body and "right angles" in body and "cm" in body:
        d1, d2 = lengths
        return Fraction(math.isqrt((d1 // 2) ** 2 + (d2 // 2) ** 2))
    if "rectangle" in body and "Find the length" in body:
        x, y = lengths
        return Fraction(math.isqrt(x * x + y * y), 2)
    if "perpendicular bisector" in body:
        ab, bd = lengths[0], lengths[-1]
        return Fraction(math.isqrt(ab * ab - (bd // 2) ** 2))
    # -- angles ----------------------------------------------------------------------
    deg = [int(x) for x in re.findall(r"[Aa]ngle [A-Z]{3} (?:=|is) (\d+) degrees", body)]
    if "kite" in body:
        alpha, beta = deg[:2]
        return Fraction(360 - alpha - 2 * beta)
    if "sum of angles" in body:
        return Fraction(360 - 2 * deg[0])
    if "cross at right angles" in body:
        return Fraction(90 - deg[0])
    if "same distance from every corner" in body:
        g = re.search(r"angle ([A-Z])M([A-Z]) = (\d+) degrees", body)
        a = re.search(r"Find angle M([A-Z])([A-Z])\b", body)
        t = int(g.group(3))
        return Fraction(180 - t, 2) if a.group(2) == g.group(2) else Fraction(t, 2)
    if "M" in body and "rectangle" in body and "angle" in body:  # angle ACB = u, find angle AMB
        return Fraction(2 * deg[0])
    if "isosceles triangles" in body:
        return Fraction(180 - deg[0], 2)
    if "same side" in body or "Consecutive angles" in body:
        return Fraction(180 - deg[0])
    raise AssertionError(f"oracle cannot read: {text!r}")


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_agrees_with_the_keyed_answer(difficulty):
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert oracle(q.text) == Fraction(q.answer_text), (q.text, q.answer_text)
        assert float(oracle(q.text)) == q.answer


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_distractor_equals_the_oracle_answer(difficulty):
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        want = oracle(q.text)
        assert all(abs(Fraction(d) - want) > Fraction(1, 10) for d in q.distractors), q.text
