"""Re-derives every answer of `f10_trig_ratios_similarity` from ``question.text`` alone, with
``math`` and ``decimal`` -- nothing is imported from the generator module, so a bug shared with its
in-generator check would have to be typed twice."""

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest

from mathgen import make

TOPIC = "f10_trig_ratios_similarity"
ROLE_WORDS = {"opposite side": "o", "adjacent side": "a", "hypotenuse": "h"}


def _round(x: float, dp: int) -> str:
    q = Decimal(1).scaleb(-dp)
    return str(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def oracle_easy(text: str) -> str:
    m = re.search(
        r"opposite x is (\d+) cm, its adjacent side is (\d+) cm and its hypotenuse is (\d+) cm\..*?"
        r"the (opposite side|adjacent side|hypotenuse) is (\d+) cm\..*?"
        r"the (opposite side|adjacent side|hypotenuse) of the second triangle",
        text,
    )
    assert m, text
    sides = {"o": int(m.group(1)), "a": int(m.group(2)), "h": int(m.group(3))}
    known, big, want = ROLE_WORDS[m.group(4)], int(m.group(5)), ROLE_WORDS[m.group(6)]
    return str(big * sides[want] // sides[known])


def _unit_sides(angle_deg: float) -> dict:
    t = math.radians(angle_deg)
    return {"h": 1.0, "o": math.sin(t), "a": math.cos(t)}


def oracle_medium(text: str) -> str:
    m = re.search(
        r"angle is (\d+) degrees\. The (side opposite the angle|leg adjacent to the angle|hypotenuse) is ([\d.]+) cm\..*?"
        r"length of the (side opposite the angle|leg adjacent to the angle|hypotenuse), correct to 1 decimal",
        text,
    )
    assert m, text
    words = {"side opposite the angle": "o", "leg adjacent to the angle": "a", "hypotenuse": "h"}
    s = _unit_sides(int(m.group(1)))
    return _round(float(m.group(3)) * s[words[m.group(4)]] / s[words[m.group(2)]], 1)


def oracle_hard(text: str) -> str:
    m = re.search(
        r"Triangle (\w{3}) has a right angle at (\w), and the angle at (\w) is (\d+) degrees\. "
        r"Side (\w\w) is ([\d.]+) cm\..*?length of side (\w\w), correct to 2",
        text,
    )
    assert m, text
    letters, right, at = m.group(1), m.group(2), m.group(3)
    other = next(c for c in letters if c not in (right, at))

    def role(side):
        s = set(side)
        return "h" if s == {at, other} else ("a" if s == {at, right} else "o")

    s = _unit_sides(int(m.group(4)))
    return _round(float(m.group(6)) * s[role(m.group(7))] / s[role(m.group(5))], 2)


@pytest.mark.parametrize("difficulty,oracle", [("easy", oracle_easy), ("medium", oracle_medium), ("hard", oracle_hard)])
def test_answer_matches_oracle(difficulty, oracle):
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert oracle(q.text) == q.answer_text, f"seed {seed}: {q.text}"


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_printed_ratios_are_correct_to_4dp(difficulty):
    for seed in range(100):
        text = make(TOPIC, seed=seed, difficulty=difficulty).text
        for fn, deg, val in re.findall(r"(sin|cos|tan) (\d+) degrees = (\d+\.\d{4})", text):
            exact = getattr(math, fn)(math.radians(int(deg)))
            assert val == _round(exact, 4)
