"""Re-derives every answer of `f10_geometric_construction_algorithm` from ``question.text`` alone.
Nothing is imported from the generator module: every regex and rule is written fresh here."""

import math
import re

import pytest

from mathgen import make

TOPIC = "f10_geometric_construction_algorithm"


def _leg(hyp: int, other: int) -> int:
    sq = hyp * hyp - other * other
    root = math.isqrt(sq)
    assert root * root == sq and root > 0
    return root


def _hyp(a: int, b: int) -> int:
    sq = a * a + b * b
    root = math.isqrt(sq)
    assert root * root == sq
    return root


def oracle_easy(text: str) -> int:
    if "perpendicular bisector" in text:
        seg = int(re.search(r"segment \w\w, (\d+) cm long", text).group(1))
        refined = int(re.search(r"set to (\d+) cm instead", text).group(1))
        return _leg(refined, seg // 2)  # arcs of the refined radius from each end meet this far above the midpoint
    diameter = int(re.search(r"segment \w\w, (\d+) cm long", text).group(1))
    refined = int(re.search(r"set to (\d+) cm instead", text).group(1))
    return _leg(diameter, refined)  # angle in a semicircle is a right angle


_STEP_EFFECT = [
    (r"^Bisect", lambda a: a / 2),
    (r"^Copy the angle", lambda a: a * 2),
    (r"further 30 degrees", lambda a: a + 30),
    (r"further 60 degrees", lambda a: a + 60),
    (r"further 90 degrees", lambda a: a + 90),
    (r"60 degrees back", lambda a: a - 60),
]


def _step_effect(step: str):
    for pattern, fn in _STEP_EFFECT:
        if re.search(pattern, step):
            return fn
    raise AssertionError(step)


def oracle_medium(text: str) -> int:
    m = re.search(
        r"Step 1: .*?angle AOB is (\d+) degrees.*? Step 2: (.*?) Step 3: (.*?) Step 4: (.*?) "
        r"A refinement replaces Step (\d) with: (.*?) After",
        text,
    )
    assert m, text
    angle = int(m.group(1))
    steps = [m.group(2), m.group(3), m.group(4)]
    steps[int(m.group(5)) - 2] = m.group(6)
    for step in steps:
        angle = _step_effect(step)(angle)
    assert angle == int(angle)
    return int(angle)


def oracle_hard(text: str) -> int:
    half = int(re.search(r"Step 1: Draw segment \w\w, (\d+) cm long", text).group(1)) / 2
    mark = int(re.search(r"with \w\w = (\d+) cm\. Step 5", text).group(1))
    m = re.search(
        r"A refinement changes (Step \d) so that (?:\w\w = |\w\w is |the compasses are set to )(\d+) cm", text
    )
    step, value = m.group(1), int(m.group(2))
    if step == "Step 4":
        mark = value
    elif step == "Step 1":
        half = value / 2
    # Step 2 only moves the arcs: AC is unchanged
    return _hyp(int(half), mark)


@pytest.mark.parametrize("difficulty,oracle", [("easy", oracle_easy), ("medium", oracle_medium), ("hard", oracle_hard)])
def test_answer_matches_oracle(difficulty, oracle):
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert oracle(q.text) == int(q.answer_text), f"seed {seed}: {q.text}"


def test_arcs_always_meet():
    """Every compass setting in a perpendicular-bisector construction is longer than half the segment."""
    for difficulty in ("easy", "hard"):
        for seed in range(200):
            text = make(TOPIC, seed=seed, difficulty=difficulty).text
            if "meeting at point" not in text:
                continue
            seg = int(re.search(r"segment \w\w, (\d+) cm long", text).group(1))
            for r in re.findall(r"compasses to (\d+) cm|set to (\d+) cm", text):
                assert int(next(x for x in r if x)) * 2 > seg
