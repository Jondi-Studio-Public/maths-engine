"""Independent oracle for the `p8c_line_best_fit_predict` questions: it re-derives each answer from the rendered
question text.

Parses ``question.text`` alone and independently recomputes the expected
answer from the parsed numbers -- substitution for easy, rearrangement for
medium, m * (x2 - x1) for hard. Never imports mathgen.generators.p8c_line_best_fit_predict_build.
"""

import re
from fractions import Fraction

from mathgen.core import make

TOPIC = "p8c_line_best_fit_predict"

_EQUATION_RE = re.compile(r"y = (-?\d+)x \+ (-?\d+)")
_EASY_X_RE = re.compile(r"when x = (-?\d+)\?")
_MEDIUM_Y_RE = re.compile(r"gives y = (-?\d+)\?")
_HARD_INTERVAL_RE = re.compile(r"from (-?\d+) to (-?\d+)\?")


def _oracle_answer(text):
    """Returns (answer:int, tier:str) derived purely from the question text."""
    eq = _EQUATION_RE.search(text)
    assert eq, f"could not parse the line's equation from: {text!r}"
    m, c = int(eq.group(1)), int(eq.group(2))

    easy_x = _EASY_X_RE.search(text)
    if easy_x:
        x = int(easy_x.group(1))
        return m * x + c, "easy"

    medium_y = _MEDIUM_Y_RE.search(text)
    if medium_y:
        y = int(medium_y.group(1))
        # x = (y - c) / m, exact by the skill's own construction
        x = Fraction(y - c, m)
        assert x.denominator == 1, f"non-integer inverse solution: {x} from {text!r}"
        return int(x), "medium"

    hard_interval = _HARD_INTERVAL_RE.search(text)
    if hard_interval:
        x1, x2 = int(hard_interval.group(1)), int(hard_interval.group(2))
        return m * (x2 - x1), "hard"

    raise AssertionError(f"could not classify question shape: {text!r}")


def test_oracle_matches_every_tier():
    seeds = range(40)
    tiers = ["easy", "medium", "hard"]
    challenged = []
    total = 0

    for tier in tiers:
        for seed in seeds:
            q = make(TOPIC, seed=seed, difficulty=tier)
            total += 1
            expected, detected_tier = _oracle_answer(q.text)
            assert detected_tier == tier, (
                f"seed {seed}: text shape looked like {detected_tier}, generator says {tier}: {q.text!r}"
            )
            if str(expected) != q.answer_text:
                challenged.append(f"seed {seed} {tier}: oracle={expected} answer_text={q.answer_text} text={q.text!r}")

    assert not challenged, "\n".join(challenged)
    assert total == len(tiers) * len(seeds)
