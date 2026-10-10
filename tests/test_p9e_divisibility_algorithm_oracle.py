"""Computational oracle: parses each rendered question's text and derives the
answer from scratch, using only what a learner is given (question.text,
question.answer_text, question.distractors). Deliberately does NOT import
anything from mathgen.generators.p9e_divisibility_algorithm_build -- no
helper, no constant, no shortcut. Every range template is re-executed by
literally iterating range(1, R + 1) and testing each number with % (never a
closed-form R // d division), and the iterated-state template is re-executed
by literally running the halving while-loop -- never a bit-trick 2-adic
valuation shortcut -- exactly as a learner tracing the stated algorithm by
hand would, and exactly what the question's own text describes.
"""

import dataclasses
import re

from mathgen import make

TOPIC = "p9e_factor_multiple_divisibility_algorithm"

_EASY_RE = re.compile(
    r"^Follow this rule: for each whole number from 1 to (\d+) inclusive, test "
    r"whether it is divisible by (\d+)\. Count how many numbers in the range "
    r"pass the test\.$"
)
_MEDIUM_RE = re.compile(
    r"^Follow this rule: for each whole number from 1 to (\d+) inclusive, test "
    r"whether it is divisible by (\d+) (AND|OR) divisible by (\d+)\. Count how "
    r"many numbers in the range pass the test\.$"
)
# \1 backreference: the algorithm's start value and the question's own
# "starting from N" restatement must literally be the same number, not merely
# both parseable -- structurally enforced by the regex, not a separate assert.
_HARD_RE = re.compile(
    r"^Start with (\d+)\. While the current number is even, divide it by 2 and "
    r"count one step\. Stop as soon as the number is odd\. How many steps does "
    r"this algorithm take, starting from \1\?$"
)


def _oracle_easy_count(text: str):
    """Literal re-execution: range(1, R+1) tested with %, never R // d."""
    m = _EASY_RE.match(text)
    if not m:
        return None
    R, d = int(m.group(1)), int(m.group(2))
    count = 0
    for n in range(1, R + 1):
        if n % d == 0:
            count += 1
    return count


def _oracle_medium_count(text: str):
    """Literal re-execution: range(1, R+1) tested with %, never a closed-form
    lcm/gcd count."""
    m = _MEDIUM_RE.match(text)
    if not m:
        return None
    R, d1, gate, d2 = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
    count = 0
    for n in range(1, R + 1):
        passes_d1 = n % d1 == 0
        passes_d2 = n % d2 == 0
        passes = (passes_d1 and passes_d2) if gate == "AND" else (passes_d1 or passes_d2)
        if passes:
            count += 1
    return count


def _oracle_hard_steps(text: str):
    """Literal while-loop re-execution, never bin(N).count-style bit tricks.
    Returns (step_count, final_odd_value)."""
    m = _HARD_RE.match(text)
    if not m:
        return None
    N = int(m.group(1))
    current = N
    steps = 0
    while current % 2 == 0:
        current //= 2
        steps += 1
    return steps, current


def _challenge_easy(q) -> str | None:
    expected = _oracle_easy_count(q.text)
    if expected is None:
        return f"could not parse {q.text!r}"
    if int(q.answer_text) != expected:
        return f"answer {q.answer_text} != oracle {expected} for {q.text!r}"
    return None


def _challenge_medium(q) -> str | None:
    expected = _oracle_medium_count(q.text)
    if expected is None:
        return f"could not parse {q.text!r}"
    if int(q.answer_text) != expected:
        return f"answer {q.answer_text} != oracle {expected} for {q.text!r}"
    return None


def _challenge_hard(q) -> str | None:
    result = _oracle_hard_steps(q.text)
    if result is None:
        return f"could not parse {q.text!r}"
    expected_steps, final_odd = result
    if int(q.answer_text) != expected_steps:
        return f"answer {q.answer_text} != oracle steps {expected_steps} for {q.text!r}"
    if int(q.answer_text) == final_odd:
        return f"step count {q.answer_text} coincides with final odd value {final_odd}"
    return None


def test_easy_oracle():
    challenged = [(seed, _challenge_easy(make(TOPIC, seed=seed, difficulty="easy"))) for seed in range(60)]
    challenged = [(s, c) for s, c in challenged if c]
    assert not challenged, f"{len(challenged)} of 60 challenged: {challenged}"
    print("easy oracle: examined 60, challenged 0")


def test_medium_oracle():
    challenged = [(seed, _challenge_medium(make(TOPIC, seed=seed, difficulty="medium"))) for seed in range(60)]
    challenged = [(s, c) for s, c in challenged if c]
    assert not challenged, f"{len(challenged)} of 60 challenged: {challenged}"
    print("medium oracle: examined 60, challenged 0")


def test_hard_oracle():
    challenged = [(seed, _challenge_hard(make(TOPIC, seed=seed, difficulty="hard"))) for seed in range(60)]
    challenged = [(s, c) for s, c in challenged if c]
    assert not challenged, f"{len(challenged)} of 60 challenged: {challenged}"
    print("hard oracle: examined 60, challenged 0")


def test_oracle_fires_on_a_corrupted_answer():
    """Corrupt each tier's answer_text via dataclasses.replace (Question is a
    dataclass) -- never re-derive from generator internals -- and confirm the
    matching _challenge function actually flags it. Proves the oracle's
    comparison is load-bearing, not a tautological x+1 != x check."""
    q = make(TOPIC, seed=0, difficulty="easy")
    corrupted = dataclasses.replace(q, answer_text=str(int(q.answer_text) + 1))
    assert _challenge_easy(corrupted) is not None

    q = make(TOPIC, seed=0, difficulty="medium")
    corrupted = dataclasses.replace(q, answer_text=str(int(q.answer_text) + 1))
    assert _challenge_medium(corrupted) is not None

    q = make(TOPIC, seed=0, difficulty="hard")
    corrupted = dataclasses.replace(q, answer_text=str(int(q.answer_text) + 1))
    assert _challenge_hard(corrupted) is not None
