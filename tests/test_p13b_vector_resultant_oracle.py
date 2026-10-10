"""Independent oracle for the `p13b_vector_resultant` questions: it re-derives each answer from the rendered
question text.

Imports nothing from `mathgen.generators.p13b_vector_resultant_build`: not
the angle pools, not the templates, not the misconception formulas. Parses
the numbers straight out of the RENDERED `question.text` and re-derives the
correct answer from scratch via plain `math`/`sympy` trig calls and its own
half-up rounding, a route wholly independent of the generator's own
`math.cos`/`math.radians` construction.

Run across >= 1000 seeds total, spread over the four tiers (250 each).
"""

from __future__ import annotations

import math
import re

from mathgen import make

TOPIC = "p13b_vector_resultant_angle"
SEEDS_PER_TIER = 260  # 4 tiers x 260 = 1040 >= 1000


def _round_half_up_1dp(value: float) -> str:
    """Independent re-implementation of half-up rounding to 1 decimal place
    -- deliberately NOT a call to `mathgen.core.round_half_up`."""
    scaled = math.floor(abs(value) * 10 + 0.5)
    sign = "-" if value < 0 and scaled != 0 else ""
    whole, frac = divmod(scaled, 10)
    return f"{sign}{whole}.{frac}"


# --- easy: resultant magnitude via the cosine rule --------------------------
# All three easy templates state F1, then F2, then theta, in that order.

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def test_oracle_easy_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="easy")
        nums = [int(n) for n in _NUM_RE.findall(q.text)]
        F1, F2, theta = nums[0], nums[1], nums[2]
        R = math.sqrt(F1 * F1 + F2 * F2 + 2 * F1 * F2 * math.cos(math.radians(theta)))
        assert _round_half_up_1dp(R) == q.answer_text, (seed, q.text, q.answer_text, R)
        checked += 1
    assert checked == SEEDS_PER_TIER


# --- medium: resultant direction via the sine rule --------------------------
# All three medium templates state F1, F2, theta, then repeat F1 once more.


def test_oracle_medium_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="medium")
        nums = [int(n) for n in _NUM_RE.findall(q.text)]
        F1, F2, theta = nums[0], nums[1], nums[2]
        theta_r = math.radians(theta)
        x = F1 + F2 * math.cos(theta_r)
        y = F2 * math.sin(theta_r)
        alpha = math.degrees(math.atan2(y, x))
        assert _round_half_up_1dp(alpha) == q.answer_text, (seed, q.text, q.answer_text, alpha)
        checked += 1
    assert checked == SEEDS_PER_TIER


# --- hard: river-crossing drift, one regex per template (orders differ) ---

_HARD_RES = [
    re.compile(
        r"speed through the water is (?P<u>\d+) km/h, directed straight "
        r"across a river (?P<w>\d+) km wide\. The current flows parallel "
        r"to the bank at (?P<c>\d+) km/h"
    ),
    re.compile(
        r"swimming at (?P<u>\d+) km/h relative to the water, heads "
        r"straight across a channel (?P<w>\d+) km wide\. A current of "
        r"(?P<c>\d+) km/h"
    ),
    re.compile(
        r"crosses a river (?P<w>\d+) km wide, travelling straight "
        r"across at (?P<u>\d+) km/h through the water while a current "
        r"of (?P<c>\d+) km/h"
    ),
]


def _parse_hard(text: str):
    for rx in _HARD_RES:
        m = rx.search(text)
        if m:
            return int(m["u"]), int(m["w"]), int(m["c"])
    raise AssertionError(f"no hard-tier template matched: {text!r}")


def test_oracle_hard_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="hard")
        u, w, c = _parse_hard(q.text)
        assert w % u == 0, (seed, q.text, u, w)
        t = w // u
        drift = c * t
        assert str(drift) == q.answer_text, (seed, q.text, q.answer_text, drift)
        checked += 1
    assert checked == SEEDS_PER_TIER


# --- exam: resultant of three vectors via full component resolution -------
# All three exam templates state F1, F2, a2, F3, a3, in that order.


def test_oracle_exam_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="exam")
        nums = [int(n) for n in _NUM_RE.findall(q.text)]
        F1, F2, a2, F3, a3 = nums[0], nums[1], nums[2], nums[3], nums[4]
        Fx = F1 + F2 * math.cos(math.radians(a2)) + F3 * math.cos(math.radians(a3))
        Fy = F2 * math.sin(math.radians(a2)) + F3 * math.sin(math.radians(a3))
        R = math.hypot(Fx, Fy)
        assert _round_half_up_1dp(R) == q.answer_text, (seed, q.text, q.answer_text, R)
        checked += 1
    assert checked == SEEDS_PER_TIER
