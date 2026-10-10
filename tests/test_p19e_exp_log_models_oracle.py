"""Independent oracle for the `p19e_exp_log_models` questions: it re-derives each answer from the rendered
question text.

Imports NOTHING from `mathgen.generators.p19e_exp_log_models_build` -- every formula here is re-derived from
`question.text` alone, using regexes written against the question WORDING, never the generator's own code or
tables.

Seed count is `N_SEEDS` per tier (default 250, so >= 1000 questions over the
four tiers per topic), overridable by `P19E_ORACLE_SEEDS`.
"""

from __future__ import annotations

import math
import os
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen import make

N_SEEDS = int(os.environ.get("P19E_ORACLE_SEEDS", "250"))

TOPIC_EXP = "p19e_exponential_model_value"
TOPIC_LOG = "p19e_log_model_solve_input"
DIFFS = ["easy", "medium", "hard", "exam"]

_TOL = 0.06  # > half a unit at 1 d.p. (0.05); the oracle only needs to agree
# with the canonical to within its own stated rounding, not
# exactly reproduce every intermediate float bit-for-bit.


def _round(value: float, dp: int) -> str:
    scaled = round(value, dp)
    return f"{scaled:.{dp}f}"


# =============================================================================
# Row 1 oracle: p19e_exponential_model_value
# =============================================================================


def _oracle_exp_value(text: str) -> tuple[float, int]:
    """Returns (recomputed value, dp) for any tier/variant of the exponential
    row, parsed entirely from the rendered text."""
    if "then" in text or "for a further" in text or "for the next" in text:
        return _oracle_exp_two_phase(text), 1
    if "between" in text:
        return _oracle_exp_two_time_gap(text), 1
    if "e^(" in text:
        return _oracle_exp_continuous(text), 1
    if "half-life" in text or "doubling time" in text:
        return _oracle_exp_half_life(text), 1
    return _oracle_exp_single_phase(text)


def _direction_and_r(text: str) -> tuple[str, int]:
    r = int(re.search(r"(\d+)%", text).group(1))
    if re.search(r"\b(grows|increases|gains)\b", text):
        return "growth", r
    if re.search(r"\b(decays|decreases|loses|shrinks)\b", text):
        return "decay", r
    raise AssertionError(f"could not determine direction from: {text!r}")


def _oracle_exp_single_phase(text: str) -> tuple[float, int]:
    A0 = int(re.search(r"(?:starts at|begins at|Starting at) (\d+)", text).group(1))
    direction, r = _direction_and_r(text)
    period = re.search(r"(?:every|each|per) (year|month|week|day|hour)", text).group(1)
    elapsed_count, elapsed_unit = re.search(r"after (\d+) (\w+)", text).groups()
    elapsed_count = int(elapsed_count)
    if period == "year" and elapsed_unit == "months":
        exponent = elapsed_count / 12
        dp = 1
    else:
        exponent = elapsed_count
        dp = 0 if "nearest whole number" in text else 1
    factor = 1 + r / 100 if direction == "growth" else 1 - r / 100
    value = A0 * math.exp(exponent * math.log(factor))
    return value, dp


def _oracle_exp_continuous(text: str) -> float:
    m = re.search(r"Q = (\d+) e\^\(([-\d.]+)t\)", text)
    A0, k = int(m.group(1)), float(m.group(2))
    t = int(re.search(r"t = (\d+)", text).group(1))
    return A0 * math.exp(k * t)


def _oracle_exp_half_life(text: str) -> float:
    A0 = int(re.search(r"(?:starts at|begins at) (\d+)", text).group(1))
    mode = "half_life" if "half-life" in text else "doubling"
    h = int(re.search(r"(?:half-life|doubling time)(?: of| is)? (\d+)", text).group(1))
    t = int(re.search(r"t = (\d+)", text).group(1))
    base = 0.5 if mode == "half_life" else 2.0
    return A0 * math.exp((t / h) * math.log(base))


def _oracle_exp_two_phase(text: str) -> float:
    A0 = int(re.search(r"(?:starts at|begins at) (\d+)", text).group(1))
    r1, r2 = (int(v) for v in re.findall(r"(\d+)%", text))
    t1, t2 = (int(v) for v in re.findall(r"for(?: a further| the next)? (\d+) \w+s", text))
    verb1, verb2 = re.findall(r"\b(grows|decays)\b", text)
    factor1 = 1 + r1 / 100 if verb1 == "grows" else 1 - r1 / 100
    factor2 = 1 + r2 / 100 if verb2 == "grows" else 1 - r2 / 100
    mid = A0 * math.exp(t1 * math.log(factor1))
    return mid * math.exp(t2 * math.log(factor2))


def _oracle_exp_two_time_gap(text: str) -> float:
    A0 = int(re.search(r"(?:starts at|begins at) (\d+)", text).group(1))
    direction, r = _direction_and_r(text)
    t1, t2 = (int(v) for v in re.search(r"between (?:t = )?(\d+) \w+s and (?:t = )?(\d+) \w+s", text).groups())
    factor = 1 + r / 100 if direction == "growth" else 1 - r / 100
    val1 = A0 * math.exp(t1 * math.log(factor))
    val2 = A0 * math.exp(t2 * math.log(factor))
    return abs(val2 - val1)


@pytest.mark.parametrize("difficulty", DIFFS)
def test_exp_oracle_matches_canonical(difficulty):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(TOPIC_EXP, seed=seed, difficulty=difficulty)
        recomputed, dp = _oracle_exp_value(q.text)
        recomputed_rounded = float(_round(recomputed, dp))
        expected = float(q.answer_text)
        assert abs(recomputed_rounded - expected) <= _TOL, (difficulty, seed, q.text, q.answer_text, recomputed)
        assert grade_number(q, _round(recomputed, dp)).correct is True, (difficulty, seed, q.text)
        checked += 1
    assert checked == N_SEEDS


# =============================================================================
# Row 2 oracle: p19e_log_model_solve_input
# =============================================================================


def _parse_b_part(b_part: str) -> int:
    b_part = b_part.strip()
    if not b_part:
        return 0
    sign = -1 if b_part.startswith("-") else 1
    return sign * int(b_part.lstrip("+- ").strip())


def _oracle_log_value(text: str) -> tuple[float, int]:
    if "reach" in text:
        return _oracle_log_exam(text), 1
    if "log10(t +" in text:
        return _oracle_log_hard(text), 1
    if "ln(x)" in text:
        return _oracle_log_medium(text), 1
    return _oracle_log_easy(text), 1


def _oracle_log_easy(text: str) -> float:
    m = re.search(r"y = (\d+) log10\(x\)([+\- ]*\d*)", text)
    a = int(m.group(1))
    b = _parse_b_part(m.group(2))
    y = int(re.search(r"when y = (-?\d+)", text).group(1))
    return 10 ** ((y - b) / a)


def _oracle_log_medium(text: str) -> float:
    # The coefficient is bracketed only when it's a genuine fraction ("(5/2)
    # ln(x)"); a bare integer prints without parens ("2 ln(x)"), so the
    # parens are optional here.
    m = re.search(r"y = \(?(-?\d+(?:/\d+)?)\)? ln\(x\)([+\- ]*\d*)", text)
    a_text = m.group(1)
    a = (int(a_text.split("/")[0]) / int(a_text.split("/")[1])) if "/" in a_text else float(a_text)
    b = _parse_b_part(m.group(2))
    y = int(re.search(r"when y = (-?\d+)", text).group(1))
    return math.exp((y - b) / a)


def _oracle_log_hard(text: str) -> float:
    m = re.search(r"y = (\d+) log10\(t \+ (\d+)\)([+\- ]*\d*)", text)
    a, c = int(m.group(1)), int(m.group(2))
    b = _parse_b_part(m.group(3))
    y = int(re.search(r"when y = (-?\d+)", text).group(1))
    return 10 ** ((y - b) / a) - c


def _oracle_log_exam(text: str) -> float:
    m = re.search(r"y = (\d+) log10\(x\)([+\- ]*\d*)", text)
    a = int(m.group(1))
    b = _parse_b_part(m.group(2))
    y1, y2 = (int(v) for v in re.findall(r"reach(?:es)? (-?\d+)", text))
    x1 = 10 ** ((y1 - b) / a)
    x2 = 10 ** ((y2 - b) / a)
    return x2 - x1


@pytest.mark.parametrize("difficulty", DIFFS)
def test_log_oracle_matches_canonical(difficulty):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(TOPIC_LOG, seed=seed, difficulty=difficulty)
        recomputed, dp = _oracle_log_value(q.text)
        expected = float(q.answer_text)
        assert abs(recomputed - expected) <= _TOL, (difficulty, seed, q.text, q.answer_text, recomputed)
        assert grade_number(q, _round(recomputed, dp)).correct is True, (difficulty, seed, q.text)
        checked += 1
    assert checked == N_SEEDS
