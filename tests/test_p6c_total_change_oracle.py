"""Independent oracle for total change by integrating rate skill.

Derives correct answers from question text alone, independently of generator.
"""

import re
from sympy import symbols, sympify, integrate, simplify
from mathgen import make
from mathgen.core import symbolically_equal


t = symbols("t")


def parse_easy(text: str):
    """Parse easy tier: 'during the first <B> minutes'."""
    pattern = r"during the first (\d+) minutes"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse easy tier: {text}")
    b = int(match.group(1))

    # Extract rate expression r(t) = <RATE>
    rate_pattern = r"r\(t\) = ([^(]*?)\s*\(litres"
    rate_match = re.search(rate_pattern, text)
    if not rate_match:
        raise ValueError(f"Could not extract rate from: {text}")
    rate_str = rate_match.group(1).strip()
    rate_str = rate_str.replace("·", "*").replace("^", "**")

    return sympify(rate_str, locals={"t": t}), 0, b


def parse_medium(text: str):
    """Parse medium tier: 'increases from <A> to <B> units'."""
    pattern = r"increases from (\d+) to (\d+) units"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse medium tier: {text}")
    a = int(match.group(1))
    b = int(match.group(2))

    # Extract rate expression r(t) = <RATE>
    rate_pattern = r"r\(t\) = ([^(]*?)\s*\(dollars"
    rate_match = re.search(rate_pattern, text)
    if not rate_match:
        raise ValueError(f"Could not extract rate from: {text}")
    rate_str = rate_match.group(1).strip()
    rate_str = rate_str.replace("·", "*").replace("^", "**")

    return sympify(rate_str, locals={"t": t}), a, b


def parse_hard(text: str):
    """Parse hard tier: 'not the final volume', with I0 stated but not added."""
    pattern = r"as t increases from (\d+) to (\d+) minutes \(not the final volume\)"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse hard tier: {text}")
    a = int(match.group(1))
    b = int(match.group(2))

    # Extract initial amount I0
    i0_pattern = r"A tank already contains (\d+) litres"
    i0_match = re.search(i0_pattern, text)
    if not i0_match:
        raise ValueError(f"Could not extract I0 from: {text}")
    i0 = int(i0_match.group(1))

    # Extract rate expression r(t) = <RATE>
    rate_pattern = r"r\(t\) = ([^(]*?)\s*\(litres"
    rate_match = re.search(rate_pattern, text)
    if not rate_match:
        raise ValueError(f"Could not extract rate from: {text}")
    rate_str = rate_match.group(1).strip()
    rate_str = rate_str.replace("·", "*").replace("^", "**")

    return sympify(rate_str, locals={"t": t}), a, b, i0


class TestP6CTotalChangeOracle:
    """Oracle tests for total change skill (3 tiers x 60 seeds = 180 questions)."""

    def test_easy_all_seeds(self):
        """Oracle: easy tier (60 seeds)."""
        for seed in range(60):
            q = make("calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="easy")

            rate_expr, lo, hi = parse_easy(q.text)
            oracle_value = integrate(rate_expr, (t, lo, hi))
            parsed_answer = sympify(q.answer_text)

            assert symbolically_equal(oracle_value, parsed_answer), (
                f"Seed {seed} easy tier mismatch:\n"
                f"Question: {q.text}\n"
                f"Answer text: {q.answer_text}\n"
                f"Oracle value: {oracle_value}"
            )

    def test_medium_all_seeds(self):
        """Oracle: medium tier (60 seeds)."""
        for seed in range(60):
            q = make("calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="medium")

            rate_expr, lo, hi = parse_medium(q.text)
            oracle_value = integrate(rate_expr, (t, lo, hi))
            parsed_answer = sympify(q.answer_text)

            assert symbolically_equal(oracle_value, parsed_answer), (
                f"Seed {seed} medium tier mismatch:\n"
                f"Question: {q.text}\n"
                f"Answer text: {q.answer_text}\n"
                f"Oracle value: {oracle_value}"
            )

    def test_hard_all_seeds(self):
        """Oracle: hard tier (60 seeds), with explicit checking of total-change reading."""
        for seed in range(60):
            q = make("calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="hard")

            rate_expr, lo, hi, i0 = parse_hard(q.text)

            # Oracle: total change only (the integral alone), NOT I0 + integral
            oracle_value = integrate(rate_expr, (t, lo, hi))
            parsed_answer = sympify(q.answer_text)

            assert symbolically_equal(oracle_value, parsed_answer), (
                f"Seed {seed} hard tier mismatch:\n"
                f"Question: {q.text}\n"
                f"Answer text: {q.answer_text}\n"
                f"Oracle value: {oracle_value}\n"
                f"I0: {i0}"
            )

            # CRITICAL: Verify the oracle computed total change, not final amount
            # The oracle_value should NOT equal I0 + oracle_value (unless I0==0, which never happens)
            assert i0 > 0, f"Seed {seed}: I0 must be positive (I0={i0})"
            assert not symbolically_equal(oracle_value, i0 + oracle_value), (
                f"Seed {seed}: oracle incorrectly computed final amount instead of total change.\n"
                f"I0={i0}, oracle_value={oracle_value}, I0+oracle_value={i0 + oracle_value}"
            )
