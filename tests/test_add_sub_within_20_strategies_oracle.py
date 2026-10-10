"""Independent oracle for `add_sub_within_20_strategies`: it evaluates the arithmetic printed in the prompt.

The prompt always contains one expression of two or three whole numbers joined by `+` and `-` (the strategy words
around it never contain a digit sequence joined by an operator). The oracle finds it, evaluates it left to right,
checks every running total stays within 0 to 20, and requires the printed answer to equal it and every
distractor to differ. For a three-number problem it also evaluates the expression with the terms regrouped
(add everything that adds, take away everything that is taken away), a second route. It does not import the
generator.

Seed count is `N_SEEDS` (default 250 per tier), overridable by `ADD_SUB_20_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "add_sub_within_20_strategies"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("ADD_SUB_20_ORACLE_SEEDS", "250"))
_EXPR = re.compile(r"\d+(?: [+-] \d+)+")


def _terms(text: str):
    found = _EXPR.findall(text)
    assert len(found) == 1, f"oracle: expected one expression in {text!r}, found {found}"
    tokens = found[0].split()
    return [int(tokens[0])] + [int(n) if s == "+" else -int(n) for s, n in zip(tokens[1::2], tokens[2::2])]


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_printed_expression(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        terms = _terms(q.text)
        running = 0
        for t in terms:
            running += t
            assert 0 <= running <= 20, f"{tier} seed {seed}: stage {running} in {q.text!r}"
        assert int(q.answer_text) == running, f"{tier} seed {seed}: {q.text!r} -> {q.answer_text}"
        # regrouped second route: everything that is added, less everything that is taken away
        assert sum(t for t in terms if t > 0) - sum(-t for t in terms if t < 0) == running
        assert all(int(d) != running for d in q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_strategy_named_in_the_prompt_fits_the_numbers(tier):
    """Where the prompt names a strategy the numbers must make it apply."""
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        terms = _terms(q.text)
        text = q.text.lower()
        if len(terms) == 2 and terms[1] > 0:
            x, y = terms
            if "ten" in text and "add 10" not in text and "add-10" not in text:
                assert x + y > 10, q.text  # make a ten: the sum passes ten
            if "near double" in text or "double that is close" in text:
                assert abs(x - y) == 1, q.text
            if "doubles" in text or ("double" in text and "near" not in text and "close" not in text):
                if abs(x - y) != 1:
                    assert x == y, q.text
        if len(terms) == 2 and terms[1] < 0:
            a, b = terms[0], -terms[1]
            if "through ten" in text or "to 10 first" in text:
                assert a > 10 > a - b, q.text
            if "add to" in text or "related addition" in text or "think" in text:
                assert a - b >= 3, q.text
        if "take" in text and "off" in text and len(terms) == 2:
            k = int(
                re.search(r"To add (\d+)|Add (\d+) by", q.text).group(1) or re.search(r"Add (\d+) by", q.text).group(1)
            )
            assert k in terms, q.text
