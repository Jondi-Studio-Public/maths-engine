"""Answer-class balance for pv1f_graph_is_function: a learner who always guesses
"it is not a function" (or always "it is a function") must not win more than about
half the time, at any tier."""

from __future__ import annotations

import pytest

from mathgen import make

_TOPIC = "pv1f_graph_is_function"
_SEEDS = 400


@pytest.mark.parametrize(
    ("tier", "lo", "hi"),
    [("easy", 0.45, 0.55), ("medium", 0.40, 0.60), ("hard", 0.40, 0.60), ("exam", 0.40, 0.60)],
)
def test_answer_class_balance(tier, lo, hi):
    not_function = sum(
        make(_TOPIC, seed=s, difficulty=tier).answer.startswith("It is not a function") for s in range(_SEEDS)
    )
    share = not_function / _SEEDS
    assert lo <= share <= hi, f"{tier}: 'not a function' is the key {share:.1%} of the time"
