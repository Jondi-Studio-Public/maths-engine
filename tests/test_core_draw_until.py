"""`core.draw_until` and `core.attempts`: bounded rejection sampling that keeps every seed's rng stream."""

import math
import random

import pytest

from mathgen.core import VerificationError, attempts, draw_until


def test_accepts_on_first_try():
    rng = random.Random(1)
    calls = []

    def draw(r):
        calls.append(1)
        return r.randint(1, 5)

    assert draw_until(rng, draw, lambda v: True) in range(1, 6)
    assert len(calls) == 1


def test_accepts_after_several_tries():
    rng = random.Random(3)
    value = draw_until(rng, lambda r: r.randint(1, 100), lambda v: v % 10 == 0, what="a multiple of ten")
    assert value % 10 == 0


def test_exhaustion_raises_naming_what_and_tries():
    rng = random.Random(0)
    with pytest.raises(VerificationError) as err:
        draw_until(rng, lambda r: r.randint(1, 3), lambda v: v > 3, tries=7, what="an impossible value")
    assert "an impossible value" in str(err.value)
    assert "7" in str(err.value)


def test_exhaustion_draws_exactly_tries_times():
    rng = random.Random(0)
    n = []
    with pytest.raises(VerificationError):
        draw_until(rng, lambda r: n.append(1) or r.random(), lambda v: False, tries=12)
    assert len(n) == 12


def _hand_written(rng):
    while True:
        a, b = rng.randint(1, 12), rng.randint(1, 12)
        if math.gcd(a, b) == 1 and a != b:
            return a, b


@pytest.mark.parametrize("seed", range(200))
def test_rng_stream_identical_to_while_true(seed):
    hand, helper = random.Random(seed), random.Random(seed)
    expected = _hand_written(hand)
    got = draw_until(
        helper,
        lambda r: (r.randint(1, 12), r.randint(1, 12)),
        lambda ab: math.gcd(*ab) == 1 and ab[0] != ab[1],
    )
    assert got == expected
    assert hand.getstate() == helper.getstate()


@pytest.mark.parametrize("seed", range(50))
def test_attempts_stream_identical_to_while_true(seed):
    hand, helper = random.Random(seed), random.Random(seed)
    while True:
        x = hand.randint(1, 50)
        if x % 7 == 0:
            break
    for _ in attempts(1000, "a multiple of seven"):
        y = helper.randint(1, 50)
        if y % 7 == 0:
            break
    assert x == y
    assert hand.getstate() == helper.getstate()


def test_attempts_exhaustion_raises_naming_what_and_limit():
    seen = 0
    with pytest.raises(VerificationError) as err:
        for _ in attempts(5, "never"):
            seen += 1
    assert seen == 5
    assert "never" in str(err.value) and "5" in str(err.value)
