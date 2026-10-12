"""The FSRS-5 model (mathgen/scheduling/fsrs.py). The golden sequence was computed
with the reference implementation, py-fsrs 5.1.3 (FSRS-5, default parameters), and
pins this port to it."""

from __future__ import annotations

import pytest

from mathgen.scheduling import fsrs

# (rating, elapsed days since the previous review, stability after, difficulty after)
GOLDEN = [
    (3, 0, 3.173, 5.282434),
    (3, 3, 10.738926, 5.272968),
    (2, 8, 14.868805, 6.027056),
    (1, 20, 2.697155, 7.297586),
    (3, 0.5, 3.796977, 7.27885),
    (3, 2, 7.153752, 7.2602),
    (4, 6, 34.307021, 6.799103),
    (3, 4.75, 40.407799, 6.782659),  # fractional gaps count as whole days, rounded down
    (2, 1.9, 40.759103, 7.285958),
]


def test_matches_reference_implementation():
    memory = None
    for rating, elapsed, stability, difficulty in GOLDEN:
        memory, _ = fsrs.step(memory, rating, elapsed)
        assert memory.stability == pytest.approx(stability, abs=1e-5)
        assert memory.difficulty == pytest.approx(difficulty, abs=1e-5)


def test_retrievability_is_ninety_percent_at_stability():
    assert fsrs.retrievability(12.5, 12.5) == pytest.approx(0.9)
    assert fsrs.retrievability(0, 12.5) == 1.0


def test_interval_equals_stability_at_ninety_percent_and_shrinks_with_higher_retention():
    assert fsrs.interval_days(20, 0.9) == pytest.approx(20)
    assert fsrs.interval_days(20, 0.95) < 20 < fsrs.interval_days(20, 0.8)


def test_first_review_has_no_prediction():
    memory, r = fsrs.step(None, fsrs.GOOD, 0)
    assert r is None
    assert memory.stability == fsrs.DEFAULT_WEIGHTS[2]


def test_a_lapse_never_raises_stability():
    memory = fsrs.Memory(stability=40.0, difficulty=5.0)
    after, r = fsrs.step(memory, fsrs.AGAIN, 40)
    assert r == pytest.approx(0.9)
    assert after.stability < memory.stability
    assert after.difficulty > memory.difficulty


def test_fractional_days_round_down():
    memory = fsrs.Memory(stability=10.0, difficulty=5.0)
    assert fsrs.step(memory, fsrs.GOOD, 4.99) == fsrs.step(memory, fsrs.GOOD, 4)
    assert fsrs.step(memory, fsrs.GOOD, 0.99) == fsrs.step(memory, fsrs.GOOD, 0)


@pytest.mark.parametrize("retention", [0, 1, -0.5, 1.5])
def test_interval_rejects_retention_outside_zero_to_one(retention):
    with pytest.raises(ValueError):
        fsrs.interval_days(10, retention)


@pytest.mark.parametrize("stability", [0, -1])
def test_retrievability_rejects_non_positive_stability(stability):
    with pytest.raises(ValueError):
        fsrs.retrievability(3, stability)


def test_bad_rating_rejected():
    with pytest.raises(ValueError):
        fsrs.step(None, 5, 0)
