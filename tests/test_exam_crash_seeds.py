"""Seeds that used to raise because a part's slips left it with fewer than two distractors.

Each generator now draws again in that case (`DrawRejected` under `distinct_part_answers`, or the network chain's own
rejection loop), so these seeds build a full question with at least two distractors in every part."""

import pytest

from mathgen import make

CASES = [
    ("exd_x_r2e_network_crash", 232),
    ("exd_x_sc5_log_equation", 278),
    ("exd_x_sc5_log_equation", 362),
    ("exd_x_sc5_log_equation", 470),
    ("exd_x_u3_double_integral_average", 235),
    ("exd_x_u3_double_integral_average", 493),
    ("exd_x_r3j_collision_impulse", 245),
    ("exd_x_r3j_collision_impulse", 348),
]


@pytest.mark.parametrize(("topic", "seed"), CASES)
def test_former_crash_seed_builds(topic, seed):
    q = make(topic, seed=seed, difficulty="exam")
    assert q.parts
    for part in q.parts:
        assert len(part.distractors) >= 2, part.prompt
