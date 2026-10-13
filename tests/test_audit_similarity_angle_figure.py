"""The angle shape of the similarity exam source must describe a triangle that can exist (sine rule)."""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make


@pytest.mark.parametrize("seed", range(120))
def test_angle_variant_obeys_sine_rule(seed):
    q = make("exd_x_sc6_similarity", seed=seed)
    m = re.match(r"In triangle (\w{3}), ∠(\w) = (\d+)° and ∠(\w) = (\d+)°\.", q.stem)
    if not m:
        pytest.skip("not the angle shape")
    tri, v1, a1, v2, a2 = m.groups()
    ang = {v1: int(a1), v2: int(a2)}
    ang[next(c for c in tri if c not in ang)] = 180 - int(a1) - int(a2)
    given = {name: float(val) for name, val in re.findall(r"([A-Z]{2}) = (\d+) cm", q.stem) if set(name) <= set(tri)}
    assert len(given) == 2
    ratios = []
    for name, length in given.items():
        opposite = next(c for c in tri if c not in name)
        ratios.append(length / math.sin(math.radians(ang[opposite])))
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-9)
