"""The cylinder chains canonically carry the rounded earlier answer forward, so the stems must say to use it."""

from __future__ import annotations

import pytest

from mathgen import make


@pytest.mark.parametrize("seed", range(40))
def test_m4_later_parts_say_to_use_the_radius_from_a(seed):
    q = make("exd_x_m4_cylinder_surface_chain", seed=seed)
    assert all("radius from (a)" in p.prompt for p in q.parts[1:])


@pytest.mark.parametrize("seed", range(40))
def test_m5_later_parts_say_to_use_the_height_from_a(seed):
    q = make("exd_x_m5_cylinder_surface_chain", seed=seed)
    assert all("height from (a)" in p.prompt for p in q.parts[1:])
