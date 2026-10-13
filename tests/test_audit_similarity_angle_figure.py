"""The angle shape of the similarity exam source: the printed figure must exist (sine rule, to rounding) and
must not give an answer away through equal angles or equal sides."""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make


def _angle_case(seed):
    q = make("exd_x_sc6_similarity", seed=seed)
    m = re.match(r"In triangle (\w{3}), ∠(\w) = (\d+)° and ∠(\w) = (\d+)°\.", q.stem)
    if not m:
        pytest.skip("not the angle shape")
    tri, v1, a1, v2, a2 = m.groups()
    ang = {v1: int(a1), v2: int(a2)}
    third = next(c for c in tri if c not in ang)
    ang[third] = 180 - int(a1) - int(a2)
    given = {name: float(val) for name, val in re.findall(r"([A-Z]{2}) = (\d+(?:\.\d+)?) cm", q.stem)}
    return q, tri, ang, third, given


@pytest.mark.parametrize("seed", range(120))
def test_shown_side_matches_sine_rule_to_rounding(seed):
    q, tri, ang, third, given = _angle_case(seed)
    sides = {n: v for n, v in given.items() if set(n) <= set(tri)}
    assert len(sides) == 2
    (n1, l1), (n2, l2) = sides.items()
    opp1 = next(c for c in tri if c not in n1)
    opp2 = next(c for c in tri if c not in n2)
    # whichever side is shown to 1 d.p. is within 0.05 of the sine-rule length from the other
    est2 = l1 * math.sin(math.radians(ang[opp2])) / math.sin(math.radians(ang[opp1]))
    est1 = l2 * math.sin(math.radians(ang[opp1])) / math.sin(math.radians(ang[opp2]))
    assert abs(est2 - l2) <= 0.05 + 1e-9 or abs(est1 - l1) <= 0.05 + 1e-9


@pytest.mark.parametrize("seed", range(120))
def test_no_giveaways(seed):
    q, tri, ang, third, given = _angle_case(seed)
    assert len(set(ang.values())) == 3
    theta = int(q.parts[0].answer_text)
    assert theta == ang[third] and theta not in (int(v) for v in re.findall(r"∠\w = (\d+)°", q.stem))
    sides = [v for n, v in given.items() if set(n) <= set(tri)]
    assert sides[0] != sides[1]
    # the side asked for in the second triangle (VW) differs from the one the stem states there (UV)
    stated_in_second = [v for n, v in given.items() if not set(n) <= set(tri)]
    assert float(q.parts[2].answer_text) != stated_in_second[0]


@pytest.mark.parametrize("seed", range(120))
def test_distractors_have_the_keys_decimal_places(seed):
    q, *_ = _angle_case(seed)
    for part in q.parts[1:]:
        places = len(part.answer_text.split(".")[1])
        assert all(len(d.split(".")[1]) == places for d in part.distractors if "." in d or places)
