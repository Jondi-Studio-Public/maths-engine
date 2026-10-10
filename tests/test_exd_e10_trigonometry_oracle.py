"""Independent oracle for triangle-area-sine's hard tier. Re-derives from rendered question text alone -- does
not import mathgen.generators.geometry.
"""

from __future__ import annotations

import math
import re

from mathgen import make

_AREA_RE = re.compile(r"side a = (\d+), included angle C = (\d+)°, and area (-?\d+\.?\d*)")


def test_oracle_triangle_area_reverse():
    checked = 0
    for seed in range(200):
        q = make("trigonometry", seed=seed, kinds=("triangle_area_reverse",))
        m = _AREA_RE.search(q.text)
        assert m, f"unexpected text: {q.text!r}"
        a_side, c_deg, area = int(m.group(1)), int(m.group(2)), float(m.group(3))
        expected = round(2 * area / (a_side * math.sin(math.radians(c_deg))), 1)
        actual = float(q.answer_text.replace("b = ", ""))
        assert abs(expected - actual) < 0.05, f"seed={seed}: {q.text!r} -> {q.answer_text!r}"
        checked += 1
    assert checked == 200
