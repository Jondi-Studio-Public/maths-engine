"""Independent oracle for `pv1f_trig_graph_id`, re-deriving the key from the drawn curve.

Imports NOTHING from ``mathgen.generators.pv1f_trig_build``. Every question is generated
through the skill's own sources; the key is then re-derived FROM THE FIGURE by FEATURES, not
by fitting a formula (the generator's reader does that):

- a curve whose height leaves the band [-1.5, 1.5] is tan (sin and cos never exceed 1);
- otherwise the x of its highest sample, taken modulo 2 (x is in units of pi): 0 means a
  peak at 0, 2, -2 ... so cos; 1/2 means a peak at pi/2, -3pi/2 ... so sin.

The curve is the single drawn curve or, when the PROMPT names a lettered point, the curve
that point lies on. 1000 seeds per tier. ``PV1F_ORACLE_BIAS=1`` rotates the derived answer
(sin -> cos -> tan -> sin), which must turn the whole file RED.
"""

import functools
import os
import re

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-recognise-the-graphs-of-y-sin-x-y"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = 1000
BIAS = int(os.environ.get("PV1F_ORACLE_BIAS", "0"))
ORDER = ["sin", "cos", "tan"]


@functools.lru_cache(maxsize=1)
def _skill():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _gen(tier, seed):
    spec = next(s for s in _skill().sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


def _name_by_features(curve) -> str:
    samples = [(x, y) for piece in curve["pieces"] for x, y in piece]
    ys = [y for _x, y in samples]
    if max(abs(y) for y in ys) > 1.5:
        return "tan"
    top = max(ys)
    xs = [x for x, y in samples if abs(y - top) < 1e-9]
    assert top > 0.999
    residues = {round(x % 2, 6) for x in xs}
    assert len(residues) == 1, f"the peaks of one curve must share a phase, got {residues}"
    (r,) = residues
    if r in (0.0, 2.0):
        return "cos"
    assert r == 0.5, f"peak phase {r} is neither a cosine nor a sine"
    return "sin"


def _on(curve, pt) -> bool:
    px, py = pt
    for piece in curve["pieces"]:
        for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
            if x0 <= px <= x1:
                y = y0 if x1 == x0 else y0 + (y1 - y0) * (px - x0) / (x1 - x0)
                if abs(y - py) < 1e-6 and min(y0, y1) - 1e-6 <= py <= max(y0, y1) + 1e-6:
                    return True
    return False


def _derive(q) -> str:
    fig = q.figure
    m = re.search(r"point marked ([A-Z])\b", q.text)
    if m is None:
        assert len(fig["curves"]) == 1
        name = _name_by_features(fig["curves"][0])
    else:
        pt = next(p["at"] for p in fig["points"] if p["label"] == m.group(1))
        on = [c for c in fig["curves"] if _on(c, pt)]
        assert len(on) == 1, "the lettered point must lie on exactly one curve"
        name = _name_by_features(on[0])
    return f"y = {ORDER[(ORDER.index(name) + BIAS) % 3]} x"


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rederives_the_key_from_the_figure_1000_seeds(tier):
    for seed in range(SEEDS):
        q = _gen(tier, seed)
        assert _derive(q) == q.answer_text, (tier, seed, q.text)


@pytest.mark.parametrize("tier", ["hard", "exam"])
def test_oracle_drawn_curves_are_distinct_functions_by_features(tier):
    for seed in range(300):
        q = _gen(tier, seed)
        names = [_name_by_features(c) for c in q.figure["curves"]]
        assert len(set(names)) == len(names) == (2 if tier == "hard" else 3), (tier, seed, names)


def test_oracle_distractors_are_the_other_two_functions():
    for tier in TIERS:
        for seed in range(200):
            q = _gen(tier, seed)
            assert sorted([q.answer_text, *q.distractors]) == ["y = cos x", "y = sin x", "y = tan x"]


def test_oracle_is_red_on_a_hand_built_wrong_figure():
    """Flip a cosine's samples into a sine's phase: the oracle must read sine, not the key."""
    q = None
    for seed in range(40):
        q = _gen("easy", seed)
        if q.answer_text == "y = cos x":
            break
    assert q.answer_text == "y = cos x"
    shifted = {"pieces": [[[x, y] for x, y in piece] for piece in q.figure["curves"][0]["pieces"]]}
    xs = [x for piece in shifted["pieces"] for x, _y in piece]
    # re-label every sample's x by half a unit of pi: the peak moves from 0 to 1/2
    shifted["pieces"] = [[[x + 0.5, y] for x, y in piece] for piece in shifted["pieces"]]
    assert _name_by_features(shifted) == "sin" and min(xs) < 0
