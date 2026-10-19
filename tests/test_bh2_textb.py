"""Regression tests for the 2026-10-19 bug hunt, text and key defects in the algebra, geometry, calculus,
vector and finance skills (worker textb)."""

import math
import re

import pytest

from mathgen import engine
from mathgen.core import display_question

_ERRS, _SKILLS = engine.load_skills()
SKILLS = {s.id: s for s in _SKILLS}


def _make(skill_id, seed, source):
    return engine.make_for_skill(SKILLS[skill_id], seed, source=source)


def _shown(skill_id, seed, source):
    q = _make(skill_id, seed, source)
    d = display_question(q)
    return q, d


def _all_text(d):
    out = [d["text"], d["answer_text"], *d["options"], *d["working"]]
    for p in d.get("parts", ()):
        out += [p["text"], p["answer_text"], *p["options"], *p["working"]]
    return out


# --- A1: the cosine/sine ratio angle question asked for one angle and keyed another ------------------------

COS_SIN = "senior-apply-the-cosine-and-sine-ratios-to-find"


@pytest.mark.parametrize("seed", range(80))
def test_angle_from_sides_key_matches_the_angle_asked(seed):
    q = _make(COS_SIN, seed, 1)
    m = re.search(r"an (opposite|adjacent) side of (\d+) units and a hypotenuse of (\d+) units", q.text)
    direction, leg, hyp = m.group(1), int(m.group(2)), int(m.group(3))
    other = math.sqrt(hyp**2 - leg**2)  # Pythagoras, no trig function shared with the generator
    if direction == "opposite":
        # "the acute angle that the opposite side is opposite to"
        assert "that the opposite side is opposite to" in q.text
        expected = math.degrees(math.atan2(leg, other))
    else:
        # the vertex where the adjacent side meets the hypotenuse: tan(angle) = other / leg
        assert "where the adjacent side meets the hypotenuse" in q.text
        expected = math.degrees(math.atan2(other, leg))
    assert q.answer_text == str(round(expected))
    assert engine.grade(SKILLS[COS_SIN], q, q.answer_text).correct
    for d in q.distractors:
        assert d != q.answer_text
