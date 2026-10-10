"""Independent oracle for `pv1f_graph_is_function`, checking exactly one option is true and it is the key.

Imports NOTHING from ``mathgen.generators.pv1f_function_graph_build``. Every question is
generated through the skill's own sources. The options are parsed from the RENDERED TEXT
(the key and the three distractors as the learner reads them) and each is judged against
the FIGURE in exact ``Fraction`` arithmetic:

- the set of points of the drawn graph on the line x = a is built from the polylines, the
  segments (an open end dot removes that end point unless a filled dot or another filled end
  restores it) and the separate dots;
- "is a function" holds when no line x = k/2 over the window meets the graph twice;
- an option is TRUE when its verdict matches and its stated count matches (once: exactly one
  point; more than once: at least two).

Exactly one option must be true and it must be the key, over 1000 seeds on every tier. ``PV1F_ORACLE_BIAS=1``
makes the oracle read every line as meeting the graph one point more often than it does, which must turn the
whole file RED.
"""

import functools
import os
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-understand-the-concept-of-the-graph-of-a"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = 1000
BIAS = int(os.environ.get("PV1F_ORACLE_BIAS", "0"))
OPT = re.compile(r"^It is (not )?a function: the line x = (-?\d+) meets the graph (once|more than once)\.$")


@functools.lru_cache(maxsize=1)
def _skill():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _gen(tier, seed):
    spec = next(s for s in _skill().sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


def _fr(v) -> Fraction:
    return Fraction(str(v))


def _on_line(fig, a: Fraction) -> int:
    """Number of distinct points of the drawn graph on the line x = a."""
    present, opened, filled = set(), set(), set()
    for curve in fig["curves"]:
        for piece in curve["pieces"]:
            for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
                x0, y0, x1, y1 = _fr(x0), _fr(y0), _fr(x1), _fr(y1)
                if x0 == x1:
                    if x0 == a:
                        present.update({y0, y1})
                elif min(x0, x1) <= a <= max(x0, x1):
                    present.add(y0 + (y1 - y0) * (a - x0) / (x1 - x0))
    for seg in fig["segments"]:
        (x0, y0), (x1, y1) = (_fr(seg["from"][0]), _fr(seg["from"][1])), (_fr(seg["to"][0]), _fr(seg["to"][1]))
        if x0 == x1:
            continue
        if min(x0, x1) <= a <= max(x0, x1):
            y = y0 + (y1 - y0) * (a - x0) / (x1 - x0)
            present.add(y)
            if a == x0 and seg.get("open_start") is True:
                opened.add(y)
            if a == x1 and seg.get("open_end") is True:
                opened.add(y)
            if a == x0 and seg.get("open_start") is False:
                filled.add(y)
            if a == x1 and seg.get("open_end") is False:
                filled.add(y)
    for p in fig["points"]:
        if _fr(p["at"][0]) == a:
            y = _fr(p["at"][1])
            if p.get("open"):
                opened.add(y)
            else:
                filled.add(y)
                present.add(y)
    return len((present - opened) | (filled & present))


def _is_function(fig) -> bool:
    lo, hi = (_fr(v) for v in fig["x_range"])
    k = lo * 2
    while k <= hi * 2:
        if _on_line(fig, k / 2) > 1:
            return False
        k += 1
    return True


def _true_options(q):
    fn = _is_function(q.figure)
    out = []
    for opt in [q.answer_text, *q.distractors]:
        m = OPT.match(opt)
        assert m, f"unparseable option {opt!r}"
        not_fn, a, how = bool(m.group(1)), Fraction(int(m.group(2))), m.group(3)
        n = _on_line(q.figure, a) + BIAS
        ok = (n >= 2 and how == "more than once" and not fn) if not_fn else (n == 1 and how == "once" and fn)
        if ok:
            out.append(opt)
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_exactly_one_option_is_true_and_it_is_the_key_1000_seeds(tier):
    for seed in range(SEEDS):
        q = _gen(tier, seed)
        assert _true_options(q) == [q.answer_text], (tier, seed, q.text, q.answer_text, q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_non_function_distractors_meet_at_most_once_and_the_key_meets_twice(tier):
    for seed in range(400):
        q = _gen(tier, seed)
        not_fn, a, _how = OPT.match(q.answer_text).groups()
        n_key = _on_line(q.figure, Fraction(int(a)))
        if not_fn:
            assert n_key >= 2 and not _is_function(q.figure)
            for d in q.distractors:
                assert _on_line(q.figure, Fraction(int(OPT.match(d).group(2)))) <= 1, (tier, seed, d)
        else:
            assert n_key == 1 and _is_function(q.figure)


def test_oracle_medium_jump_and_exam_dot_decide_the_verdict():
    for seed in range(300):
        q = _gen("medium", seed)
        not_fn, a, _h = OPT.match(q.answer_text).groups()
        if not_fn:
            assert _on_line(q.figure, Fraction(int(a))) >= 2  # both facing dots filled: two points at the jump
        else:
            assert _is_function(q.figure) and _on_line(q.figure, Fraction(int(a))) == 1
        q = _gen("exam", seed)
        not_fn, a, _h = OPT.match(q.answer_text).groups()
        dot_x = [p["at"][0] for p in q.figure["points"]][-1]
        if not_fn:
            assert int(a) == dot_x and _on_line(q.figure, Fraction(dot_x)) >= 2
        else:
            assert int(a) != dot_x and _on_line(q.figure, Fraction(dot_x)) == 1  # the dot adds no second point


def test_oracle_is_red_on_a_hand_built_wrong_figure():
    """Remove the second filled dot at a jump: the 'not a function' key stops being true."""
    q = None
    for seed in range(100):
        q = _gen("medium", seed)
        if q.answer_text.startswith("It is not"):
            break
    assert q.answer_text.startswith("It is not")
    a = int(OPT.match(q.answer_text).group(2))
    fig = {**q.figure, "segments": [dict(s) for s in q.figure["segments"]]}
    for s in fig["segments"]:
        if s["to"][0] == a and "open_end" in s:
            s["open_end"] = True
    q.figure = fig
    assert _true_options(q) != [q.answer_text]
