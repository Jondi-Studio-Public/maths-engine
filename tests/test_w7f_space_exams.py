"""Exam chains for the Years 3-6 space skills and the scale-plan skill (the w7f work).

Oracles read the plan numbers, the shaded and hatched cells, the point P and the scale off the FIGURE and the numbers
off the prompts, and use turning and counting of their own; none calls the generator's helpers.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import mathgen.generators.w7f_space_exams  # noqa: F401  (registers the topics)
from tests import w7b_support as sup

MODEL = ("f10-model-plans-cube-counts", "grid_model_plans", "exd_x_w7f_model_plan_chain")
VIEWS = ("f10-two-dimensional-object-representations", "grid_plan_views", "exd_x_w7f_plan_views_chain")
TURNS = ("f10-turn-shapes-on-a-grid", "grid_shape_turns", "exd_x_w7f_turn_pictures_chain")
BOX = ("f10-box-nets-on-a-grid", "grid_prism_nets", "exd_x_w7f_box_net_chain")
PLAN = ("senior-construct-scale-drawings-by-hand-and-by-using", "grid_scale_plan", "exd_x_w7f_scale_plan_chain")
ANSWER_KIND = {
    MODEL[0]: "number",
    VIEWS[0]: "grid_cells",
    TURNS[0]: "grid_cells",
    BOX[0]: "grid_cells",
    PLAN[0]: "grid_cells",
}


def _wiring(spec):
    skill = sup.skill_of(spec[0])
    sup.check_wiring(skill, spec[1], spec[2], ANSWER_KIND[spec[0]])
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and steps[-1] > steps[0]
    return skill


def test_model_plan_chain():
    skill = _wiring(MODEL)
    sup.check_exam_chain(skill, MODEL[2], [1, 2, 1], ["number"] * 3)
    for seed in range(80):
        q = sup.exam(MODEL[2], seed)
        total = int(re.search(r"built from (\d+) cubes", q.text).group(1))
        shown = {(lb["at"][0], lb["at"][1]): lb["text"] for lb in q.figure["labels"]}
        assert list(shown.values()).count("?") == 1
        known = [int(t) for t in shown.values() if t != "?"]
        a = total - sum(known)
        stacks = len(shown)
        tallest = max(known)
        assert 1 <= a <= tallest
        assert [int(p.answer_text) for p in q.parts] == [a, tallest * stacks - total, tallest * stacks], seed
        # every stack with a number sits on a shaded square; bare squares are unshaded
        shaded = {tuple(c) for c in q.figure["cells"][0]["cells"]}
        assert {(int(x), int(y)) for x, y in shown} == shaded


def test_model_follow_through():
    def wrong(q, canon):
        total = int(re.search(r"built from (\d+) cubes", q.text).group(1))
        known = [int(lb["text"]) for lb in q.figure["labels"] if lb["text"] != "?"]
        bad = int(canon["a"]) + 1
        mx, n = max(known), len(known) + 1
        if bad > mx:
            return None
        add = mx * n - (sum(known) + bad)
        return {"a": str(bad), "b": str(add), "c": str(total + add)}

    assert sup.check_follow_through(sup.skill_of(MODEL[0]), MODEL[2], "b", wrong) >= 20


def test_plan_views_chain():
    skill = _wiring(VIEWS)
    sup.check_exam_chain(skill, VIEWS[2], [1, 2, 2, 1], ["number", "mcq", "mcq", "number"])
    for seed in range(80):
        q = sup.exam(VIEWS[2], seed)
        w, d = q.figure["grid"]["width"], q.figure["grid"]["height"]
        h = {(int(lb["at"][0]), int(lb["at"][1])): int(lb["text"]) for lb in q.figure["labels"]}
        get = lambda x, y: h.get((x, y), 0)  # noqa: E731
        front = [max(get(x, y) for y in range(d)) for x in range(w)]
        right = [max(get(x, y) for x in range(w)) for y in range(d)]
        line = lambda hs: "Column heights, left to right: " + ", ".join(map(str, hs))  # noqa: E731
        want = [str(sum(h.values())), line(front), line(right), str(sum(front))]
        assert [p.answer_text for p in q.parts] == want, seed
        # no row or column of the plan is empty, so every column of each view is drawn
        assert all(max(get(x, y) for y in range(d)) for x in range(w))
        assert len(set(q.parts[1].distractors) | {q.parts[1].answer_text}) == 4


def _spin(cells, p, quarters):
    out = set()
    for x, y in cells:
        cx, cy = x + 0.5 - p[0], y + 0.5 - p[1]
        for _ in range(quarters % 4):
            cx, cy = -cy, cx
        out.add((int(p[0] + cx - 0.5), int(p[1] + cy - 0.5)))
    return out


def test_turn_pictures_chain():
    skill = _wiring(TURNS)
    sup.check_exam_chain(skill, TURNS[2], [1, 1, 2], ["mcq"] * 3)
    asked = set()
    for seed in range(80):
        q = sup.exam(TURNS[2], seed)
        fig = q.figure
        pw, ph = fig["grid"]["width"] // 2, fig["grid"]["height"] // 2
        shaded = {tuple(c) for c in fig["cells"][0]["cells"]}
        hatched = {tuple(c) for c in fig["cells"][1]["cells"]}
        letters = {lb["text"]: lb["at"] for lb in fig["labels"] if lb["text"] in ("A", "B", "C", "D")}
        assert set(letters) == {"A", "B", "C", "D"}
        kind_of = {}
        tiles = []
        for letter, (lx, ly) in letters.items():
            ox, oy = int(lx - 0.5), int(ly + 0.5 - ph)
            region = lambda c: ox <= c[0] < ox + pw and oy <= c[1] < oy + ph  # noqa: E731
            (p,) = [lb["at"] for lb in fig["labels"] if lb["text"] == "P" and region(lb["at"])]
            tile = {c for c in shaded if region(c)}
            image = {c for c in hatched if region(c)}
            assert tile and image and not tile & image
            tiles.append(frozenset((x - ox, y - oy) for x, y in tile))
            matches = [
                name for name, quarters in (("cw", 3), ("ccw", 1), ("half", 2)) if _spin(tile, p, quarters) == image
            ]
            assert len(matches) <= 1
            kind_of[letter] = matches[0] if matches else "other"
        assert len(set(tiles)) == 1  # the same shaded shape in every picture
        assert sorted(kind_of.values()) == ["ccw", "cw", "half", "other"]
        letter_for = {kind: f"Picture {letter}" for letter, kind in kind_of.items()}
        names = {"a quarter turn clockwise": "cw", "a quarter turn anticlockwise": "ccw", "a half turn": "half"}
        compose = {
            frozenset(("cw", "half")): "ccw",
            frozenset(("ccw", "half")): "cw",
            frozenset(("cw", "cw")): "half",
            frozenset(("ccw", "ccw")): "half",
        }
        a = names[
            re.search(
                r"after (a quarter turn clockwise|a quarter turn anticlockwise|a half turn) about P\?",
                q.parts[0].prompt,
            ).group(1)
        ]
        b = names[
            re.search(
                r"after (a quarter turn clockwise|a quarter turn anticlockwise|a half turn) about P\?",
                q.parts[1].prompt,
            ).group(1)
        ]
        m = re.search(r"turned (.+?) about P and then (.+?) about P", q.parts[2].prompt)
        c = (
            compose[frozenset((names[m.group(1)], names[m.group(2)]))]
            if m.group(1) != m.group(2)
            else compose[frozenset((names[m.group(1)],) * 2)]
        )
        assert len({a, b, c}) == 3  # each part asks about a different picture
        assert [p.answer_text for p in q.parts] == [letter_for[a], letter_for[b], letter_for[c]], seed
        asked.add((a, b, c))
    assert len(asked) >= 4


def test_box_net_chain():
    skill = _wiring(BOX)
    sup.check_exam_chain(skill, BOX[2], [1, 1, 1, 2], ["number"] * 4)
    for seed in range(80):
        q = sup.exam(BOX[2], seed)
        cells = {tuple(c) for c in q.figure["cells"][0]["cells"]}
        rows = Counter(y for _, y in cells)
        widest = max(rows.values())
        full_rows = [y for y, n in rows.items() if n == widest]
        height = len(full_rows)
        assert sorted(full_rows) == list(range(min(full_rows), min(full_rows) + height))  # one block of full rows
        extra = len(cells) - widest * height
        assert extra % 2 == 0 and extra > 0
        top = extra // 2
        # the top is joined above the block, the bottom below it
        above = {c for c in cells if c[1] > max(full_rows)}
        below = {c for c in cells if c[1] < min(full_rows)}
        assert len(above) == len(below) == top
        assert [int(p.answer_text) for p in q.parts] == [len(cells), top, height, top * height], seed
        a_side = widest // 2
        assert widest % 2 == 0 and a_side >= 2  # the row of four side faces is the whole perimeter of the base
        assert len(cells) == 2 * top + widest * height


def test_box_follow_through():
    def wrong(q, canon):
        bad = int(canon["b"]) + 3
        h = int(canon["c"])
        return {"b": str(bad), "d": str(bad * h)}

    assert sup.check_follow_through(sup.skill_of(BOX[0]), BOX[2], "d", wrong) >= 30


def _scale_cm(text):
    m = re.fullmatch(r"1 cm : (\d+) (cm|m)|1 : (\d+)", text)
    return int(m.group(3)) if m.group(3) else int(m.group(1)) * (100 if m.group(2) == "m" else 1)


def test_scale_plan_chain():
    skill = _wiring(PLAN)
    sup.check_exam_chain(skill, PLAN[2], [1, 1, 2, 2], ["number"] * 4)
    scales = set()
    for seed in range(80):
        q = sup.exam(PLAN[2], seed)
        per_cm = _scale_cm(q.figure["scale"]["text"])
        scales.add(q.figure["scale"]["text"])
        nums = [
            Fraction(x)
            for x in re.findall(r"(\d+(?:\.\d+)?) m across the page and (\d+(?:\.\d+)?) m up", q.text)
            for x in x
        ]
        wa, wb, wc, wd = (n * 100 / per_cm for n in nums)
        assert all(v == int(v) for v in (wa, wb, wc, wd))
        a, b, c, d = (int(v) for v in (wa, wb, wc, wd))
        assert 1 <= c < a and 1 <= d < b
        assert a <= q.figure["grid"]["width"] and b <= q.figure["grid"]["height"]
        assert [int(p.answer_text) for p in q.parts] == [a, b, a * b - c * d, 2 * (a + b)], seed
        # an L-shape (a corner cut out of a rectangle) has the same perimeter as the rectangle
        assert 2 * (a + b) == 2 * (a - c + b) + 2 * c
    assert len(scales) >= 5


def test_scale_plan_follow_through():
    def wrong(q, canon):
        a, b = int(canon["a"]) + 1, int(canon["b"])
        cd = int(canon["a"]) * int(canon["b"]) - int(canon["c"])
        return {"a": str(a), "c": str(a * b - cd), "d": str(2 * (a + b))}

    assert sup.check_follow_through(sup.skill_of(PLAN[0]), PLAN[2], "c", wrong) >= 30


def test_variety():
    for spec in (MODEL, VIEWS, TURNS, BOX, PLAN):
        counts = Counter(sup.exam(spec[2], s).answer_text for s in range(60))
        assert len(counts) >= (12 if spec is TURNS else 25), (spec[2], len(counts))
        assert max(counts.values()) <= 8
