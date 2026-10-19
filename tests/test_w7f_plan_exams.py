"""Exam chains for the floor-plan, site-plan and map-landmark skills (the w7f work).

Oracles read the rooms, symbols, dashed outlines, dimension labels and landmarks off the FIGURE and the numbers off
the prompts, with arithmetic of their own; none calls the generator's helpers.
"""

from __future__ import annotations

import re
from collections import Counter

import mathgen.generators.w7f_plan_exams  # noqa: F401  (registers the topics)
from tests import w7b_support as sup

SYMBOLS = (
    "senior-interpret-commonly-used-symbols-and-abbreviations-in-scale",
    "grid_plan_symbols",
    "exd_x_w7f_plan_symbols_chain",
)
CONVENTIONS = (
    "senior-use-symbols-and-conventions-for-the-representation-of",
    "grid_plan_conventions",
    "exd_x_w7f_plan_conventions_chain",
)
LANDMARKS = ("f10-map-feature-and-landmark-representation", "grid_map_landmarks", "exd_x_w7f_map_landmarks_chain")

ROOMS = {
    "KIT": "Kitchen", "LIV": "Living room", "DIN": "Dining room", "BATH": "Bathroom", "ENS": "Ensuite",
    "WC": "Toilet", "L'DRY": "Laundry", "GAR": "Garage", "WIR": "Walk-in robe", "PAN": "Pantry", "STU": "Study",
}  # fmt: skip
FEATURES = {
    "HSE": "House", "GAR": "Garage", "SH": "Shed", "PERG": "Pergola", "DK": "Deck", "GH": "Greenhouse",
    "PL": "Pool", "CP": "Carport",
}  # fmt: skip


def _full_room(abbr):
    return f"Bedroom {abbr[-1]}" if abbr.startswith("BR ") else ROOMS[abbr]


def _wiring(spec, answer):
    skill = sup.skill_of(spec[0])
    sup.check_wiring(skill, spec[1], spec[2], answer)
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and steps[-1] > steps[0]
    return skill


def _in(box, at):
    return box[0] <= at[0] <= box[2] and box[1] <= at[1] <= box[3]


def _boxes(fig):
    out = []
    for poly in fig["polygons"]:
        xs, ys = [v[0] for v in poly["vertices"]], [v[1] for v in poly["vertices"]]
        out.append((min(xs), min(ys), max(xs), max(ys), poly))
    return out


def test_symbols_wiring_and_shape():
    skill = _wiring(SYMBOLS, "mcq")
    sup.check_exam_chain(skill, SYMBOLS[2], [1, 1, 2, 2, 1], ["mcq", "mcq", "number", "number", "number"])


def test_symbols_keys_from_the_picture():
    seen_counts = Counter()
    for seed in range(80):
        q = sup.exam(SYMBOLS[2], seed)
        fig = q.figure
        rooms = {}
        for x0, y0, x1, y1, _ in _boxes(fig):
            names = [
                lb["text"]
                for lb in fig["labels"]
                if (lb["text"] in ROOMS or lb["text"].startswith("BR ")) and _in((x0, y0, x1, y1), lb["at"])
            ]
            assert len(names) == 1
            rooms[names[0]] = (x0, y0, x1, y1)
        assert len(rooms) == 8

        def with_symbol(kind):
            return {a for a, b in rooms.items() if any(s["symbol"] == kind and _in(b, s["at"]) for s in fig["symbols"])}

        def home(code):
            at = [lb["at"] for lb in fig["labels"] if lb["text"] == code]
            assert len(at) == 1
            (room,) = [a for a, b in rooms.items() if _in(b, at[0])]
            return room

        toilets, windows = with_symbol("toilets"), with_symbol("window")
        (lone,) = toilets - windows
        visits = toilets | {home("HWS"), home("DW")}
        price = int(re.search(r"\$(\d+) for each", q.parts[4].prompt).group(1))
        got = [p.answer_text for p in q.parts]
        assert got == [
            _full_room(lone),
            _full_room(home("HWS")),
            str(len(windows)),
            str(len(visits)),
            str(len(visits) * price),
        ], seed
        seen_counts[len(visits)] += 1
        # exactly one toilets room lacks a window: the question has one answer
        assert len(toilets - windows) == 1
    assert set(seen_counts) == {3, 4}


def test_symbols_follow_through_from_a_wrong_count():
    def wrong(q, canon):
        price = int(re.search(r"\$(\d+) for each", q.parts[4].prompt).group(1))
        bad = int(canon["d"]) + 1
        return {"d": str(bad), "e": str(bad * price)}

    assert sup.check_follow_through(sup.skill_of(SYMBOLS[0]), SYMBOLS[2], "e", wrong) >= 30


def test_conventions_wiring_and_shape():
    skill = _wiring(CONVENTIONS, "mcq")
    sup.check_exam_chain(skill, CONVENTIONS[2], [1, 1, 2, 2], ["mcq", "mcq", "number", "number"])


def test_conventions_keys_from_the_picture():
    directions = set()
    for seed in range(80):
        q = sup.exam(CONVENTIONS[2], seed)
        fig = q.figure
        feats = []
        for x0, y0, x1, y1, poly in _boxes(fig):
            (name,) = [lb["text"] for lb in fig["labels"] if lb["text"] in FEATURES and _in((x0, y0, x1, y1), lb["at"])]
            (w,) = [lb["text"] for lb in fig["labels"] if lb["at"] == [(x0 + x1) / 2, y1 + 0.5]]
            (h,) = [lb["text"] for lb in fig["labels"] if lb["at"] == [x1 + 1, (y0 + y1) / 2]]
            feats.append((name, poly["style"], (x0 + x1, y0 + y1), int(w[:-2]) * int(h[:-2])))
        assert len(feats) == 4
        (prop,) = [f for f in feats if f[1] == "dashed"]
        start_name = re.search(r"From the (.+?),", q.parts[1].prompt).group(1)
        (start,) = [f for f in feats if FEATURES[f[0]].lower() == start_name]
        dx, dy = prop[2][0] - start[2][0], prop[2][1] - start[2][1]
        assert dx == 0 or dy == 0 or abs(dx) == abs(dy)
        sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
        compass = {(0, 1): "north", (1, 1): "north-east", (1, 0): "east", (1, -1): "south-east", (0, -1): "south",
                   (-1, -1): "south-west", (-1, 0): "west", (-1, 1): "north-west"}  # fmt: skip
        directions.add(compass[(sx, sy)])
        got = [p.answer_text for p in q.parts]
        assert got == [FEATURES[prop[0]], compass[(sx, sy)], str(prop[3]), str(sum(f[3] for f in feats))], seed
        # no two outlines closer than the labels need
        for i, a in enumerate(_boxes(fig)):
            for b in _boxes(fig)[i + 1 :]:
                assert a[2] + 2 <= b[0] or b[2] + 2 <= a[0] or a[3] + 2 <= b[1] or b[3] + 2 <= a[1]
    assert len(directions) >= 6


def test_conventions_follow_through_from_a_wrong_area():
    def wrong(q, canon):
        bad = int(canon["c"]) + 10
        total = int(canon["d"]) - int(canon["c"]) + bad
        return {"c": str(bad), "d": str(total)}

    assert sup.check_follow_through(sup.skill_of(CONVENTIONS[0]), CONVENTIONS[2], "d", wrong) >= 30


def test_landmarks_wiring_and_shape():
    skill = _wiring(LANDMARKS, "mcq")
    sup.check_exam_chain(skill, LANDMARKS[2], [1, 1, 2, 2], ["mcq"] * 4)


def test_landmarks_keys_from_the_picture():
    names = {"house", "school", "church", "hospital", "tree", "bridge", "airport", "lighthouse", "railway_station"}
    for seed in range(80):
        q = sup.exam(LANDMARKS[2], seed)
        fig = q.figure
        cols, rows = fig["column_labels"], fig["row_labels"]
        at = {(int(s["at"][0]), int(s["at"][1])): s["symbol"] for s in fig["symbols"]}
        assert set(at.values()) <= names and len(set(at.values())) == len(at)
        pretty = lambda sym: sym.replace("_", " ").capitalize()  # noqa: E731
        sq = re.search(r"square ([A-Z]\d)\?", q.parts[0].prompt).group(1)
        a = pretty(at[(cols.index(sq[0]), rows.index(sq[1]))])
        nm = re.search(r"reference of the (.+?)\?", q.parts[1].prompt).group(1).replace(" ", "_")
        (cell,) = [c for c, s in at.items() if s == nm]
        b = cols[cell[0]] + rows[cell[1]]
        m = re.match(r"Start at the (.+?)\. Go (\d+) squares? (\w+), then (\d+) squares? (\w+)\.", q.parts[2].prompt)
        (x, y) = next(c for c, s in at.items() if s == m.group(1).replace(" ", "_"))
        for n, way in ((m.group(2), m.group(3)), (m.group(4), m.group(5))):
            dx, dy = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}[way]
            x, y = x + int(n) * dx, y + int(n) * dy
        assert 0 <= x < len(cols) and 0 <= y < len(rows)
        assert [p.answer_text for p in q.parts] == [a, b, cols[x] + rows[y], pretty(at[(x, y)])], seed


def test_variety():
    for spec in (SYMBOLS, CONVENTIONS, LANDMARKS):
        counts = Counter(sup.exam(spec[2], s).answer_text for s in range(60))
        assert len(counts) >= 30, (spec[2], len(counts))
        assert len({sup.exam(spec[2], s).text for s in range(60)}) >= 3
