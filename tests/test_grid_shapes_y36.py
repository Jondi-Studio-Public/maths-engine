"""Years 3 to 6 shapes and one senior row, drawn on a grid: ``grid_shapes_y36_build``.

Four skills, every figure a ``grid_drawing``, every tier generated through the skill's own sources so what is tested
is what a learner is served. The answers are re-derived here by routes that share no code with the generators: a
turn by multiplying a cell's centre by a power of i, a box net by its faces' areas and where its faces meet, a
latitude or longitude by the position of a place against the labelled lines.
"""

from __future__ import annotations

import functools
import re

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.core import Part
from mathgen.grading import grade
from mathgen.grading.grid import cell_text
from mathgen.skills import load_skills

SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

# skill id -> (topic, answer kind, tiers, learner band, ACARA or senior code tag)
SKILLS = {
    "f10-model-plans-cube-counts": ("grid_model_plans", "number", TIERS, "primary", "ac9m3sp01"),
    "f10-turn-shapes-on-a-grid": ("grid_shape_turns", "grid_cells", TIERS, "primary", "ac9m5sp02"),
    "f10-box-nets-on-a-grid": ("grid_prism_nets", "grid_cells", TIERS, "primary", "ac9m6sp01"),
    "senior-latitude-longitude-grid-places": (
        "grid_latlong_places",
        "mcq",
        [*TIERS, "exam"],
        "general",
        "acmem159",
    ),
}
LATLONG = "senior-latitude-longitude-grid-places"
CASES = [(sid, tier) for sid, (_t, _k, tiers, _b, _c) in SKILLS.items() for tier in tiers]


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


@functools.lru_cache(maxsize=None)
def _q(skill_id, tier, seed):
    src = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _topic(skill_id, tier):
    return next(s for s in _skills()[skill_id].sources if s.difficulty == tier).topic


def _kind(q, skill_id, tier):
    """The grader kind a learner's reply to this question goes to."""
    return q.answer_kind or SKILLS[skill_id][1]


def _cells(q):
    return {tuple(c) for c in q.answer["cells"]}


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", SKILLS)
def test_each_skill_is_active_behind_the_figure_gate_with_the_tiers_its_year_needs(skill_id):
    topic, answer, tiers, band, code = SKILLS[skill_id]
    skill = _skills()[skill_id]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.answer == answer
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False
    assert skill.learner_band == band
    assert code in skill.tags
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    # the senior row always had an exam tier; the Years 3 to 6 rows gained one from later exam chains
    assert [s.difficulty for s in skill.sources] in (tiers, [*tiers, "exam"])
    assert len(skill.misconceptions) >= 5 and all(m.description for m in skill.misconceptions)
    assert skill.prerequisites and all(_skills()[p].status == "active" for p in skill.prerequisites)


def test_the_skills_do_not_take_over_a_deferred_or_existing_row():
    existing = {"f10-two-dimensional-object-representations", "f10-nets-build-solids", "f10-prism-cross-sections"}
    for skill_id in SKILLS:
        assert skill_id not in existing
        assert not set(_skills()[skill_id].prerequisites) & {"f10-two-dimensional-object-representations"}


# ------------------------------------------------------------------ every figure, every tier


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_figure_validates_does_not_leak_and_the_reader_agrees_with_the_key(skill_id, tier):
    topic = _topic(skill_id, tier)
    reader = figures.get_reader(topic)
    assert reader is not None
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        assert q.figure["kind"] == "grid_drawing"
        assert figures.validate(q.figure) == [], (seed, figures.validate(q.figure))
        answers = [q.answer_text, *(p.answer_text for p in q.parts or [])]
        figures.assert_no_leak(q.figure, answers)
        assert not any(ch.isdigit() for ch in q.figure["alt"]), q.figure["alt"]
        read = reader(q)
        if isinstance(q.answer, dict) and q.answer.get("match", "exact") != "exact":
            assert grade("grid_cells", q, read).correct, (skill_id, tier, seed, q.text, read)
        else:
            assert read == q.answer_text, (skill_id, tier, seed, q.text)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_the_key_grades_correct_and_every_distractor_wrong_through_grade(skill_id, tier):
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        if q.parts:
            assert all(isinstance(p, Part) for p in q.parts)
            assert grade("number", q, q.answer_text).correct
            for part in q.parts:
                assert grade(part.answer_kind, part, part.answer_text).correct, (seed, part.label)
                assert len(part.distractors) == 3 and len(set(part.distractors)) == 3
                for d in part.distractors:
                    assert not grade(part.answer_kind, part, d).correct, (seed, part.label, d, part.answer_text)
            continue
        kind = _kind(q, skill_id, tier)
        assert grade(kind, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        for d in q.distractors:
            result = grade(kind, q, d)
            assert not result.correct, (skill_id, tier, seed, d, q.answer_text)
            if kind == "grid_cells":
                assert result.given_normalised is not None  # read as a drawing, then judged wrong
                w, h = q.answer["grid"]
                cells = {tuple(map(int, m)) for m in re.findall(r"\((\d+), (\d+)\)", d)}
                assert cells and all(0 <= x < w and 0 <= y < h for x, y in cells), (seed, d)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_deterministic_varied_and_short_enough_to_read(skill_id, tier):
    seen = set()
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        src = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
        again = make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
        assert q.to_json() == again.to_json()
        seen.add((q.text, q.answer_text, repr(q.figure)))
        assert q.text and "\\(" not in q.text
        assert "(a)" not in q.text or tier == "exam"
    assert len(seen) >= 40, (skill_id, tier, len(seen))


@pytest.mark.parametrize("skill_id", [s for s, v in SKILLS.items() if v[1] == "grid_cells"])
def test_the_grid_answer_sits_on_the_grid_the_learner_shades(skill_id):
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            grid = q.figure["grid"]
            assert q.figure["canvas"] == "grid" and q.answer["grid"] == [grid["width"], grid["height"]]
            assert all(0 <= x < grid["width"] and 0 <= y < grid["height"] for x, y in _cells(q))
            given = {tuple(c) for g in q.figure.get("cells", []) for c in g["cells"]}
            assert not given & _cells(q), "the answer must not repeat a square that is already drawn"
            assert "plain:" in q.answer_latex


# ------------------------------------------------------------------ Year 3: the cube model


def _plan(q):
    return {(int(lb["at"][0]), int(lb["at"][1])): lb["text"] for lb in q.figure["labels"]}


def test_the_model_totals_are_the_sum_of_the_stacks_the_plan_shows():
    kinds = set()
    for tier in TIERS:
        for seed in SEEDS:
            q = _q("f10-model-plans-cube-counts", tier, seed)
            plan = _plan(q)
            shaded = {tuple(c) for g in q.figure["cells"] for c in g["cells"]}
            assert shaded == set(plan), "every stack is a shaded square with its number, and bare squares are blank"
            known = [int(v) for v in plan.values() if v != "?"]
            if "?" in plan.values():
                kinds.add("missing")
                total = int(re.search(r"(\d+) cubes", q.text).group(1))
                assert q.answer == total - sum(known) >= 1
                assert tier == "hard" and len(known) >= 3
            elif "more cubes" in q.text or "cubes are added" in q.text:
                kinds.add("fill")
                assert q.answer == sum(max(known) - v for v in known) and tier == "medium"
            else:
                kinds.add("total")
                assert q.answer == sum(known)
            if tier == "easy":
                w, h = q.figure["grid"]["width"], q.figure["grid"]["height"]
                assert len(plan) == w * h, "an easy plan has no bare squares"
            if tier == "medium":
                assert len(plan) < q.figure["grid"]["width"] * q.figure["grid"]["height"]
    assert kinds == {"total", "fill", "missing"}


def test_the_cube_model_slips_are_the_ones_the_skill_names():
    seen = set()
    for tier in TIERS:
        for seed in SEEDS:
            q = _q("f10-model-plans-cube-counts", tier, seed)
            plan = _plan(q)
            known = [int(v) for v in plan.values() if v != "?"]
            wrong = {int(d) for d in q.distractors}
            if len(plan) in wrong and tier != "hard":
                seen.add("squares")
            if max(known) * len(plan) in wrong:
                seen.add("tallest-times-squares")
    assert seen == {"squares", "tallest-times-squares"}


# ------------------------------------------------------------------ Year 5: turns, by complex multiplication


def _turned(cells, p, quarters):
    """Each cell's centre, as a complex number about the point p, times i to the number of quarter turns."""
    out = set()
    for x, y in cells:
        z = complex(x + 0.5 - p[0], y + 0.5 - p[1]) * (1j**quarters)
        out.add((round(p[0] + z.real - 0.5), round(p[1] + z.imag - 0.5)))
    return out


def _point_p(q):
    lab = next(lb for lb in q.figure["labels"] if lb["text"] == "P")
    return int(lab["at"][0]), int(lab["at"][1])


def test_a_half_turn_and_a_quarter_turn_are_what_the_wording_says():
    seen = set()
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = _q("f10-turn-shapes-on-a-grid", tier, seed)
            tile = {tuple(c) for c in q.figure["cells"][0]["cells"]}
            quarters = 2 if "half turn" in q.text else 3 if "clockwise" in q.text and "anti" not in q.text else 1
            seen.add((tier, quarters))
            assert _cells(q) == _turned(tile, _point_p(q), quarters)
            assert len(_cells(q)) == len(tile) and _cells(q) != tile
    assert seen == {("easy", 2), ("medium", 1), ("medium", 3)}


def test_the_repeated_move_is_read_from_the_picture_and_done_again():
    kinds = set()
    for seed in SEEDS:
        q = _q("f10-turn-shapes-on-a-grid", "hard", seed)
        tile, first = ({tuple(c) for c in g["cells"]} for g in q.figure["cells"])
        assert [g["style"] for g in q.figure["cells"]] == ["shaded", "hatched"]
        if any(lb["text"] == "P" for lb in q.figure.get("labels", [])):
            kinds.add("turn")
            p = _point_p(q)
            which = [k for k in (1, 3) if _turned(tile, p, k) == first]
            assert len(which) == 1
            assert _cells(q) == _turned(first, p, which[0]) == _turned(tile, p, 2 * which[0])
        else:
            kinds.add("slide")
            dx = min(x for x, _ in first) - min(x for x, _ in tile)
            dy = min(y for _, y in first) - min(y for _, y in tile)
            assert {(x + dx, y + dy) for x, y in tile} == first
            assert _cells(q) == {(x + 2 * dx, y + 2 * dy) for x, y in tile}
    assert kinds == {"turn", "slide"}


def test_a_turn_can_be_told_from_a_slide_by_the_shape_alone():
    """Every shape drawn has all eight turns and flips different, so one picture fits one kind of move."""
    from mathgen.grading.grid import orientations

    for tier in TIERS:
        for seed in SEEDS:
            q = _q("f10-turn-shapes-on-a-grid", tier, seed)
            tile = {tuple(c) for c in q.figure["cells"][0]["cells"]}
            assert len(orientations(tile)) == 8


def test_the_turn_slips_are_real_ones():
    seen = set()
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = _q("f10-turn-shapes-on-a-grid", tier, seed)
            tile = {tuple(c) for c in q.figure["cells"][0]["cells"]}
            p = _point_p(q)
            quarters = 2 if "half turn" in q.text else 3 if "clockwise" in q.text and "anti" not in q.text else 1
            wrong = set(q.distractors)
            if cell_text(_turned(tile, p, 4 - quarters if quarters != 2 else 2) | set()) in wrong:
                seen.add("other way")
            for other in (1, 2, 3):
                if other != quarters and cell_text(_turned(tile, p, other)) in wrong:
                    seen.add("other size")
            if any(
                cell_text(_turned(tile, (p[0] + dx, p[1] + dy), quarters)) in wrong
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            ):
                seen.add("off-centre")
    assert {"other size", "off-centre"} <= seen


# ------------------------------------------------------------------ Year 6: box nets, by faces and edges

_WIDE = re.compile(r"base (\d+) squares? by (\d+) squares? and is (\d+) squares? tall")


def _rows(cells):
    """The columns of each row of a net, by row."""
    rows: dict[int, set[int]] = {}
    for x, y in cells:
        rows.setdefault(y, set()).add(x)
    return rows


def test_a_blank_net_has_the_faces_of_the_box_and_every_join_fits():
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = _q("f10-box-nets-on-a-grid", tier, seed)
            a, b, h = (int(v) for v in _WIDE.search(q.text).groups())
            net = _cells(q)
            assert len(net) == 2 * (a * b + a * h + b * h)  # the surface area of the box
            rows = _rows(net)
            long_rows = sorted(y for y, xs in rows.items() if len(xs) == 2 * (a + b))
            assert len(long_rows) == h and long_rows == list(range(long_rows[0], long_rows[0] + h))
            top = [y for y in rows if y > long_rows[-1]]
            bottom = [y for y in rows if y < long_rows[0]]
            # the top and the bottom are each a rectangle as wide as the side face above or below which it hangs
            for part, edge_row in ((top, long_rows[-1]), (bottom, long_rows[0])):
                assert part and len(part) == len(set(part))
                width = len(rows[part[0]])
                assert all(rows[y] == rows[part[0]] for y in part)
                assert width * len(part) == a * b and width in (a, b)
                assert len(part) == (a * b) // width and rows[part[0]] <= rows[edge_row]
                widths = [a, b, a, b]
                starts = [sum(widths[:i]) + min(rows[edge_row]) for i in range(4)]
                assert min(rows[part[0]]) in starts and widths[starts.index(min(rows[part[0]]))] == width
            if tier == "easy":  # both are joined to the first face, so they start at the same column
                assert len({min(rows[y]) for y in top + bottom}) == 1


def test_the_missing_bottom_is_as_wide_as_the_face_it_joins_and_makes_the_base_a_by_b():
    seen_other_width = False
    for seed in SEEDS:
        q = _q("f10-box-nets-on-a-grid", "hard", seed)
        given = {tuple(c) for c in q.figure["cells"][0]["cells"]}
        n = int(re.search(r"face (\d)", q.text).group(1))
        labels = {lb["text"]: lb["at"] for lb in q.figure["labels"]}
        centre_x = labels[f"Face {n}"][0]
        rows = _rows(given)
        perimeter = max(len(xs) for xs in rows.values())
        strip = [y for y, xs in rows.items() if len(xs) == perimeter]
        top_rows = [y for y in rows if y > max(strip)]
        top_w, top_h = len(rows[top_rows[0]]), len(top_rows)
        base_sides = {top_w, top_h}
        bottom = _cells(q)
        xs = sorted({x for x, _ in bottom})
        ys = sorted({y for _, y in bottom})
        width, height = len(xs), len(ys)
        assert len(bottom) == width * height == top_w * top_h and {width, height} == base_sides
        assert xs[0] <= centre_x <= xs[-1] + 1 and max(ys) == min(strip) - 1  # hangs from the lower edge of the strip
        seen_other_width |= width != top_w
    assert seen_other_width, (
        "some bottoms must be turned from the top's size, because the face they join is the other width"
    )


def test_the_net_slips_include_a_face_left_out_and_a_face_the_wrong_size_for_its_edge():
    seen = set()
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = _q("f10-box-nets-on-a-grid", tier, seed)
            full = len(_cells(q))
            sizes = {len(re.findall(r"\(", d)) for d in q.distractors}
            if any(s < full for s in sizes):
                seen.add("left out")
            if any(s == full for s in sizes):
                seen.add("same area, wrong layout")
    assert seen == {"left out", "same area, wrong layout"}


# ------------------------------------------------------------------ senior: latitude and longitude, from the lines


def _lines(q):
    """Degrees for each labelled line: {x: longitude} along the top, {y: latitude} up the left."""
    lat, lon = {}, {}
    for lb in q.figure["labels"]:
        if len(lb["text"]) == 1:
            continue
        n = int(lb["text"].split("°")[0])
        n = -n if lb["text"].endswith(("S", "W")) else n
        (lat if lb["at"][0] == 0 else lon)[lb["at"][1] if lb["at"][0] == 0 else lb["at"][0]] = n
    return lat, lon


def _places(q):
    lat, lon = _lines(q)
    step_y = lat[1] - lat[0]
    x0 = next(iter(lon))
    return {
        lb["text"]: (lat[0] + step_y * lb["at"][1], lon[x0] + 10 * (lb["at"][0] - x0))
        for lb in q.figure["labels"]
        if len(lb["text"]) == 1
    }


def _deg(v, plus, minus):
    return "0°" if v == 0 else f"{abs(v)}°{plus if v > 0 else minus}"


def test_the_lines_are_ten_degrees_apart_and_labelled_up_the_side_and_along_the_top():
    for tier in [*TIERS, "exam"]:
        for seed in SEEDS:
            q = _q(LATLONG, tier, seed)
            lat, lon = _lines(q)
            h = q.figure["grid"]["height"]
            assert sorted(lat) == list(range(h + 1))
            assert all(lat[y + 1] - lat[y] == 10 for y in range(h))
            xs = sorted(lon)
            assert all(lon[b] - lon[a] == 10 * (b - a) for a, b in zip(xs, xs[1:]))
            assert len(_places(q)) >= 2


def test_easy_reads_latitude_then_longitude_and_hard_adds_across_the_line():
    for seed in SEEDS:
        q = _q(LATLONG, "easy", seed)
        letter = re.search(r"place ([ABC])", q.text).group(1)
        la, lo = _places(q)[letter]
        assert q.answer_text == f"{_deg(la, 'N', 'S')}, {_deg(lo, 'E', 'W')}"
        assert grade("mcq", q, q.answer_text).correct
    seen = set()
    for seed in SEEDS:
        q = _q(LATLONG, "hard", seed)
        (la1, lo1), (la2, lo2) = _places(q)["A"], _places(q)["B"]
        assert la1 * la2 < 0 and lo1 * lo2 < 0, "the two places are on opposite sides of both lines"
        on_lon = "longitude" in q.text
        seen.add(on_lon)
        want = abs(lo1 - lo2) if on_lon else abs(la1 - la2)
        assert q.answer_text == f"{want}°" and want == (abs(lo1) + abs(lo2) if on_lon else abs(la1) + abs(la2)) >= 70
        assert grade("mcq", q, q.answer_text).correct and not any(grade("mcq", q, d).correct for d in q.distractors)


def test_medium_names_the_one_place_at_the_coordinates_and_the_others_are_slips():
    slips = set()
    for seed in SEEDS:
        q = _q(LATLONG, "medium", seed)
        places = _places(q)
        m = re.search(r"(\d+°[NS]) (?:and longitude|,)? ?(\d+°[EW])|(\d+°[NS]), (\d+°[EW])", q.text)
        la_t, lo_t = [g for g in m.groups() if g]
        want = [t for t, (la, lo) in places.items() if _deg(la, "N", "S") == la_t and _deg(lo, "E", "W") == lo_t]
        assert len(want) == 1 and q.answer_text == f"Place {want[0]}"
        (la, lo) = places[want[0]]
        for t, (a, b) in places.items():
            if t == want[0]:
                continue
            assert (a, b) in {(-la, lo), (la, -lo), (lo, la)}
            slips.add("hemisphere" if abs(a) == abs(la) and abs(b) == abs(lo) else "swapped")
        assert sorted(d for d in q.distractors) == sorted(f"Place {t}" for t in places if t != want[0])
    assert slips == {"hemisphere", "swapped"}


def test_the_exam_question_is_the_distance_along_a_meridian_from_the_latitudes():
    ways = set()
    for seed in SEEDS:
        q = _q(LATLONG, "exam", seed)
        assert not q.parts and q.answer_kind == "mcq"
        (la1, lo1), (la2, lo2) = _places(q)["A"], _places(q)["B"]
        assert q.answer_text == f"{111 * (abs(la1) + abs(la2))} km"
        ways.add(re.search(r"due (north|south)", q.text).group(1))
        assert ("south" if la1 > la2 else "north") == re.search(r"due (north|south)", q.text).group(1)
        wrong = {int(d.split()[0]) for d in q.distractors}
        assert all(w % 111 == 0 for w in wrong), "every wrong distance is a whole number of 111 km steps"
    assert ways == {"north", "south"}


def test_the_latlong_slips_are_the_ones_the_skill_names():
    seen = set()
    for seed in SEEDS:
        q = _q(LATLONG, "easy", seed)
        letter = re.search(r"place ([ABC])", q.text).group(1)
        la, lo = _places(q)[letter]
        if f"{_deg(lo, 'E', 'W')}, {_deg(la, 'N', 'S')}" in q.distractors:
            seen.add("longitude first")
        if f"{_deg(la + 10, 'N', 'S')}, {_deg(lo, 'E', 'W')}" in q.distractors:
            seen.add("one line out")
        if any(d.endswith(("W", "S")) != q.answer_text.endswith(("W", "S")) for d in q.distractors):
            seen.add("other side")
    assert seen == {"longitude first", "one line out", "other side"}
    hard = set()
    for seed in SEEDS:
        q = _q(LATLONG, "hard", seed)
        (la1, lo1), (la2, lo2) = _places(q)["A"], _places(q)["B"]
        on_lon = "longitude" in q.text
        a, b = (lo1, lo2) if on_lon else (la1, la2)
        wrong = {int(d.rstrip("°")) for d in q.distractors}
        if abs(abs(a) - abs(b)) in wrong:
            hard.add("subtracts")
        if abs(a - b) // 10 in wrong:
            hard.add("squares")
    assert hard == {"subtracts", "squares"}
