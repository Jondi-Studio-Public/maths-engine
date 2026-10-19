"""Exam chains for the four scale-drawing skills (the w7f work).

Oracles read the vertices, road, labels and scale off the FIGURE and the numbers off the prompts, and recompute every
part with Fraction arithmetic of their own (a shoelace area, a polyline length); none calls the generator's helpers.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import mathgen.generators.w7f_scale_exams  # noqa: F401  (registers the topics)
from tests import w7b_support as sup

ACTUAL = ("senior-find-actual-measurements-from-scale-drawings-such-as", "exd_x_w7f_scale_actual_chain")
MAP = ("senior-obtain-measurements-from-scale-drawings-such-as-maps", "exd_x_w7f_scale_map_chain")
MATERIALS = ("senior-estimate-and-compare-quantities-materials-and-costs-using", "exd_x_w7f_scale_materials_chain")
CONVENTIONS = ("senior-understand-and-apply-drawing-conventions-of-scale-drawings", "exd_x_w7f_scale_conventions_chain")
TOPICS = {
    ACTUAL[0]: "grid_scale_actual",
    MAP[0]: "grid_scale_map",
    MATERIALS[0]: "grid_scale_materials",
    CONVENTIONS[0]: "grid_scale_conventions",
}


def _scale_m_per_cm(fig) -> Fraction:
    text = fig["scale"]["text"]
    if text.startswith("1 :"):
        return Fraction(int(text.split(":")[1])) / 100
    m = re.fullmatch(r"1 cm : (\d+(?:\.\d+)?) (m|km)", text)
    return Fraction(m.group(1)) * (1000 if m.group(2) == "km" else 1)


def _shoelace(v) -> Fraction:
    n = len(v)
    return abs(sum(v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1] for i in range(n))) / 2


def _walls(v):
    n = len(v)
    return [abs(v[(i + 1) % n][0] - v[i][0]) + abs(v[(i + 1) % n][1] - v[i][1]) for i in range(n)]


def _wiring(pair, answer="number"):
    skill = sup.skill_of(pair[0])
    sup.check_wiring(skill, TOPICS[pair[0]], pair[1], answer)
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and steps[-1] > steps[0]
    return skill


def test_wiring_of_all_four():
    for pair in (ACTUAL, MAP, MATERIALS, CONVENTIONS):
        _wiring(pair)


def test_actual_chain_shape_and_keys():
    skill = _wiring(ACTUAL)
    sup.check_exam_chain(skill, ACTUAL[1], [1, 2, 2, 2], ["number"] * 4)
    kinds = set()
    for seed in range(80):
        q = sup.exam(ACTUAL[1], seed)
        verts = q.figure["polygons"][0]["vertices"]
        k = _scale_m_per_cm(q.figure)
        got = [Fraction(p.answer_text) for p in q.parts]
        price = int(re.search(r"\$(\d+)", q.parts[3].prompt).group(1))
        area = _shoelace(verts) * k * k
        assert got == [max(_walls(verts)) * k, sum(_walls(verts)) * k, area, area * price], (seed, got)
        kinds.add(len(verts))
    assert kinds == {6, 8}


def test_actual_follow_through_from_a_wrong_area():
    def wrong(q, canon):
        area = Fraction(canon["c"]) + 7
        price = int(re.search(r"\$(\d+)", q.parts[3].prompt).group(1))
        return {"c": str(area), "d": str(area * price)}

    assert sup.check_follow_through(sup.skill_of(ACTUAL[0]), ACTUAL[1], "d", wrong) >= 30


def test_map_chain_shape_and_keys():
    skill = _wiring(MAP)
    sup.check_exam_chain(skill, MAP[1], [1, 2, 2], ["number"] * 3)
    for seed in range(80):
        q = sup.exam(MAP[1], seed)
        fig = q.figure
        road = next(ln["points"] for ln in fig["lines"] if ln["style"] == "road")
        k = _scale_m_per_cm(fig) / 1000
        cells = sum(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(road, road[1:]))
        direct = abs(road[-1][0] - road[0][0]) + abs(road[-1][1] - road[0][1])
        speed = int(re.search(r"(\d+) km/h", q.parts[1].prompt).group(1))
        dist = cells * k
        assert [Fraction(p.answer_text) for p in q.parts] == [dist, dist * 60 / speed, (cells - direct) * k], seed
        assert cells > direct  # the road really does double back


def test_map_follow_through_from_a_wrong_length():
    def wrong(q, canon):
        bad = Fraction(canon["a"]) + 2
        speed = int(re.search(r"(\d+) km/h", q.parts[1].prompt).group(1))
        mins = bad * 60 / speed
        if mins.denominator != 1:
            return None
        return {"a": str(bad), "b": str(mins)}

    assert sup.check_follow_through(sup.skill_of(MAP[0]), MAP[1], "b", wrong, seeds=60) >= 10


def test_materials_chain_shape_and_keys():
    skill = _wiring(MATERIALS)
    sup.check_exam_chain(skill, MATERIALS[1], [2, 1, 1, 2], ["number"] * 4)
    for seed in range(80):
        q = sup.exam(MATERIALS[1], seed)
        k = _scale_m_per_cm(q.figure)
        area = _shoelace(q.figure["polygons"][0]["vertices"]) * k * k
        cover, price = (int(g) for g in re.search(r"cover (\d+) square metres and cost \$(\d+)", q.text).groups())
        rolls = -(-int(area) // cover)
        assert area == int(area)
        assert [Fraction(p.answer_text) for p in q.parts] == [area, rolls, rolls * price, rolls * cover - area], seed
        assert 0 < rolls * cover - area < cover


def test_materials_follow_through_chain():
    def wrong(q, canon):
        rolls = int(canon["b"]) + 2
        price = int(re.search(r"cost \$(\d+)", q.text).group(1))
        return {"b": str(rolls), "c": str(rolls * price)}

    assert sup.check_follow_through(sup.skill_of(MATERIALS[0]), MATERIALS[1], "c", wrong) >= 30


def test_conventions_chain_shape_and_keys():
    skill = _wiring(CONVENTIONS)
    sup.check_exam_chain(skill, CONVENTIONS[1], [1, 1, 2, 2], ["number"] * 4)
    for seed in range(80):
        q = sup.exam(CONVENTIONS[1], seed)
        fig = q.figure
        assert "scale" not in fig  # the scale is what the learner finds
        verts = fig["polygons"][0]["vertices"]
        label = next(lab for lab in fig["labels"] if lab["text"].endswith(" m"))
        real = Fraction(label["text"][:-2])
        n = len(verts)
        wall = [
            abs(verts[(i + 1) % n][0] - verts[i][0]) + abs(verts[(i + 1) % n][1] - verts[i][1])
            for i in range(n)
            if (verts[i][0] + verts[(i + 1) % n][0]) / 2 == label["at"][0]
            and (verts[i][1] + verts[(i + 1) % n][1]) / 2 == label["at"][1]
        ]
        assert len(wall) == 1
        k = real / wall[0]
        fence = Fraction(re.search(r"A fence (\d+) m", q.text).group(1))
        want = [k, k * 100, fence / k, sum(_walls(verts)) * k]
        assert [Fraction(p.answer_text) for p in q.parts] == want, seed


def test_answer_variety_and_no_python_notation():
    for pair in (ACTUAL, MAP, MATERIALS, CONVENTIONS):
        counts = Counter(sup.exam(pair[1], s).answer_text for s in range(60))
        assert len(counts) >= 30, (pair[1], len(counts))
        assert max(counts.values()) <= 3
        prompts = {sup.exam(pair[1], s).text for s in range(60)}
        assert len(prompts) >= 20
