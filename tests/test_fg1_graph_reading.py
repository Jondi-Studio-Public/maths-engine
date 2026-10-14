"""Nine "read from a shown figure" skills over `cartesian` and `number_line` figures.

``gradient-from-graph``, ``equation-from-graph``, ``solve-linear-equations-graphically``,
``compare-linear-representations``, ``compare-functions-representations``,
``inequality-solutions-number-line``, ``graph-y-mx``, ``graph-y-mx-plus-b`` and ``plot-linear-points``.
Every tier of every skill is generated through the skill's own sources; the figure is validated, must not
leak an answer, must agree with the key through the registered reader, and every distractor must grade
wrong with the real grader.
"""

import functools
import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.grading import grade, mcq_options
from mathgen.skills import load_skills

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

# The two comparison skills offer three options (A, B or the same); every other mcq offers four.
THREE_OPTIONS = {"compare-linear-representations", "compare-functions-representations"}

# skill id -> (topic, answer kind)
SKILLS = {
    "gradient-from-graph": ("fg1_gradient_from_graph", "fraction"),
    "equation-from-graph": ("fg1_equation_from_graph", "mcq"),
    "solve-linear-equations-graphically": ("fg1_solve_graphically", "mcq"),
    "compare-linear-representations": ("fg1_compare_linear", "mcq"),
    "compare-functions-representations": ("fg1_compare_functions", "mcq"),
    "inequality-solutions-number-line": ("fg1_inequality_number_line", "mcq"),
    "graph-y-mx": ("fg1_proportional_graph", "mcq"),
    "graph-y-mx-plus-b": ("fg1_line_table", "mcq"),
    "plot-linear-points": ("fg1_plotted_points", "mcq"),
}


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


def _gen(skill_id, tier, seed, notes=True):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=tier, notes=notes, **(spec.params or {}))


def _line(fig, i=0):
    (x1, y1), (x2, y2) = fig["lines"][i]["through"]
    m = Fraction(y2 - y1, x2 - x1)
    return m, y1 - m * x1


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", SKILLS)
def test_skill_loads_active_with_the_figure_gate(skill_id):
    topic, kind = SKILLS[skill_id]
    skill = _skills()[skill_id]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.mental is None
    assert skill.answer == kind
    # Skills with an exam-chain source are tested in tests/test_exd_x_w12*_*.py; the reading tiers are the
    # sources before it.
    practice = [s for s in skill.sources if s.difficulty != "exam"]
    assert [s.difficulty for s in practice] == TIERS
    assert {s.topic for s in practice} == {topic}
    assert [list(s.stages) for s in practice] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
    assert skill.misconceptions


@pytest.mark.parametrize("skill_id", ["graph-y-mx", "graph-y-mx-plus-b", "plot-linear-points"])
def test_reworded_skills_say_they_are_reading_skills(skill_id):
    skill = _skills()[skill_id]
    assert "do not ask them to draw, plot or sketch" in skill.description
    assert not skill.name.lower().startswith(("graph", "plot"))


def test_nets_skill_is_built_with_an_exam_tier():
    # The nets skill was deferred until its own generators and figures were built; it is now
    # active, so pin that it carries an exam tier and the grid nets rather than that it stays switched off.
    skill = _skills()["f10-nets-build-solids"]
    assert skill.status == "active"
    assert any(s.difficulty == "exam" for s in skill.sources)
    assert "grid_nets_build_solids" in {s.topic for s in skill.sources}


# ------------------------------------------------------------------ figure, leak, reader, grader


@pytest.mark.parametrize("skill_id", SKILLS)
@pytest.mark.parametrize("tier", TIERS)
def test_figure_reader_key_and_distractors(skill_id, tier):
    topic, kind = SKILLS[skill_id]
    reader = figures.get_reader(topic)
    assert reader is not None
    ids = {m.id for m in _skills()[skill_id].misconceptions}
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert reader(q) == q.answer_text, (tier, seed)
        n_distractors = 2 if skill_id in THREE_OPTIONS else 3
        assert len(q.distractors) == n_distractors and q.answer_text not in q.distractors
        assert len(set(q.distractors)) == n_distractors
        assert set(q.option_notes) == set(q.distractors)
        assert set(q.option_notes.values()) <= ids, set(q.option_notes.values()) - ids
        if kind == "mcq":
            options = mcq_options(q)
            assert len(options) == n_distractors + 1
            for i, opt in enumerate(options):
                assert grade("mcq", q, str(i)).correct is (opt == q.answer_text)
        else:
            assert grade(kind, q, q.answer_text).correct
            for d in q.distractors:
                assert not grade(kind, q, d).correct, (tier, seed, d)


@pytest.mark.parametrize("skill_id", SKILLS)
def test_deterministic_and_bad_tier_raises(skill_id):
    topic, _kind = SKILLS[skill_id]
    for tier in TIERS:
        for seed in range(15):
            assert _gen(skill_id, tier, seed).to_json() == _gen(skill_id, tier, seed).to_json()
    with pytest.raises(ValueError):
        make(topic, seed=1, tier="bogus")


@pytest.mark.parametrize("skill_id", SKILLS)
@pytest.mark.parametrize("tier", TIERS)
def test_prompts_ask_to_read_and_never_to_draw(skill_id, tier):
    shapes = set()
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert not re.search(r"\b(draw|plot|sketch|construct)\w*", q.text, re.I), q.text
        assert "\\(" not in q.latex and "$" not in q.text
        shapes.add(q.text)
    assert len(shapes) >= 3


@pytest.mark.parametrize("skill_id", SKILLS)
def test_alt_text_names_the_object_only(skill_id):
    for tier in TIERS:
        for seed in range(20):
            alt = _gen(skill_id, tier, seed).figure["alt"]
            assert not any(ch.isdigit() for ch in alt), alt


@pytest.mark.parametrize("skill_id", SKILLS)
def test_grid_values_are_readable(skill_id):
    """Every cartesian figure keeps at most ten ticks per axis, so every gridline is labelled."""
    for tier in TIERS:
        for seed in SEEDS:
            fig = _gen(skill_id, tier, seed).figure
            if fig["kind"] != "cartesian":
                continue
            (x0, x1), (y0, y1) = fig["x_range"], fig["y_range"]
            assert (x1 - x0) / fig["x_grid"] + 1 <= 10 and (y1 - y0) / fig["y_grid"] + 1 <= 10
            for ln in fig["lines"]:
                for x, y in ln["through"]:
                    assert x0 <= x <= x1 and y0 <= y <= y1
                    assert x % fig["x_grid"] == 0 and y % fig["y_grid"] == 0


# ------------------------------------------------------------------ gradient from graph


def test_gradient_tiers():
    for seed in SEEDS:
        easy = _gen("gradient-from-graph", "easy", seed)
        m, _c = _line(easy.figure)
        assert m in (2, 3) and easy.figure["equal_scale"] is True
        assert len(easy.figure["points"]) == 2
        medium = _gen("gradient-from-graph", "medium", seed)
        m, _c = _line(medium.figure)
        assert m < 0 or m.denominator != 1
        assert len(medium.figure["points"]) == 2
        hard = _gen("gradient-from-graph", "hard", seed)
        assert hard.figure["equal_scale"] is False and hard.figure["x_grid"] != hard.figure["y_grid"]
        assert hard.figure["points"] == []


def test_gradient_marked_points_are_on_the_line_and_give_the_key():
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = _gen("gradient-from-graph", tier, seed)
            (x1, y1), (x2, y2) = (p["at"] for p in q.figure["points"])
            m, c = _line(q.figure)
            assert y1 == m * x1 + c and y2 == m * x2 + c
            assert Fraction(y2 - y1, x2 - x1) == Fraction(q.answer_text)


def test_gradient_hard_line_crosses_three_grid_intersections():
    for seed in SEEDS:
        fig = _gen("gradient-from-graph", "hard", seed).figure
        m, c = _line(fig)
        xg, yg = fig["x_grid"], fig["y_grid"]
        (x0, x1), (y0, y1) = fig["x_range"], fig["y_range"]
        hits = [x for x in range(x0, x1 + 1) if x % xg == 0 and y0 <= m * x + c <= y1 and (m * x + c) % yg == 0]
        assert len(hits) >= 3, (seed, hits)


def test_gradient_hard_includes_the_counted_in_squares_distractor():
    seen = Counter()
    for seed in range(120):
        seen.update(_gen("gradient-from-graph", "hard", seed).option_notes.values())
    assert seen["gradient-counted-in-squares"] > 20
    assert seen["inverted-gradient"] > 20


# ------------------------------------------------------------------ equation from graph / y = mx / table


def test_equation_tiers():
    for seed in SEEDS:
        easy = _gen("equation-from-graph", "easy", seed).figure
        m, c = _line(easy)
        assert m.denominator == 1 and m > 0 and c != 0
        assert easy["x_range"][0] < 0 < easy["x_range"][1] and easy["y_range"][0] <= c <= easy["y_range"][1]
        medium = _gen("equation-from-graph", "medium", seed).figure
        m, c = _line(medium)
        assert (m < 0 or m.denominator != 1) and medium["y_range"][0] <= c <= medium["y_range"][1]
        hard = _gen("equation-from-graph", "hard", seed).figure
        assert not (hard["x_range"][0] <= 0 <= hard["x_range"][1])


def test_equation_options_are_four_different_lines():
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen("equation-from-graph", tier, seed)
            assert len({q.answer_text, *q.distractors}) == 4


def test_proportional_line_goes_through_the_origin_and_hard_has_unequal_scales():
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen("graph-y-mx", tier, seed)
            m, c = _line(q.figure)
            assert c == 0 and q.answer_text.startswith("y = ")
            assert q.figure["x_range"][0] <= 0 <= q.figure["x_range"][1]
            assert q.figure["y_range"][0] <= 0 <= q.figure["y_range"][1]
            assert (q.figure["x_grid"] != q.figure["y_grid"]) is (tier == "hard")
            if tier == "easy":
                assert m in (1, 2, 3, 4, 5, 6)
    notes = Counter()
    for seed in range(80):
        notes.update(_gen("graph-y-mx", "easy", seed).option_notes.values())
    assert notes["adds-an-unstated-intercept"] > 20


def _pairs(text):
    nums = [int(v) for v in re.findall(r"-?\d+", text)]
    return list(zip(nums[0::2], nums[1::2]))


def test_line_table_key_is_on_the_line_inside_the_window_and_distractors_are_not():
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen("graph-y-mx-plus-b", tier, seed)
            m, c = _line(q.figure)
            (x0, x1), (y0, y1) = q.figure["x_range"], q.figure["y_range"]
            key = _pairs(q.answer_text)
            assert len(key) == 3
            for x, y in key:
                assert y == m * x + c and x0 <= x <= x1 and y0 <= y <= y1
            for d in q.distractors:
                assert not all(y == m * x + c for x, y in _pairs(d)), (tier, seed, d)
            if tier == "hard":
                assert not (x0 <= 0 <= x1)


# ------------------------------------------------------------------ plotted points


def test_plotted_points_tiers():
    for seed in SEEDS:
        easy = _gen("plot-linear-points", "easy", seed)
        assert len(easy.figure["points"]) == 1 and easy.figure["points"][0]["label"] == "P"
        x, y = easy.figure["points"][0]["at"]
        assert easy.answer_text == f"({x}, {y})" and x != 0 and y != 0 and abs(x) != abs(y)
        assert f"({y}, {x})" in easy.distractors
        medium = _gen("plot-linear-points", "medium", seed)
        assert len(medium.figure["points"]) == 3
        assert all("label" not in p for p in medium.figure["points"])
        hard = _gen("plot-linear-points", "hard", seed)
        ((x, _y),) = _pairs(hard.answer_text)
        xs = [p["at"][0] for p in hard.figure["points"]]
        assert x not in xs
        assert not (hard.figure["x_range"][0] <= x <= hard.figure["x_range"][1])


def test_plotted_points_hard_key_is_the_only_option_on_the_line():
    for seed in SEEDS:
        q = _gen("plot-linear-points", "hard", seed)
        (x1, y1), (x2, y2), (x3, y3) = sorted(p["at"] for p in q.figure["points"])
        assert (y2 - y1) * (x3 - x2) == (y3 - y2) * (x2 - x1)
        on = [
            o
            for o in [q.answer_text, *q.distractors]
            if (_pairs(o)[0][1] - y1) * (x2 - x1) == (y2 - y1) * (_pairs(o)[0][0] - x1)
        ]
        assert on == [q.answer_text]


# ------------------------------------------------------------------ solve graphically


def test_solve_graphically_tiers():
    for seed in SEEDS:
        easy = _gen("solve-linear-equations-graphically", "easy", seed)
        assert [ln["label"] for ln in easy.figure["lines"]] == ["A", "B"]
        (m1, c1), (m2, c2) = _line(easy.figure, 0), _line(easy.figure, 1)
        assert m1 != m2 and Fraction(easy.answer_text[4:]) == (c2 - c1) / (m1 - m2)
        medium = _gen("solve-linear-equations-graphically", "medium", seed)
        assert len(medium.figure["lines"]) == 1
        k = int(re.search(r"f\(x\) = (-?\d+)", medium.text).group(1))
        m, c = _line(medium.figure)
        assert m * Fraction(medium.answer_text[4:]) + c == k
        hard = _gen("solve-linear-equations-graphically", "hard", seed)
        assert [ln["label"] for ln in hard.figure["lines"]] == ["A", "B", "C"]
        xs = set()
        for i, j in ((0, 1), (1, 2), (0, 2)):
            (mi, ci), (mj, cj) = _line(hard.figure, i), _line(hard.figure, j)
            xs.add((cj - ci) / (mi - mj))
        assert len(xs) == 3 and Fraction(hard.answer_text[4:]) in xs
        # the other crossings are offered as options (solving the wrong pair)
        assert len({Fraction(o[4:]) for o in [hard.answer_text, *hard.distractors]} & xs) >= 2


def test_solve_graphically_options_never_print_the_solution_on_the_grid():
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen("solve-linear-equations-graphically", tier, seed)
            assert not re.search(r"x = -?\d", q.text), q.text
            assert all("x" not in (ln.get("label") or "").replace("f(x)", "") for ln in q.figure["lines"])


# ------------------------------------------------------------------ comparisons


def _class_counts(skill_id, tier, n=300):
    c = Counter()
    for seed in range(n):
        c[_gen(skill_id, tier, seed).answer_text] += 1
    return c


@pytest.mark.parametrize("skill_id", ["compare-linear-representations", "compare-functions-representations"])
@pytest.mark.parametrize("tier", TIERS)
def test_comparison_answer_class_is_balanced(skill_id, tier):
    counts = _class_counts(skill_id, tier)
    assert len(counts) == 3
    assert all(80 <= n <= 130 for n in counts.values()), counts
    pos = Counter()
    for seed in range(300):
        q = _gen(skill_id, tier, seed)
        pos[mcq_options(q).index(q.answer_text)] += 1
    assert set(pos) == {0, 1, 2}


def _parse_b_equation(text):
    m = re.search(r"y = (-?)(\d*)x(?: ([+-]) (\d+))?", text)
    sign, mag, csign, cval = m.groups()
    mb = int(mag or 1) * (-1 if sign else 1)
    cb = int(cval or 0) * (-1 if csign == "-" else 1)
    return Fraction(mb), Fraction(cb)


def test_compare_linear_intercepts_point_the_wrong_way():
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = _gen("compare-linear-representations", tier, seed)
            mA, cA = _line(q.figure)
            if tier == "easy":
                mB, cB = _parse_b_equation(q.text)
            else:
                (x1, y1), (x2, y2) = _pairs(q.text.split("pairs")[1])[:2]
                mB = Fraction(y2 - y1, x2 - x1)
                cB = y1 - mB * x1
            assert cA != cB
            key = q.answer_text
            if key == "Relationship A":
                assert mA > mB and cA < cB
            elif key == "Relationship B":
                assert mB > mA and cB < cA
            else:
                assert mA == mB
            assert "compares-intercepts-not-gradients" in q.option_notes.values()


def test_compare_linear_hard_is_decided_by_neither_gradient_nor_intercept_alone():
    seen = Counter()
    for seed in range(200):
        q = _gen("compare-linear-representations", "hard", seed)
        mA, cA = _line(q.figure)
        (x1, y1), (x2, y2) = _pairs(q.text.split("pairs")[1])[:2]
        mB = Fraction(y2 - y1, x2 - x1)
        cB = y1 - mB * x1
        X = int(re.search(r"x = (-?\d+)", q.text).group(1))
        d0, dX = cA - cB, (mA * X + cA) - (mB * X + cB)
        assert d0 != 0
        if q.answer_text == "They have the same y-value":
            assert dX == 0
        else:
            assert (dX > 0) == (q.answer_text == "Relationship A") and dX != 0
            assert (d0 > 0) != (dX > 0)  # the starting values point the wrong way
            seen[(X > 0, (mA > mB) == (dX > 0))] += 1
    # a far positive x agrees with the gradient, a far negative x disagrees with it
    assert seen[(True, True)] and seen[(False, False)]
    assert not seen[(True, False)] and not seen[(False, True)]


def test_compare_functions_tiers():
    for seed in SEEDS:
        easy = _gen("compare-functions-representations", "easy", seed)
        mf, cf = _line(easy.figure)
        mul = re.search(r"multiplies it by (-?\d+) and then (adds|subtracts) (\d+)", easy.text)
        a, b = int(mul.group(1)), int(mul.group(3)) * (1 if mul.group(2) == "adds" else -1)
        x0 = int(re.search(r"input is (-?\d+)", easy.text).group(1))
        diff = (mf * x0 + cf) - (a * x0 + b)
        assert easy.answer_text == ("Function f" if diff > 0 else "Function g" if diff < 0 else "They are equal")
        if diff:
            assert (cf > b) != (diff > 0)  # comparing starting values would be wrong
        medium = _gen("compare-functions-representations", "medium", seed)
        assert medium.figure["lines"] == [] and len(medium.figure["curves"]) == 1
        assert "greater maximum value" in medium.text
        hard = _gen("compare-functions-representations", "hard", seed)
        assert len(hard.figure["curves"]) == 1 and "linear" in hard.text
        X = int(re.search(r"x = (-?\d+)", hard.text).group(1))
        assert abs(X) >= 3


# ------------------------------------------------------------------ inequality on a number line


def test_inequality_tiers():
    for seed in SEEDS:
        easy = _gen("inequality-solutions-number-line", "easy", seed)
        assert len(easy.figure["rays"]) == 1 and easy.figure["intervals"] == []
        options = mcq_options(easy)
        if all(re.fullmatch(r"-?\d+", o) for o in options):
            # the "which number is a solution" shape: exactly one offered number is in the shaded ray
            ray = easy.figure["rays"][0]
            p = ray["from"]
            sign = 1 if ray["direction"] == "right" else -1
            inside = [o for o in options if (int(o) - p) * sign > 0 or (not ray.get("open") and int(o) == p)]
            assert len(options) == 4 and len(inside) == 1 and inside[0] == easy.answer_text
        else:
            assert {o[:1] for o in options} == {"x"}
            assert len({re.sub(r"[<≤>≥]", "?", o) for o in options}) == 1
            assert {re.search(r"[<≤>≥]", o).group() for o in options} == {"<", "≤", ">", "≥"}
        medium = _gen("inequality-solutions-number-line", "medium", seed)
        assert len(medium.figure["intervals"]) == 1 and medium.figure["rays"] == []
        assert len({o.replace("<", "?").replace("≤", "?") for o in mcq_options(medium)}) == 1
        hard = _gen("inequality-solutions-number-line", "hard", seed)
        assert len(hard.figure["rays"]) == 1
        assert all(re.match(r"^-?\d*x [+-] \d+ [<≤>≥] -?\d+$", o) for o in mcq_options(hard))


def _solution(option):
    m = re.match(r"^(-?)(\d*)x ([+-]) (\d+) ([<≤>≥]) (-?\d+)$", option)
    sign, mag, bs, bv, sym, cv = m.groups()
    a = int(mag or 1) * (-1 if sign else 1)
    b = int(bv) * (1 if bs == "+" else -1)
    c = int(cv)
    t = Fraction(c - b, a)
    if a < 0:
        sym = {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}[sym]
    return ("left" if sym in "<≤" else "right"), t, sym in "<>"


def test_inequality_hard_has_exactly_one_matching_option_and_a_sign_flip_trap():
    flips = 0
    for seed in range(100):
        q = _gen("inequality-solutions-number-line", "hard", seed)
        ray = q.figure["rays"][0]
        shown = (ray["direction"], Fraction(ray["from"]), bool(ray["open"]))
        match = [o for o in mcq_options(q) if _solution(o) == shown]
        assert match == [q.answer_text]
        if "forgets-to-reverse-the-sign-when-dividing-by-a-negative" in q.option_notes.values():
            flips += 1
            # the option that traps the learner who forgets to reverse is the one whose
            # un-reversed solution is the shown set
            trap = next(o for o, n in q.option_notes.items() if n.startswith("forgets"))
            m = re.match(r"^(-?)(\d*)x ([+-]) (\d+) ([<≤>≥]) (-?\d+)$", trap)
            assert m.group(1) == "-"
    assert flips > 15


def test_inequality_ends_are_open_and_closed_equally_often():
    ends = Counter()
    for seed in range(200):
        q = _gen("inequality-solutions-number-line", "easy", seed)
        ray = q.figure["rays"][0]
        ends[(ray["direction"], bool(ray["open"]))] += 1
    assert len(ends) == 4 and min(ends.values()) > 30
