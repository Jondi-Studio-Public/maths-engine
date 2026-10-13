"""Geometry, tree and Venn figure skills.

Converted from a description in words to a drawn figure (the same ids, answer kinds and graders):
``straight-line-angles``, ``f10-angle-relationships``, ``f10-parallel-line-angle-theorems``,
``pythagorean-hypotenuse``, ``pythagorean-leg`` and ``triangle-area-base-height``. New:
``f10-probability-tree-diagrams`` and ``f10-venn-diagram-counts``.

Every tier of every source is generated through the skill's own sources, so what is tested is what a learner
is served.
"""

import collections
import functools
import math

import re

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.generators import _space_kit as kit
from mathgen.grading.typed import grade
from mathgen.skills import load_skills

SEEDS = range(60)

# skill id -> (answer kind, topic, figure kind, tiers)
SKILLS = {
    "straight-line-angles": ("number", "straight_line_angle", "diagram", ["easy"]),
    "f10-angle-relationships": ("number", "p10e_angle_relationships", "diagram", ["easy", "medium", "hard"]),
    "f10-parallel-line-angle-theorems": (
        "number",
        "p11d_parallel_line_angle_theorems",
        "diagram",
        ["easy", "medium", "hard"],
    ),
    "pythagorean-hypotenuse": ("number", "trigonometry", "diagram", ["easy", "medium", "hard"]),
    "pythagorean-leg": ("number", "trigonometry", "diagram", ["easy", "medium", "hard"]),
    "triangle-area-base-height": ("number", "triangle_area_base_height", "diagram", ["easy", "medium", "hard"]),
    "f10-probability-tree-diagrams": ("fraction", "fig_prob_tree", "tree_diagram", ["easy", "medium", "hard"]),
    "f10-venn-diagram-counts": ("number", "fig_venn_counts", "venn", ["easy", "medium", "hard"]),
}


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


def _source(skill_id, tier):
    return next(s for s in _skills()[skill_id].sources if s.difficulty == tier)


@functools.lru_cache(maxsize=None)
def _gen(skill_id, tier, seed):
    src = _source(skill_id, tier)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


CASES = [(sid, tier) for sid, (_a, _t, _k, tiers) in SKILLS.items() for tier in tiers]


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", SKILLS)
def test_skill_is_active_behind_the_figure_gate(skill_id):
    answer, topic, _kind, tiers = SKILLS[skill_id]
    skill = _skills()[skill_id]
    assert skill.answer == answer
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False and skill.mental is None
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert [s.difficulty for s in skill.sources if s.difficulty != "exam"] == tiers
    assert all(s.topic == topic for s in skill.sources if s.difficulty != "exam")
    assert skill.misconceptions


# ------------------------------------------------------------------ every figure, every tier


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_figure_is_valid_does_not_leak_and_the_reader_agrees_with_the_key(skill_id, tier):
    _answer, topic, kind, _tiers = SKILLS[skill_id]
    reader = figures.get_reader(topic)
    assert reader is not None
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert q.figure is not None and q.figure["kind"] == kind, (seed, q.text)
        assert figures.validate(q.figure) == [], (seed, figures.validate(q.figure))
        figures.assert_no_leak(q.figure, [q.answer_text])
        assert reader(q) == q.answer_text, (skill_id, tier, seed, q.text, q.answer_text)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_the_key_grades_correct_and_every_distractor_wrong(skill_id, tier):
    kind = SKILLS[skill_id][0]
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert grade(kind, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, (seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert not grade(kind, q, d).correct, (skill_id, tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_deterministic_and_varied(skill_id, tier):
    seen = set()
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        again = make(q.topic, seed=seed, difficulty=q.difficulty, **(_source(skill_id, tier).params or {}))
        assert q.to_json() == again.to_json()
        seen.add((q.text, q.answer_text, repr(q.figure)))
    assert len(seen) >= 50, (skill_id, tier, len(seen))


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_prompt_never_states_the_asked_value_in_the_figure_text(skill_id, tier):
    """The figure's own text (labels, alt) is clean; the prompt's sentence is plain text with no inline maths."""
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert "\\(" not in q.text and "\\(" not in q.latex
        assert q.figure["alt"] and not any(ch.isdigit() for ch in q.figure["alt"])


# ------------------------------------------------------------------ angles: drawn not to scale, answer not measurable


def _letter_label(fig, text=None):
    """The one single-letter label among the angle or side labels (the unknown, whichever letter was drawn)."""
    found = {x["label"] for x in fig["angles"] + fig["segments"] if re.fullmatch(r"[a-z]", x.get("label", ""))}
    if len(found) > 1 and text is not None:  # the parallel-line text names a given angle `a` and asks for another
        found = {
            c
            for c in found
            if re.search(rf"(?:marked|angle|find|is) {c}\b(?! is)", text) and f"angle {c} is" not in text
        }
    assert len(found) == 1, found
    return found.pop()


def _drawn_sweep(fig, label, text=None):
    label = _letter_label(fig, text) if label == "x" else label
    arcs = [a for a in kit.arcs_of(fig) if a.get("label") == label]
    assert len(arcs) == 1, (label, fig["angles"])
    return arcs[0]["sweep"]


@pytest.mark.parametrize(
    "skill_id,tier",
    [
        ("straight-line-angles", "easy"),
        ("f10-angle-relationships", "easy"),
        ("f10-angle-relationships", "medium"),
        ("f10-parallel-line-angle-theorems", "easy"),
        ("f10-parallel-line-angle-theorems", "medium"),
    ],
)
def test_the_unknown_angle_is_drawn_at_least_ten_degrees_from_its_true_size(skill_id, tier):
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert q.figure["not_to_scale"] is True
        assert abs(_drawn_sweep(q.figure, "x", q.text) - int(q.answer_text)) >= 10, (seed, q.answer_text)


@pytest.mark.parametrize("skill_id", ["f10-angle-relationships", "f10-parallel-line-angle-theorems"])
def test_hard_tier_marks_two_expressions_and_draws_the_asked_angle_off_scale(skill_id):
    for seed in SEEDS:
        q = _gen(skill_id, "hard", seed)
        expressions = [a for a in q.figure["angles"] if re.search(r"[a-z]", a["label"])]
        assert len(expressions) == 2 and q.figure["not_to_scale"] is True
        sizes = [kit.arcs_of(q.figure)[i]["sweep"] for i in range(2)]
        if skill_id == "f10-angle-relationships":
            assert abs(sizes[0] - sizes[1]) < 0.5  # vertically opposite, drawn equal
            asked = sizes[0]
        else:
            asked_label = q.text.split("angle marked ")[1].rsplit(", in degrees", 1)[0]
            asked = next(a for a in kit.arcs_of(q.figure) if a["label"] == asked_label)["sweep"]
        assert abs(asked - int(q.answer_text)) >= 10, (seed, q.answer_text, asked)


def test_angle_tiers_escalate_in_what_must_be_done():
    point = {t: _gen("f10-angle-relationships", t, 3).figure for t in ("easy", "medium", "hard")}
    assert [len(point[t]["angles"]) for t in ("easy", "medium")] == [3, 4]  # one more addend to track
    assert all(kit.is_expression(a["label"]) for a in point["hard"]["angles"])  # expressions, not sizes
    assert any("co-interior" in _gen("f10-parallel-line-angle-theorems", "medium", s).text for s in SEEDS)
    assert not any("co-interior" in _gen("f10-parallel-line-angle-theorems", "easy", s).text for s in SEEDS)


def test_no_figure_shows_a_right_or_straight_angle_that_is_not_marked():
    for skill_id in ("straight-line-angles", "f10-angle-relationships", "f10-parallel-line-angle-theorems"):
        for tier in SKILLS[skill_id][3]:
            for seed in SEEDS:
                for arc in kit.arcs_of(_gen(skill_id, tier, seed).figure):
                    assert abs(arc["sweep"] - 90) > 2 and abs(arc["sweep"] - 180) > 2


def test_the_parallel_lines_are_arrow_marked_and_rotate():
    rotations = set()
    for seed in SEEDS:
        fig = _gen("f10-parallel-line-angle-theorems", "easy", seed).figure
        marked = [s for s in fig["segments"] if s.get("arrows")]
        assert len(marked) == 2
        pos = {p["id"]: p["at"] for p in fig["points"]}
        rotations.add(round(math.atan2(pos["B1"][1] - pos["A1"][1], pos["B1"][0] - pos["A1"][0]), 1))
    assert len(rotations) >= 8


# ------------------------------------------------------------------ right triangles and the triangle area


@pytest.mark.parametrize("skill_id", ["pythagorean-hypotenuse", "pythagorean-leg"])
@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_pythagoras_figure_is_a_right_triangle_with_x_on_the_asked_side(skill_id, tier):
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        fig = q.figure
        assert fig["not_to_scale"] is True
        assert len(fig["angles"]) == 1 and fig["angles"][0]["right"] is True
        labels = sorted(s["label"] for s in fig["segments"])
        unknown = _letter_label(fig)
        assert sum(re.fullmatch(r"[a-z]", lab) is not None for lab in labels) == 1 and len(labels) == 3
        hyp = next(s for s in fig["segments"] if fig["angles"][0]["at"] not in (s["from"], s["to"]))
        assert (hyp["label"] == unknown) == (skill_id == "pythagorean-hypotenuse")
        assert ("hypotenuse" in q.text) == (skill_id == "pythagorean-hypotenuse")
        assert ("1 decimal place" in q.text) == (tier == "hard")


@pytest.mark.parametrize("skill_id", ["pythagorean-hypotenuse", "pythagorean-leg"])
def test_the_hypotenuse_is_not_always_at_the_bottom_and_the_drawing_does_not_follow_the_numbers(skill_id):
    lowest, aspects = collections.Counter(), []
    for seed in SEEDS:
        fig = _gen(skill_id, "easy", seed).figure
        pos = {p["id"]: p["at"] for p in fig["points"]}
        sides = {f"{seg['from']}{seg['to']}": (pos[seg["from"]][1] + pos[seg["to"]][1]) / 2 for seg in fig["segments"]}
        lowest[min(sides, key=sides.get)] += 1
        aspects.append(round(math.dist(pos["C"], pos["B"]) / math.dist(pos["C"], pos["A"]), 1))
    assert len(lowest) >= 3, lowest  # every side takes a turn at the bottom
    assert lowest["AB"] < 0.5 * len(SEEDS)  # the hypotenuse (AB) is usually not the bottom side
    assert max(aspects) >= 1.1 and min(aspects) <= 0.9  # both leg orders drawn, whatever the labels say


def test_pythagorean_tiers_escalate_from_exact_triples_to_rounded_roots():
    easy = [_gen("pythagorean-hypotenuse", "easy", s).answer_text for s in SEEDS]
    hard = [_gen("pythagorean-hypotenuse", "hard", s).answer_text for s in SEEDS]
    assert all("." not in a for a in easy) and all("." in a for a in hard)


def test_triangle_area_is_drawn_to_scale_and_the_slant_is_only_labelled_at_the_hard_tier():
    for tier, labelled in (("easy", 2), ("medium", 2), ("hard", 3)):
        for seed in SEEDS:
            fig = _gen("triangle-area-base-height", tier, seed).figure
            assert fig["not_to_scale"] is False
            assert len([s for s in fig["segments"] if s.get("label")]) == labelled
            assert sum(1 for s in fig["segments"] if s.get("dashed")) == 1
            assert fig["angles"][0]["right"] is True


def test_triangle_area_orientation_varies():
    bearings = set()
    for seed in SEEDS:
        fig = _gen("triangle-area-base-height", "easy", seed).figure
        pos = {p["id"]: p["at"] for p in fig["points"]}
        bearings.add(round(math.degrees(math.atan2(pos["B"][1] - pos["A"][1], pos["B"][0] - pos["A"][0])) % 360))
    assert len(bearings) >= 4


# ------------------------------------------------------------------ tree


def _tree_branches(fig):
    return [(n["branch"], [k["branch"] for k in n["children"]]) for n in fig["root"]["children"]]


def test_tree_tiers_differ_in_what_must_be_done():
    for seed in SEEDS:
        easy = _gen("f10-probability-tree-diagrams", "easy", seed).figure
        medium = _gen("f10-probability-tree-diagrams", "medium", seed).figure
        hard = _gen("f10-probability-tree-diagrams", "hard", seed).figure
        # easy: with replacement, so the second draw repeats the first; one outcome marked ?
        branches = _tree_branches(easy)
        assert branches[0][1] == branches[1][1] and "?" not in str(branches)
        leaves = [k for n in easy["root"]["children"] for k in n["children"]]
        assert [k.get("outcome_p") for k in leaves].count("?") == 1
        # medium: without replacement, so the second draw differs from the first; nothing blank or marked
        branches = _tree_branches(medium)
        assert branches[0][1] != branches[1][1] and "?" not in str(branches)
        assert not any(k.get("outcome_p") for n in medium["root"]["children"] for k in n["children"])
        # hard: two of the three sets of branches have a blank, one each
        groups = [[n["branch"] for n in hard["root"]["children"]]] + [
            [k["branch"] for k in n["children"]] for n in hard["root"]["children"]
        ]
        assert sorted(g.count("?") for g in groups) == [0, 1, 1]


def test_tree_medium_question_is_two_paths_and_hard_is_a_complement():
    assert all("different colours" in _gen("f10-probability-tree-diagrams", "medium", s).text for s in SEEDS)
    assert all("at least one" in _gen("f10-probability-tree-diagrams", "hard", s).text for s in SEEDS)
    assert all("marked ? on the tree" in _gen("f10-probability-tree-diagrams", "easy", s).text for s in SEEDS)


def test_tree_answers_are_proper_fractions_and_do_not_name_a_count():
    for tier in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = _gen("f10-probability-tree-diagrams", tier, seed)
            assert 0 < q.answer["p"] < q.answer["q"] and math.gcd(q.answer["p"], q.answer["q"]) == 1
            assert not any(ch.isdigit() for ch in q.text)


# ------------------------------------------------------------------ Venn


def test_venn_tiers_differ_in_shape():
    for seed in SEEDS:
        easy = _gen("f10-venn-diagram-counts", "easy", seed).figure
        medium = _gen("f10-venn-diagram-counts", "medium", seed).figure
        hard = _gen("f10-venn-diagram-counts", "hard", seed).figure
        assert len(easy["sets"]) == 2 and list(easy["regions"].values()).count("?") == 1
        assert len(medium["sets"]) == 2 and "?" not in medium["regions"].values()
        assert len(hard["sets"]) == 3 and list(hard["regions"].values()).count("?") == 1


def test_venn_hard_question_is_sometimes_by_group_total_and_sometimes_by_circle_total():
    texts = [_gen("f10-venn-diagram-counts", "hard", s).text for s in SEEDS]
    assert any(t.startswith("There are") for t in texts) and any(t.startswith(("In total", "The Venn")) for t in texts)


def test_venn_medium_asks_three_kinds_of_question_and_no_prompt_states_a_region():
    kinds = collections.Counter()
    for seed in SEEDS:
        q = _gen("f10-venn-diagram-counts", "medium", seed)
        kinds["either" if "or both" in q.text else "exactly one" if "exactly one" in q.text else "one set"] += 1
        assert all(r.isdigit() for r in q.figure["regions"].values())
        assert not any(ch.isdigit() for ch in q.text)
    assert set(kinds) == {"either", "exactly one", "one set"}
    for tier in ("easy", "hard"):
        for seed in SEEDS:
            q = _gen("f10-venn-diagram-counts", tier, seed)
            shown = [r for r in q.figure["regions"].values() if r.isdigit()]
            numbers = {t for t in q.text.replace(".", " ").split() if t.isdigit()}
            assert len(numbers) == 1  # the one stated total, and no region count
            assert shown


# ------------------------------------------------------------------ the old wording is still available


@pytest.mark.parametrize(
    "topic,kwargs",
    [
        ("p10e_angle_relationships", {}),
        ("p11d_parallel_line_angle_theorems", {}),
        ("trigonometry", {"kinds": ("side_hyp",)}),
        ("triangle_area_base_height", {}),
    ],
)
def test_without_figure_true_the_question_is_the_old_words_only_form(topic, kwargs):
    for seed in range(10):
        q = make(topic, seed=seed, difficulty="easy", **kwargs)
        assert q.figure is None
