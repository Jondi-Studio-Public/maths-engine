"""The eleven early-years skills of batch eyz: twos facts, estimating a collection, angles, everyday chance, solids and symmetry.

Every skill is generated through its own sources at every tier and 60 seeds: the real grader accepts the key and
rejects every distractor, every tagged distractor is a listed misconception in the shared catalogue, figures are valid,
do not show the key, and (for a figure skill) the registered reader re-derives the key from the drawing alone. Then the
checks each topic promises: that an answer built backwards matches a second route, such as parsing the stem back into
numbers, counting the bag again, or recomputing the angle from the drawn coordinates.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import pytest

from mathgen import figures, make, svg
from mathgen.generators import eyz_angles_build as angles
from mathgen.generators import eyz_chance_build as chance
from mathgen.generators import eyz_estimate_collection_build as estimate
from mathgen.generators import eyz_solids_build as solids
from mathgen.generators import eyz_symmetry_build as symmetry
from mathgen.grading.typed import grade, mcq_options
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import READ_ALOUD_MAX_WORDS, load_skills
from scripts import skill_contract

SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

# skill id -> (topic, answer kind, acara code, domain, has a figure, tiers)
SKILLS = {
    "ey-twos-doubles-related-facts": (
        "eyz_twos_facts",
        "number",
        "ac9m2a03",
        "primary-number",
        False,
        [*TIERS, "exam"],
    ),
    "ey-estimate-collection-quantity": (
        "eyz_estimate_collection",
        "number",
        "ac9m3n05",
        "primary-number",
        True,
        [*TIERS, "exam"],
    ),
    "ey-compare-with-right-angle": ("eyz_right_angle_compare", "set", "ac9m3m05", "primary-geometry-time", True, TIERS),
    "ey-name-angle-by-size": ("eyz_angle_kind", "set", "ac9m4m04", "primary-geometry-time", True, TIERS),
    "ey-estimate-angle-size": ("eyz_angle_estimate", "number", "ac9m4m04", "primary-geometry-time", True, TIERS),
    "ey-list-possible-outcomes": ("eyz_list_outcomes", "set", "ac9m3p01", "statistics-probability", False, TIERS),
    "ey-describe-chance-from-counts": (
        "eyz_chance_words",
        "mcq",
        "ac9m3p01",
        "statistics-probability",
        False,
        [*TIERS, "exam"],
    ),
    "ey-features-of-named-solids": ("eyz_named_solids", "number", "ac9m3sp01", "primary-geometry-time", True, TIERS),
    "ey-prism-pyramid-counts": (
        "eyz_prism_pyramid_counts",
        "number",
        "ac9m3sp01",
        "primary-geometry-time",
        False,
        [*TIERS, "exam"],
    ),
    "ey-symmetry-from-drawn-shapes": (
        "eyz_symmetry_count",
        "number",
        "ac9m4sp03",
        "primary-geometry-time",
        True,
        [*TIERS, "exam"],
    ),
}
READ_ALOUD = {
    s
    for s in SKILLS
    if s != "ey-list-possible-outcomes" and s != "ey-describe-chance-from-counts" and s != "ey-prism-pyramid-counts"
}
ALL = [(s, t) for s, spec in SKILLS.items() for t in spec[5]]


@functools.lru_cache(maxsize=1)
def _skills():
    return {s.id: s for s in load_skills()[1]}


@functools.lru_cache(maxsize=None)
def _q(skill_id, tier, seed):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=tier, **(spec.params or {}))


def _units(q):
    """The answer-bearing objects of a question: each part of a multi-part question, else the question."""
    return list(q.parts) if q.parts else [q]


def _reply_ok(kind, unit, text) -> bool:
    if kind == "mcq":
        return grade("mcq", unit, str(mcq_options(unit).index(text))).correct
    return grade(kind, unit, text).correct


# ------------------------------------------------------------------ wiring


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_skill_wiring(skill_id):
    topic, kind, code, domain, has_figure, tiers = SKILLS[skill_id]
    skill = _skills()[skill_id]
    assert skill.status == "active" and skill.domain == domain and skill.answer == kind
    assert code in skill.tags and "acara-f10-v9" in skill.tags
    assert skill.needs_figure is has_figure
    assert (skill.capability_gates == ["figure_stimulus"]) is has_figure
    assert skill.read_aloud is (skill_id in READ_ALOUD)
    assert [s.difficulty for s in skill.sources] == tiers
    assert skill.placement_eligible is False and skill.learner_band == "early"
    assert len(skill.misconceptions) >= 5 and all(m.description for m in skill.misconceptions)
    steps = [skill.difficulty_progression[t].steps for t in tiers]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)


@pytest.mark.parametrize(("skill_id", "tier"), ALL)
def test_every_tagged_distractor_is_a_listed_misconception(skill_id, tier):
    listed = {m.id for m in _skills()[skill_id].misconceptions}
    tags = set()
    for seed in SEEDS:
        for unit in _units(_q(skill_id, tier, seed)):
            assert set(unit.option_notes) <= set(unit.distractors), (tier, seed)
            tags |= set(unit.option_notes.values())
    assert tags and tags <= listed and tags <= set(CATALOGUE), (tags - listed, tags - set(CATALOGUE))


# ------------------------------------------------------------------ grading and figures


@pytest.mark.parametrize(("skill_id", "tier"), ALL)
def test_real_grader_accepts_the_key_and_rejects_every_distractor(skill_id, tier):
    for seed in SEEDS:
        for unit in _units(_q(skill_id, tier, seed)):
            kind = unit.answer_kind if _q(skill_id, tier, seed).parts else SKILLS[skill_id][1]
            assert _reply_ok(kind, unit, unit.answer_text), (tier, seed, unit.answer_text)
            assert len(unit.distractors) >= 3 and unit.answer_text not in unit.distractors
            for d in unit.distractors:
                assert not _reply_ok(kind, unit, d), (tier, seed, d, unit.answer_text)


@pytest.mark.parametrize(("skill_id", "tier"), [(s, t) for s, t in ALL if SKILLS[s][4]])
def test_figure_is_valid_does_not_leak_and_the_reader_agrees(skill_id, tier):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    reader = figures.get_reader(spec.topic)
    assert reader is not None
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
        everything = [q.answer_text, *q.distractors] if not q.parts else [p.answer_text for p in q.parts]
        figures.assert_no_leak(q.figure, everything)
        assert reader(q) == q.answer_text, (tier, seed, q.text)
        svg.figure_svg(q.figure)


@pytest.mark.parametrize(("skill_id", "tier"), [(s, t) for s, t in ALL if SKILLS[s][4]])
def test_a_figure_skill_gives_a_picture_for_every_question(skill_id, tier):
    assert all(_q(skill_id, tier, seed).figure is not None for seed in SEEDS)


@pytest.mark.parametrize(("skill_id", "tier"), [(s, t) for s, t in ALL if not SKILLS[s][4]])
def test_a_text_skill_draws_nothing(skill_id, tier):
    assert all(_q(skill_id, tier, seed).figure is None for seed in SEEDS)


@pytest.mark.parametrize("skill_id", sorted(READ_ALOUD))
def test_read_aloud_prompts_are_short(skill_id):
    for tier in SKILLS[skill_id][5]:
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            for prompt in [p.prompt for p in q.parts] if q.parts else [q.text]:
                assert len(prompt.split()) <= READ_ALOUD_MAX_WORDS, (tier, seed, prompt)


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("family", sorted(skill_contract.FAMILIES))
def test_skill_contract(skill_id, family):
    skill_contract.FAMILIES[family](skill_id)


@pytest.mark.parametrize("topic", sorted({v[0] for v in SKILLS.values()}))
def test_unknown_tier_raises(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, tier="bogus")


@pytest.mark.parametrize("skill_id", ["ey-describe-chance-from-counts"])
def test_mcq_key_position_is_balanced(skill_id):
    for tier in SKILLS[skill_id][5]:
        if tier == "exam":
            continue
        counts: dict[int, int] = {}
        for seed in range(200):
            q = make(SKILLS[skill_id][0], seed=seed, difficulty=tier)
            pos = mcq_options(q).index(q.answer_text)
            counts[pos] = counts.get(pos, 0) + 1
        assert max(counts.values()) / 200 < 0.5, (skill_id, tier, counts)


# ------------------------------------------------------------------ 2A03: twos facts


def _nums(text):
    return [int(n) for n in re.findall(r"\d+", text)]


def test_twos_answers_match_a_second_route():
    for seed in range(120):
        for tier in ("easy", "medium", "hard"):
            q = make("eyz_twos_facts", seed=seed, difficulty=tier)
            text, got = q.text, int(q.answer_text)
            nums = _nums(text)
            if text.startswith(("What is double", "Double", "Find double")):
                assert got == 2 * nums[0], text
            elif "groups of" in text or "lots of" in text or "times" in text:
                assert got == 2 * nums[-1], text
            elif re.match(r"^(What is )?\d+ \+ \d+", text) or re.match(r"^\d+ \+ \d+ = \?", text):
                assert got == nums[0] + nums[1], text
            elif "Use it" in text or "You know" in text:
                # "2 x a = k. ... 2 x b": the new fact is 2 b
                assert got == 2 * nums[-1] or got == 2 * nums[3], text
            elif "missing" in text or text.startswith("?"):
                assert 2 * got == [n for n in nums if n != 2][0] or got * 2 == nums[1], text
            elif "half" in text.lower() or "÷ 2" in text or "Share" in text or "shared" in text or "Pack" in text:
                assert 2 * got == [n for n in nums if n != 2][0] or 2 * got == nums[0], text


def test_twos_exam_chain_follows_through():
    for seed in SEEDS:
        q = _q("ey-twos-doubles-related-facts", "exam", seed)
        a, b, c = (int(p.answer_text) for p in q.parts)
        extra = int(re.search(r"part_a \+ 2\*(\d+)", q.parts[1].follow_through).group(1))
        assert b == a + 2 * extra and 2 * c == b and a % 2 == 0
        assert not {a, b, c} & set(_nums(q.text + " " + q.parts[1].prompt))


# ------------------------------------------------------------------ 3N05: estimating a collection


@pytest.mark.parametrize("tier", TIERS)
def test_estimate_counts_never_end_in_the_middle_of_a_ten(tier):
    for seed in range(120):
        q = make("eyz_estimate_collection", seed=seed, difficulty=tier)
        pts = q.figure["symbols"]
        assert len({tuple(s["at"]) for s in pts}) == len(pts), "two objects share a square"
        if tier != "hard":
            assert len(pts) % 10 in (0, 1, 2, 8, 9), (seed, len(pts))
            assert int(q.answer_text) == estimate._round10(len(pts))
        else:
            wall = q.figure["lines"][0]["points"][0][0]
            left = sum(1 for s in pts if s["at"][0] < wall)
            right = len(pts) - left
            assert left % 10 in (0, 1, 2, 8, 9) and right % 10 in (0, 1, 2, 8, 9)


def test_estimate_easy_box_scales_to_the_answer():
    for seed in range(120):
        q = make("eyz_estimate_collection", seed=seed, difficulty="easy")
        (box,) = q.figure["polygons"]
        xs = [v[0] for v in box["vertices"]]
        inside = sum(1 for s in q.figure["symbols"] if min(xs) < s["at"][0] < max(xs))
        assert (max(xs) - min(xs)) * 5 == q.figure["grid"]["width"]
        assert estimate._round10(5 * inside) == int(q.answer_text)


def test_estimate_exam_total_follows_from_the_two_sides():
    for seed in SEEDS:
        q = _q("ey-estimate-collection-quantity", "exam", seed)
        a, b, c, d = (int(p.answer_text) for p in q.parts)
        assert c == a + b and q.parts[2].follow_through == "part_a + part_b"
        m = int(_nums(q.parts[3].prompt)[0])
        assert d == m * c and q.parts[3].follow_through == f"{m}*part_c"


# ------------------------------------------------------------------ 3M05 and 4M04: angles


def _angle_sizes(fig):
    return {angles._label_of(fig, s): angles._sweep(fig, s) for s in fig["angles"]}


@pytest.mark.parametrize("tier", TIERS)
def test_right_angle_comparison_is_never_close_to_a_coin_toss(tier):
    near = {"easy": 20, "medium": 20, "hard": 12}[tier]
    for seed in range(120):
        q = make("eyz_right_angle_compare", seed=seed, difficulty=tier)
        sizes = _angle_sizes(q.figure)
        for label, deg in sizes.items():
            if label == "R":
                assert abs(deg - 90) < 0.5
            else:
                assert abs(deg - 90) < 0.5 or abs(deg - 90) >= near - 0.5, (seed, label, deg)
        assert ("R" in sizes) is (tier != "easy")
        assert ("dashed" in str(q.figure["segments"])) is (tier == "easy")


def test_right_angle_sets_are_never_empty_and_cover_the_four_angles():
    for tier in TIERS:
        for seed in range(120):
            q = make("eyz_right_angle_compare", seed=seed, difficulty=tier)
            assert q.answer and set(q.answer) <= set("ABCD")
            assert all(d.startswith("{") for d in q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_angle_kinds_come_from_the_drawn_coordinates(tier):
    asked_kinds = set()
    for seed in range(200):
        q = make("eyz_angle_kind", seed=seed, difficulty=tier)
        sizes = _angle_sizes(q.figure)
        assert set(sizes) == set("ABCD") and set(q.answer) <= set("ABCD") and 1 <= len(q.answer) <= 3
        text = q.text.removeprefix("Which of the angles A to D are ").removeprefix("Which angles are")
        text = text.strip().removesuffix(" List the letters.").removesuffix("?")
        group = set(angles._GROUPS.get(text, (text,)))
        asked_kinds |= group
        # every drawn angle is well clear of the 90 and 180 boundaries unless it is exactly right or straight
        for deg in sizes.values():
            kind = angles.angle_class(deg)
            assert kind in ("right", "straight") or min(abs(deg - 90), abs(deg - 180)) >= 19.9, deg
        assert q.answer == sorted(k for k, deg in sizes.items() if angles.angle_class(deg) in group)
        if tier == "easy":
            assert {angles.angle_class(d) for d in sizes.values()} <= {"acute", "right", "obtuse"}
    assert asked_kinds == ({"acute", "right", "obtuse"} if tier == "easy" else set(angles._KINDS))


@pytest.mark.parametrize("tier", TIERS)
def test_angle_estimates_are_readings_between_benchmark_rays(tier):
    for seed in range(120):
        q = make("eyz_angle_estimate", seed=seed, difficulty=tier)
        got = int(q.answer_text)
        (spec,) = q.figure["angles"]
        assert abs(angles._sweep(q.figure, spec) - got) < 0.5 and got % 10 == 0
        labelled = {int(t[:-1]) for t in figures.labels(q.figure) if t.endswith("°")}
        assert got not in labelled
        assert (got > 180) is (tier == "hard")
        if tier == "easy":
            assert got % 30 in (10, 20)
        assert labelled == set(angles._LABELLED[tier])


# ------------------------------------------------------------------ 3P01: chance


def _word_from_text(text):
    """Recompute the chance word from the counts in the stem alone."""
    t = text.split("What is the chance")[0].split("Chance of")[0]
    counts = [(int(n), c) for n, c in re.findall(r"(\d+) ([a-z]+)", t.replace(" and ", ", "))]
    return counts


def test_chance_words_agree_with_the_counts_in_the_stem():
    for tier in TIERS:
        for seed in range(150):
            q = make("eyz_chance_words", seed=seed, difficulty=tier)
            text = q.text
            m = re.match(
                r"A bag has (.+?) counters\. Which sentence about picking (?:(\w+) or (\w+)|(.+)) is true\?", text
            )
            if m:
                bag = {c: int(n) for n, c in re.findall(r"(\d+) ([a-z]+)", m.group(1))}
                total = sum(bag.values())
                if m.group(2) and m.group(2) != "a" and m.group(3):
                    fav = bag[m.group(2)] + bag[m.group(3)]
                else:
                    ask = m.group(4) or m.group(0).split("picking ")[1].removesuffix(" is true?")
                    colour = ask.split()[-2] if ask.endswith("counter") else ask.split()[-1]
                    if "not" in ask:
                        fav = total - bag[colour]
                    elif colour in bag:
                        fav = bag[colour]
                    else:
                        fav = 0
                assert chance._word_of(Fraction(fav, total)) == chance.word_in(q.answer_text), text
            elif "last 10" in text or "in 10" in text:
                n = int(re.search(r"(\d+) (?:of|times)", text).group(1))
                assert chance._word_of(Fraction(n, 10)) == chance.word_in(q.answer_text), text
            else:
                events = dict(chance._DIE_EVENTS)
                what = re.search(r"about rolling (.+) is true\?", text).group(1)
                assert events[what] == chance.word_in(q.answer_text)


def test_every_chance_word_is_asked_and_the_gaps_never_are():
    for tier in ("easy", "medium", "hard"):
        words = {chance.word_in(make("eyz_chance_words", seed=s, difficulty=tier).answer_text) for s in range(300)}
        assert words <= set(chance.WORDS) and len(words) >= 3
    assert chance._word_of(Fraction(4, 10)) is None and chance._word_of(Fraction(6, 10)) is None
    assert chance._word_of(Fraction(3, 10)) == "unlikely" and chance._word_of(Fraction(7, 10)) == "likely"


def test_chance_exam_not_red_chance_matches_the_counts():
    for seed in SEEDS:
        q = _q("ey-describe-chance-from-counts", "exam", seed)
        total, notred = int(q.parts[0].answer_text), int(q.parts[1].answer_text)
        assert chance._word_of(Fraction(notred, total)) == chance.word_in(q.parts[2].answer_text)
        assert q.parts[1].follow_through.startswith("part_a - ")


def test_listed_outcomes_are_complete_and_distinct():
    for tier in TIERS:
        for seed in range(150):
            q = make("eyz_list_outcomes", seed=seed, difficulty=tier)
            assert len(set(q.answer)) == len(q.answer) >= 2
            if "coin" in q.text.lower():
                assert q.answer == ["heads", "tails"]
            if "die" in q.text and tier == "easy":
                assert q.answer == [1, 2, 3, 4, 5, 6]
            m = re.search(r"picked from the word (\w+)", q.text)
            if m:
                assert q.answer == sorted(set(m.group(1)))
            m = re.search(r"picked from the number (\d+)", q.text)
            if m:
                assert q.answer == sorted({int(d) for d in m.group(1)})


# ------------------------------------------------------------------ 3SP01: solids


@pytest.mark.parametrize(("kind", "n"), [(k, n) for k in ("prism", "pyramid") for n in range(3, 13)])
def test_built_solids_obey_eulers_formula_and_the_closed_forms(kind, n):
    v, e, f = solids.counts_of(solids.solid_faces(kind, n))
    assert (v, e, f) == ((2 * n, 3 * n, n + 2) if kind == "prism" else (n + 1, 2 * n, n + 1))


def test_named_solid_facts_match_a_built_solid():
    assert solids.FACTS["cube"] == (6, 0, 12, 8) and solids.FACTS["square pyramid"] == (5, 0, 8, 5)
    assert solids.FACTS["cylinder"][0] == 2 and solids.FACTS["cone"][0] == 1 and solids.FACTS["sphere"][0] == 0


@pytest.mark.parametrize("tier", TIERS)
def test_named_solid_questions_recount_from_the_picture(tier):
    for seed in range(120):
        q = make("eyz_named_solids", seed=seed, difficulty=tier)
        assert {it["shape"] for it in q.figure["items"]} <= set(solids.FACTS)
        if tier == "easy":
            assert len(q.figure["items"]) == 1
        elif tier == "medium":
            assert 2 <= len(q.figure["items"]) <= 3
        else:
            assert len(q.figure["items"]) >= 4


def test_prism_pyramid_hard_works_back_to_the_base():
    for seed in range(150):
        q = make("eyz_prism_pyramid_counts", seed=seed, difficulty="hard")
        m = re.match(r"A (prism|pyramid) has (\d+) (faces|edges|vertices)\. How many (faces|edges|vertices)", q.text)
        kind, value, given, asked = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        hits = []
        for n in range(3, 13):
            v, e, f = solids.counts_of(solids.solid_faces(kind, n))
            have = {"faces": f, "edges": e, "vertices": v}
            if have[given] == value:
                hits.append(have[asked])
        assert hits == [int(q.answer_text)], q.text


def test_prism_pyramid_exam_uses_eulers_formula_on_the_first_two_answers():
    for seed in SEEDS:
        q = _q("ey-prism-pyramid-counts", "exam", seed)
        f, e, v, cm = (int(p.answer_text) for p in q.parts)
        assert v - e + f == 2 and q.parts[2].follow_through == "2 - part_a + part_b"
        k = int(_nums(q.parts[3].prompt)[0])
        assert cm == e * k and q.parts[3].follow_through == f"{k}*part_b"


# ------------------------------------------------------------------ 4SP03: symmetry


@pytest.mark.parametrize(
    ("name", "lines", "order"),
    [
        ("square", 4, 4),
        ("rectangle", 2, 2),
        ("rhombus", 2, 2),
        ("parallelogram", 0, 2),
        ("kite", 1, 1),
        ("isosceles trapezium", 1, 1),
        ("equilateral triangle", 3, 3),
        ("isosceles triangle", 1, 1),
        ("scalene triangle", 0, 1),
        ("regular pentagon", 5, 5),
        ("regular hexagon", 6, 6),
        ("L-shape, equal arms", 1, 1),
        ("L-shape, unequal arms", 0, 1),
        ("T-shape", 1, 1),
        ("plus", 4, 4),
        ("S-shape", 0, 2),
        ("dart", 1, 1),
        ("house", 1, 1),
        ("five-point star", 5, 5),
        ("stretched hexagon", 2, 2),
        ("three-blade pinwheel", 0, 3),
        ("six-blade pinwheel", 0, 6),
    ],
)
def test_symmetry_brute_force_matches_what_a_teacher_knows(name, lines, order):
    import random

    assert symmetry.symmetry_counts(symmetry._catalogue(random.Random(3))[name]) == (lines, order)


@pytest.mark.parametrize("tier", TIERS)
def test_symmetry_answers_come_from_the_drawn_points(tier):
    for seed in range(120):
        q = make("eyz_symmetry_count", seed=seed, difficulty=tier)
        pos = {p["id"]: p["at"] for p in q.figure["points"]}
        lines, order = symmetry.symmetry_counts([pos[i] for i in q.figure["polygons"][0]["points"]])
        assert int(q.answer_text) == (lines if "line" in q.text else order)
        assert ("full turn" in q.text or "order" in q.text) is ("line" not in q.text)


def test_symmetry_exam_smallest_turn_is_360_over_the_order():
    for seed in SEEDS:
        q = _q("ey-symmetry-from-drawn-shapes", "exam", seed)
        lines, order, turn, several = (int(p.answer_text) for p in q.parts)
        assert turn * order == 360 and q.parts[2].follow_through == "360/part_b"
        k = int(_nums(q.parts[3].prompt)[0])
        assert several == k * turn and q.parts[3].follow_through == f"{k}*part_c"
        assert (lines == 0) or order >= 1
