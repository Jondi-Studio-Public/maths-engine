"""PV1D's V-shaped absolute value graph skill.

Covers ``mathgen/generators/pv1d_abs_vertex_build.py`` (topic ``pv1d_abs_vertex``,
skill ``senior-use-and-apply-the-notation-left-x-right``): determinism, the REAL pair
grader on the key, spelling variants, every distractor and a deliberately wrong value,
the in-generator check firing on a corrupted figure reading, the deferred figure skill's
shape, F1 (every tier generates a valid figure through the skill's own sources), the
no-leak rule, the figure reader, distractor counts and the clean-pair boundary proven
RED against the unconstrained draw.
"""

import functools
import random
from collections import Counter

import pytest

import mathgen.generators.pv1d_abs_vertex_build as gen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import figures, make
from mathgen.core import VerificationError
from mathgen.text_quality import prompt_shape

TOPIC = "pv1d_abs_vertex"
SKILL = "senior-use-and-apply-the-notation-left-x-right"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ------------------------------------------------------------------ determinism


def test_deterministic_at_every_tier():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            a = make(TOPIC, seed=seed, difficulty=difficulty)
            b = make(TOPIC, seed=seed, difficulty=difficulty)
            assert a.to_json() == b.to_json(), (difficulty, seed)


# ------------------------------------------------------------- the real grader


def test_real_grader_accepts_key_and_its_spellings():
    grader = GRADERS["pair"]
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = _make_from_sources(difficulty, seed)
            h, k = q.answer["x"], q.answer["y"]
            for given in (q.answer_text, f"{h}, {k}", f"({h},{k})", f"x = {h}, y = {k}"):
                assert grader(q, given).correct, (difficulty, seed, given)


def test_real_grader_rejects_every_distractor_and_wrong_values():
    grader = GRADERS["pair"]
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = _make_from_sources(difficulty, seed)
            h, k = q.answer["x"], q.answer["y"]
            for wrong in (f"({h + 1}, {k})", f"({h}, {k + 1})", f"({-h}, {k})", f"({k}, {h})", f"{h}"):
                assert not grader(q, wrong).correct, (difficulty, seed, wrong, q.answer_text)


def test_a_bare_single_number_is_not_accepted():
    grader = GRADERS["pair"]
    for difficulty in DIFFICULTIES:
        q = _make_from_sources(difficulty, 0)
        assert not grader(q, str(q.answer["x"])).correct
        assert not grader(q, str(q.answer["y"])).correct


# ---------------------------------------------------- the in-generator check fires


def test_in_generator_check_fires_on_a_corrupted_corner_reading(monkeypatch):
    real = gen._corner_of_curve

    def wrong(curve):
        h, k = real(curve)
        return h + 1, k

    monkeypatch.setattr(gen, "_corner_of_curve", wrong)
    for difficulty in DIFFICULTIES:
        with pytest.raises(VerificationError):
            make(TOPIC, seed=0, difficulty=difficulty)


def test_in_generator_check_fires_when_the_arms_route_disagrees(monkeypatch):
    monkeypatch.setattr(gen, "_corner_from_arms", lambda curve: (0, 0))
    with pytest.raises(VerificationError):
        make(TOPIC, seed=0, difficulty="medium")


def test_options_never_contain_the_key_even_for_a_degenerate_pair():
    # |h| == |k| is excluded from every draw, but if it were drawn, (k, h) would equal
    # the key: the helper drops it rather than offering a second correct answer.
    texts, notes = gen._options(3, 3)
    assert "(3, 3)" not in texts and set(notes) == set(texts)


# ----------------------------------------------------------------- the skill file


def test_skill_loads_with_the_deferred_figure_shape():
    skill = _skills_dict()[SKILL]
    assert skill.answer == "pair"
    assert skill.status == "deferred" and skill.deferred_reason == "visual"
    assert skill.needs_figure is True
    assert skill.capability_gates == ["figure_stimulus"]
    assert skill.mental is None
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["function-notation"]
    assert [s.difficulty for s in skill.sources] == DIFFICULTIES
    assert [s.stages for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert all(s.topic == TOPIC for s in skill.sources)


def test_every_distractor_note_names_a_declared_misconception():
    declared = {m.id for m in _skills_dict()[SKILL].misconceptions}
    used = set()
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert not q.option_notes  # the shared inertness sweep forbids setting it
            h, k = q.answer["x"], q.answer["y"]
            other = None
            if difficulty in ("hard", "exam"):
                other = [v for v in (gen._corner_of_curve(c) for c in q.figure["curves"]) if v != (h, k)][0]
            texts, notes = gen._options(h, k, other)
            assert texts == q.distractors and set(notes) == set(texts), (difficulty, seed)
            used |= set(notes.values())
    assert used <= declared
    assert declared <= used, declared - used


# ---------------------------------------------------- F1: a valid figure, always


def test_f1_every_tier_draws_a_valid_figure_through_the_skills_own_sources():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = _make_from_sources(difficulty, seed)
            assert q.figure is not None, (difficulty, seed)
            assert figures.validate(q.figure) == [], (difficulty, seed)
            n_curves = 1 if difficulty in ("easy", "medium") else 2
            assert len(q.figure["curves"]) == n_curves
            n_points = {"easy": 1, "medium": 0, "hard": 2, "exam": 0}[difficulty]
            assert len(q.figure["points"]) == n_points


def test_no_leak_on_every_seed():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
            for text in figures.labels(q.figure):
                assert not any(c.isdigit() for c in text), (difficulty, seed, text)


def test_alt_text_names_only_the_object():
    for difficulty, alt in (
        ("easy", "a V-shaped graph on a coordinate grid"),
        ("medium", "a V-shaped graph on a coordinate grid"),
        ("hard", "two V-shaped graphs on a coordinate grid"),
        ("exam", "two V-shaped graphs on a coordinate grid"),
    ):
        assert make(TOPIC, seed=1, difficulty=difficulty).figure["alt"] == alt


def test_the_prompt_never_prints_the_answer_or_a_digit():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert not any(c.isdigit() for c in q.text), (difficulty, seed, q.text)
            assert q.answer_text not in q.text
            assert "**" not in q.text and "*" not in q.text


# --------------------------------------------------------------- the reader


def test_registered_reader_matches_the_key_on_every_tier():
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for difficulty in DIFFICULTIES:
        for seed in range(1000):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert reader(q) == q.answer_text, (difficulty, seed)


def test_reader_does_not_read_the_key():
    """The reader must work from the figure and the prompt: blank the key and re-read."""
    reader = figures.get_reader(TOPIC)
    for difficulty in DIFFICULTIES:
        q = make(TOPIC, seed=5, difficulty=difficulty)
        stripped = type("Q", (), {"figure": q.figure, "text": q.text})()
        assert reader(stripped) == q.answer_text


# ----------------------------------------------------------- shape and variety


def test_each_tier_has_three_prompt_shapes_and_twenty_distinct_questions_in_60_seeds():
    for difficulty in DIFFICULTIES:
        qs = [make(TOPIC, seed=s, difficulty=difficulty) for s in range(60)]
        assert len({prompt_shape(q) for q in qs}) >= 3, difficulty
        assert len({q.text for q in qs}) >= 20, difficulty
        assert len({str(q.figure["curves"]) for q in qs}) >= 20, difficulty


def test_hard_and_exam_add_prompt_shapes_over_every_lower_tier():
    shapes = {d: {prompt_shape(make(TOPIC, seed=s, difficulty=d)) for s in range(60)} for d in DIFFICULTIES}
    assert shapes["hard"] - (shapes["easy"] | shapes["medium"])
    assert shapes["exam"] - (shapes["easy"] | shapes["medium"] | shapes["hard"])


def test_at_least_three_distractors_on_every_draw():
    for difficulty in DIFFICULTIES:
        counts = Counter(len(make(TOPIC, seed=s, difficulty=difficulty).distractors) for s in range(200))
        assert min(counts) >= 3, (difficulty, counts)


def test_two_graph_tiers_always_offer_the_other_graphs_vertex():
    for difficulty in ("hard", "exam"):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            h, k = q.answer["x"], q.answer["y"]
            other = [v for v in (gen._corner_of_curve(c) for c in q.figure["curves"]) if v != (h, k)]
            assert len(other) == 1 and f"({other[0][0]}, {other[0][1]})" in q.distractors, (difficulty, seed)


def test_key_values_are_integers_in_range_and_never_degenerate():
    for difficulty in DIFFICULTIES:
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            h, k = q.answer["x"], q.answer["y"]
            assert isinstance(h, int) and isinstance(k, int)
            assert -5 <= h <= 5 and -5 <= k <= 5
            assert h != 0 and k != 0 and abs(h) != abs(k)


def test_the_prompt_asks_for_the_right_object():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            text = make(TOPIC, seed=seed, difficulty=difficulty).text
            assert "(h, k)" in text and "vertex" in text.lower(), (difficulty, seed, text)


# ------------------------------------------------- the boundary, RED vs unconstrained


def _violations(pair):
    return not gen._compatible(*pair)


def test_clean_pair_boundary_is_violated_by_the_unconstrained_draw_and_never_by_the_generator():
    rng = random.Random("unconstrained")
    unconstrained = [(rng.choice(gen._CELLS), rng.choice(gen._CELLS)) for _ in range(400)]
    assert sum(_violations(p) for p in unconstrained) > 40  # RED: a plain draw is often unreadable
    for difficulty in ("hard", "exam"):
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            a, b = [gen._corner_of_curve(c) for c in q.figure["curves"]]
            assert gen._compatible(a, b), (difficulty, seed, a, b)


def test_exam_has_exactly_one_graph_above_the_axis_and_none_touching_it():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        ks = sorted(gen._corner_of_curve(c)[1] for c in q.figure["curves"])
        assert ks[0] < 0 < ks[1]


def test_exam_answer_graph_is_not_always_the_higher_one():
    higher = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        ks = [gen._corner_of_curve(c)[1] for c in q.figure["curves"]]
        higher += q.answer["y"] == max(ks)
    assert 60 <= higher <= 140, higher


def test_answer_concentration_is_under_a_quarter():
    for difficulty in DIFFICULTIES:
        counts = Counter(make(TOPIC, seed=s, difficulty=difficulty).answer_text for s in range(200))
        assert max(counts.values()) / 200 <= 0.25, (difficulty, counts.most_common(3))
