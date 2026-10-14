"""Tests `pv1f_graph_is_function` (ACMMM024), an active figure skill whose every tier is generated through its
own sources.

``pv1f_graph_is_function`` (``answer: mcq``, ACMMM024). The skill is ``status: active``
(visual): every tier of every source is generated here through the skill's own sources.
"""

import collections
import functools
import re

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1f_function_graph_build as gen
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape
from scripts.measure_tiers import classify_skill

SKILL_ID = "senior-understand-the-concept-of-the-graph-of-a"
TOPIC = "pv1f_graph_is_function"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
OPT = re.compile(r"^It is (not )?a function: the line x = (-?\d+) meets the graph (once|more than once)\.$")


@functools.lru_cache(maxsize=1)
def _skill():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _gen(tier, seed):
    spec = next(s for s in _skill().sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


def _parse(option):
    m = OPT.match(option)
    assert m, option
    return ("N" if m.group(1) else "F"), int(m.group(2)), m.group(3)


# ------------------------------------------------------------------ the skill file


def test_skill_loads_with_the_figure_stimulus_shape():
    skill = _skill()
    assert skill.answer == "mcq" and skill.domain == "functions-sequences"
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.mental is None
    assert [s.difficulty for s in skill.sources] == TIERS
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert skill.misconceptions


def test_every_tier_draws_a_valid_figure_with_no_leak():
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen(tier, seed)
            assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
            assert q.figure["alt"] == "a graph on a coordinate grid"
            assert all(not p.get("label") for p in q.figure["points"])


def test_deterministic_at_every_tier():
    for tier in TIERS:
        for seed in SEEDS:
            assert _gen(tier, seed).to_json() == _gen(tier, seed).to_json()


# ------------------------------------------------------------------ the real grader and the option set


@pytest.mark.parametrize("tier", TIERS)
def test_mcq_grader_accepts_the_key_and_rejects_every_other_option(tier):
    for seed in SEEDS:
        q = _gen(tier, seed)
        options = mcq_options(q)
        assert len(options) == 4 and len(set(options)) == 4, (tier, seed, options)
        assert len(q.distractors) == 3 and q.answer_text not in q.distractors
        for idx, opt in enumerate(options):
            assert grade_mcq(q, idx).correct == (opt == q.answer_text), (tier, seed, idx)
        assert q.answer_latex == f"plain:{q.answer_text}"


@pytest.mark.parametrize("tier", TIERS)
def test_every_option_pairs_a_verdict_with_a_witness_and_the_key_is_never_the_odd_one_out(tier):
    for seed in range(120):
        q = _gen(tier, seed)
        parsed = [_parse(o) for o in [q.answer_text, *q.distractors]]
        verdicts = collections.Counter(v for v, _a, _how in parsed)
        assert verdicts == {"N": 2, "F": 2}, (tier, seed, parsed)
        assert all(how == ("once" if v == "F" else "more than once") for v, _a, how in parsed)
        assert len({(v, a) for v, a, _h in parsed}) == 4


@pytest.mark.parametrize("tier", TIERS)
def test_class_draw_and_key_position_are_balanced_at_400_seeds(tier):
    pos, cls = collections.Counter(), collections.Counter()
    n = 400
    for seed in range(n):
        q = _gen(tier, seed)
        pos[mcq_options(q).index(q.answer_text)] += 1
        cls[_parse(q.answer_text)[0]] += 1
    assert len(pos) == 4 and len(cls) == 2
    assert max(pos.values()) / n < 0.36 and min(pos.values()) / n > 0.16, (tier, dict(pos))
    assert 0.40 < cls["N"] / n < 0.60, (tier, dict(cls))


def test_key_witness_concentration_over_200_seeds():
    for tier in TIERS:
        counts = collections.Counter(_gen(tier, s).answer_text for s in range(200))
        assert max(counts.values()) / 200 < 0.25, (tier, counts.most_common(2))
        assert len(counts) >= 10, (tier, len(counts))


# ------------------------------------------------------------------ the witness lines (construction's counts)


@pytest.mark.parametrize("tier", TIERS)
def test_non_function_key_meets_twice_and_every_distractor_at_most_once(tier):
    seen = collections.Counter()
    for seed in range(300):
        q = _gen(tier, seed)
        verdict, a, _how = _parse(q.answer_text)
        n_key = gen._counts_at(q.figure, a)
        if verdict == "N":
            seen["N"] += 1
            assert n_key >= 2 and not gen._graph_is_function(q.figure)
            for d in q.distractors:
                assert gen._counts_at(q.figure, _parse(d)[1]) <= 1, (tier, seed, d)
        else:
            seen["F"] += 1
            assert n_key == 1 and gen._graph_is_function(q.figure)
            zero = [d for d in q.distractors if _parse(d)[0] == "F"]
            assert len(zero) == 1 and gen._counts_at(q.figure, _parse(zero[0])[1]) == 0
            for d in q.distractors:
                if _parse(d)[0] == "N":
                    assert gen._counts_at(q.figure, _parse(d)[1]) <= 1
    assert seen["N"] > 80 and seen["F"] > 80


# ------------------------------------------------------------------ the reader


@pytest.mark.parametrize("tier", TIERS)
def test_registered_reader_equals_the_key_1000_seeds(tier):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in range(1000):
        q = _gen(tier, seed)
        assert reader(q) == q.answer_text, (tier, seed)


def test_pure_figure_reading_does_not_use_the_answer():
    q = _gen("exam", 4)
    key = q.answer_text
    options = [key, *q.distractors]
    q.answer_text, q.answer, q.distractors = "zzz", "zzz", []
    assert gen._read_from_figure(q.figure, options) == key


def test_in_generator_check_fires_when_the_construction_and_figure_disagree(monkeypatch):
    real = gen._counts_at
    monkeypatch.setattr(gen, "_counts_at", lambda fig, a: real(fig, a) + 1)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, tier="easy")


def test_reader_refuses_a_figure_where_two_options_are_true():
    q = _gen("easy", 2)
    options = [q.answer_text, q.answer_text]
    with pytest.raises(VerificationError):
        gen._read_from_figure(q.figure, options)


# ------------------------------------------------------------------ boundaries per tier


def test_medium_join_depends_on_whether_both_facing_dots_are_filled():
    flags = collections.Counter()
    for seed in range(300):
        q = _gen("medium", seed)
        jump = []
        for s in q.figure["segments"]:
            if (
                "open_end" in s
                and s["to"][0] not in (min(t["from"][0] for t in q.figure["segments"]),)
                and s["to"][0] != max(t["to"][0] for t in q.figure["segments"])
            ):
                jump.append((s["to"][0], s["open_end"]))
            if "open_start" in s and s["from"][0] != min(t["from"][0] for t in q.figure["segments"]):
                jump.append((s["from"][0], s["open_start"]))
        assert len(jump) == 2 and jump[0][0] == jump[1][0], (seed, jump)
        j, ends = jump[0][0], sorted(f for _x, f in jump)
        verdict = _parse(q.answer_text)[0]
        if verdict == "N":
            assert ends == [False, False] and _parse(q.answer_text)[1] == j
        else:
            assert ends == [False, True]
            # the jump's own 'is not a function' claim is always offered, and is false
            assert any(_parse(d)[0] == "N" and _parse(d)[1] == j for d in q.distractors), (seed, j)
        flags[(verdict, tuple(ends))] += 1
    assert len(flags) == 2


def test_hard_distractor_is_a_line_that_only_touches_the_graph():
    for seed in range(200):
        q = _gen("hard", seed)
        touching = [d for d in q.distractors if _parse(d)[0] == "N" and gen._counts_at(q.figure, _parse(d)[1]) == 1]
        assert touching, (seed, q.distractors)


def test_exam_has_one_separate_dot_and_its_line_is_the_key_or_the_lure():
    for seed in range(200):
        q = _gen("exam", seed)
        dots = [p for p in q.figure["points"] if not p.get("open")]
        assert len(dots) in (1, 3)  # a segment base draws its end dots itself; an arc base adds two points
        dot_x = dots[-1]["at"][0]  # the separate dot is always the last point
        verdict, a, _how = _parse(q.answer_text)
        if verdict == "N":
            assert a == dot_x, (seed, a, dot_x)
        else:
            assert a != dot_x and any(_parse(d) == ("N", dot_x, "more than once") for d in q.distractors), seed


# ------------------------------------------------------------------ variety and escalation


@pytest.mark.parametrize("tier", TIERS)
def test_each_tier_has_three_shapes_and_twenty_distinct_questions(tier):
    qs = [make(TOPIC, seed=s, difficulty=tier) for s in range(60)]
    assert len({prompt_shape(q) for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 20


def test_every_tier_adds_a_prompt_shape():
    verdict, adds = classify_skill(_skill(), 60)
    assert verdict == "escalates" and adds == [False, True, True, True], adds


@pytest.mark.parametrize("tier", TIERS)
def test_prompts_state_no_witness_and_no_python_notation(tier):
    for seed in SEEDS:
        q = _gen(tier, seed)
        assert "**" not in q.text and "*" not in q.text
        assert not re.search(r"x = -?\d", q.text), q.text


@pytest.mark.parametrize("tier", TIERS)
def test_no_witness_line_is_named_by_two_options_so_repetition_is_no_tell(tier):
    """A line named twice would point at the key (function) or away from it (non-function)
    without the figure being read. Every option names its own line, on every tier."""
    for seed in range(400):
        q = _gen(tier, seed)
        lines = [_parse(o)[1] for o in [q.answer_text, *q.distractors]]
        assert len(set(lines)) == 4, (tier, seed, lines)


def test_the_tiers_lure_is_offered_as_a_false_claim():
    """medium: the one-dot jump (function) / the continuous join (non-function); hard: the
    touching line; exam: the dot's line (function). Each is an 'is not a function' claim
    whose line meets the graph once."""
    for tier, class_with_lure in (("medium", "F"), ("hard", "F"), ("hard", "N"), ("exam", "F")):
        hits = 0
        for seed in range(150):
            q = _gen(tier, seed)
            if _parse(q.answer_text)[0] != class_with_lure:
                continue
            hits += 1
            touching = [d for d in q.distractors if _parse(d)[0] == "N" and gen._counts_at(q.figure, _parse(d)[1]) == 1]
            assert touching, (tier, seed, q.distractors)
        assert hits > 40
