"""See mathgen/generators/p19g_concurrent_forces_build.py for the tier design, the BOUNDARY against the shipping
1-D momentum/force skill, and the exam tier's inverse-resolution escalation.

Includes a from-scratch, generator-independent oracle
(`_oracle_resultant`/`test_oracle_reresolves_every_rendered_question`) that parses `question.text` with a
regular expression and recomputes the resultant/acceleration from the STATED forces
alone.generators.p19g_concurrent_forces_build`.
"""

import math
import re
from collections import Counter
from itertools import combinations

import pytest

from mathgen.grading.typed import GRADERS, grade_pair
from mathgen.skills import load_skills
from mathgen import make

TOPIC = "p19g_concurrent_forces_resultant"
SKILL_ID = "senior-understand-motion-of-a-body-under-concurrent-forces"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


def _pair(q):
    return tuple(q.answer.values())


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_and_not_the_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3
        assert all(d for d in q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_an_unlabelled_ordered_pair_of_integers(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict)
        mag, direction = _pair(q)
        assert isinstance(mag, int) and isinstance(direction, int)
        assert q.answer_text == f"({mag}, {direction})"
        assert "=" not in q.answer_text  # unlabelled: no "label = value" segment
        assert 0 <= direction < 360
        # every distractor is formatted alike -- an unlabelled integer pair,
        # so no option is identifiable by shape alone
        for d in q.distractors:
            assert re.fullmatch(r"\(-?\d+, -?\d+\)", d), d


# --- non-collinearity: never answerable by 1-D signed addition alone --------

#: Every force-direction mention in this skill's prompts is written "at N
#: degrees" (checked against every tier's own templates) -- deliberately
#: NOT the bare `r"(-?\d+) degrees"` pattern, which also matches the fixed
#: reference-convention phrase "...0-360 degrees" and would misparse "-360"
#: as a spurious stated angle out of "0-360".
_ANGLE_RE = re.compile(r"at (-?\d+) degrees")


def _stated_angles(text: str) -> list[int]:
    return [int(a) for a in _ANGLE_RE.findall(text)]


@pytest.mark.parametrize("difficulty", TIERS)
def test_stated_forces_are_never_all_collinear(difficulty):
    """At least one pair of stated force directions differs by neither 0 nor
    180 degrees, every seed, every difficulty -- the BOUNDARY against the
    shipping 1-D senior-examine-momentum-force-resultant-force-action-and-
    reaction skill (module docstring)."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        angles = _stated_angles(q.text)
        assert len(angles) >= 2, q.text
        assert any(abs(a - b) % 180 != 0 for a, b in combinations(angles, 2)), (
            f"all-collinear angles {angles} in: {q.text}"
        )


# --- real grader accept/reject, and the 0.05-off proof -----


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_answer_rejects_distractors_and_a_wrong_value(difficulty):
    grade_pair_direct = GRADERS["pair"]
    assert grade_pair_direct is grade_pair
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_pair(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_pair(q, d).correct
        mag, direction = _pair(q)
        deliberately_wrong = f"({mag + 1000}, {(direction + 47) % 360})"
        assert not grade_pair(q, deliberately_wrong).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_grader_rejects_a_value_0_05_off_either_component(difficulty):
    """S24's own required proof: because both components are stated as
    integers (no decimal point at all), `mathgen/grading/typed.py::_component_target`
    grades each one EXACTLY -- a value 0.05 off either the magnitude or the
    direction must be rejected, even though `grade_pair` still applies the
    0.051 voice-transcript tolerance ceiling by default."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        mag, direction = _pair(q)
        off_magnitude = f"({mag + 0.05}, {direction})"
        off_direction = f"({mag}, {direction + 0.05})"
        assert not grade_pair(q, off_magnitude).correct, (q.answer_text, off_magnitude)
        assert not grade_pair(q, off_direction).correct, (q.answer_text, off_direction)


@pytest.mark.parametrize("difficulty", TIERS)
def test_grader_rejects_the_reversed_order_pair(difficulty):
    """`answer_text` carries no `label = value` segments, so
    `_labelled_pair_verdict` never engages and `grade_pair` falls to its
    POSITIONAL comparison -- a swapped magnitude/direction pair is a
    genuinely different (wrong) answer, unless it happens to coincide
    (skipped in that rare case)."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        mag, direction = _pair(q)
        if mag == direction:
            continue
        swapped = f"({direction}, {mag})"
        assert not grade_pair(q, swapped).correct


# --- tier-specific structural checks ----------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_two_force_tiers_ask_for_resultant_force_not_acceleration(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "acceleration" not in q.text
        assert "resultant force" in q.text
        assert len(_stated_angles(q.text)) == 2


def test_easy_tier_uses_only_nice_angles():
    nice = {a for a in range(0, 360, 15) if a % 30 == 0 or a % 45 == 0}
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        for a in _stated_angles(q.text):
            assert a in nice, (a, q.text)


def test_medium_tier_uses_only_arbitrary_non_multiple_of_15_angles():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        for a in _stated_angles(q.text):
            assert a % 15 != 0, (a, q.text)


def test_hard_tier_has_three_forces_a_mass_and_asks_for_acceleration():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert len(_stated_angles(q.text)) == 3, q.text
        assert "acceleration" in q.text
        assert re.search(r"mass \d+ kg", q.text), q.text


def test_exam_tier_has_one_unknown_magnitude_force_and_a_given_component():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "unknown magnitude" in q.text or "magnitude is not given" in q.text
        assert re.search(r"component of -?\d+ N in the 90-degree direction", q.text) or re.search(
            r"90-degree direction is -?\d+ N", q.text
        ), q.text
        assert len(_stated_angles(q.text)) == 2, q.text


# --- answer-concentration measure (>= 200 seeds/tier, top pair < 15%) -----


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_not_concentrated_at_200_seeds(difficulty):
    counts = Counter()
    n = 200
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        counts[q.answer_text] += 1
    top_share = counts.most_common(1)[0][1] / n
    assert top_share < 0.15, (difficulty, counts.most_common(5), top_share)


# --- mathgen.skills.load_skills / skill shape --------------------------------


def test_skill_loads_with_expected_shape():
    _, skills = load_skills()
    by_id = {s.id: s for s in skills}
    assert SKILL_ID in by_id
    skill = by_id[SKILL_ID]
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []
    assert skill.status == "active"
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
    exam_sources = [src for src in skill.sources if src.difficulty == "exam"]
    assert all(src.stages == ["exam"] for src in exam_sources)


# --- independent oracle (generator-blind) -----------------------------------

_FORCE_RE = re.compile(r"(-?\d+(?:\.\d+)?) N at (-?\d+) degrees")
_MASS_RE = re.compile(r"mass (\d+) kg")
#: The resultant's stated 90-degree component: written either before ("a
#: component of {Ry} N in the 90-degree direction") or after ("in the
#: 90-degree direction is {Ry} N") across this skill's exam-tier templates.
_COMPONENT_BEFORE_RE = re.compile(r"component of (-?\d+) N in the 90-degree direction")
_COMPONENT_AFTER_RE = re.compile(r"90-degree direction is (-?\d+) N")
_ANSWER_PAIR_RE = re.compile(r"\((-?\d+), (-?\d+)\)")


def _oracle_resultant(text: str, answer_text: str) -> None:
    """Re-derive the graded (magnitude, direction) pair from `question.text`
    ALONE, via a route that shares no code with
    `mathgen.generators.p19g_concurrent_forces_build` (not imported here at
    all). Handles the two shapes this skill's own tiers produce: N stated
    forces with both magnitude and direction known (easy/medium/hard, mass
    optional), or exactly one known force plus one direction-only force with
    a stated resultant component (exam)."""
    stated_forces = [(float(f), float(a)) for f, a in _FORCE_RE.findall(text)]
    mass_match = _MASS_RE.search(text)
    mass = float(mass_match.group(1)) if mass_match else None
    component_match = _COMPONENT_BEFORE_RE.search(text) or _COMPONENT_AFTER_RE.search(text)

    if component_match is not None:
        # exam shape: one known force ("...50 N, acts at 203 degrees" or
        # "50 N at 203 degrees" -- wording differs by template, so the
        # magnitude is read as the first bare "N N" numeral in the text,
        # not via `_FORCE_RE`'s stricter "N at" adjacency) + one
        # direction-only unknown force.
        known_f = float(re.search(r"(-?\d+(?:\.\d+)?) N", text).group(1))
        all_angles = [int(a) for a in _ANGLE_RE.findall(text)]
        assert len(all_angles) == 2, (all_angles, text)
        known_theta = float(all_angles[0])
        unknown_theta = float(all_angles[1])
        ry = float(component_match.group(1))
        sin_b = math.sin(math.radians(unknown_theta))
        sin_a = math.sin(math.radians(known_theta))
        x = (ry - known_f * sin_a) / sin_b
        fx = known_f * math.cos(math.radians(known_theta)) + x * math.cos(math.radians(unknown_theta))
        fy = ry
    else:
        assert stated_forces, text
        fx = math.fsum(f * math.cos(math.radians(a)) for f, a in stated_forces)
        fy = math.fsum(f * math.sin(math.radians(a)) for f, a in stated_forces)
        if mass is not None and "acceleration" in text:
            fx, fy = fx / mass, fy / mass

    mag = math.hypot(fx, fy)
    direction = math.degrees(math.atan2(fy, fx)) % 360

    expected_mag, expected_dir = (int(v) for v in _ANSWER_PAIR_RE.fullmatch(answer_text).groups())
    assert abs(mag - expected_mag) < 0.51, (mag, expected_mag, text)
    diff = abs(direction - expected_dir) % 360
    diff = min(diff, 360 - diff)
    assert diff < 0.51, (direction, expected_dir, text)


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_reresolves_every_rendered_question(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        _oracle_resultant(q.text, q.answer_text)


# =========================================================================== Regression guard: the
# hard tier's three-force resultant divided by mass must never produce an implausible acceleration
# for an unqualified "body". Proved RED against the pre-fix generator (6/200 hard-tier seeds over 50 m/s^2, up to
# 93.0, via a scratch copy of that generator loaded into an isolated
# registry, never a checkout/stash) before this fix; green here, 200 seeds.
# ===========================================================================


def test_hard_tier_acceleration_magnitude_is_bounded():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        mag, _direction = (int(v) for v in _ANSWER_PAIR_RE.fullmatch(q.answer_text).groups())
        assert 2 <= mag <= 50, (mag, seed, q.text)  # Lower bound too
