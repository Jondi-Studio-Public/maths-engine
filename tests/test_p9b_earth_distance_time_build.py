"""Earth distance along a meridian (ACMEM160) and longitude-based solar-time difference (ACMEM165)."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, stated_precision_tolerance

SEEDS = range(40)
TOPIC_1 = "earth_meridian_distance"
TOPIC_2 = "earth_longitude_time_difference"
DIFFICULTIES = ["easy", "medium", "hard"]

# (topic, params) per difficulty, matching each skill's own `sources` entries
# -- generating through the skill's own params, never `make(topic, seed,
# difficulty=tier)` alone (a params-driven generator hands back its default
# variant for every tier otherwise).
SKILL_1_PARAMS = {
    "easy": dict(mode="same_hemisphere"),
    "medium": dict(mode="opposite_hemisphere"),
    "hard": dict(mode="combined_km"),
}
SKILL_2_PARAMS = {
    "easy": dict(mode="same_side"),
    "medium": dict(mode="opposite_side"),
    "hard": dict(mode="inverse"),
}


def _make1(seed, difficulty):
    return make(TOPIC_1, seed=seed, difficulty=difficulty, **SKILL_1_PARAMS[difficulty])


def _make2(seed, difficulty):
    return make(TOPIC_2, seed=seed, difficulty=difficulty, **SKILL_2_PARAMS[difficulty])


MAKERS = {TOPIC_1: _make1, TOPIC_2: _make2}


# --- determinism / distractor hygiene ----------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    maker = MAKERS[topic]
    for seed in SEEDS:
        question = maker(seed, difficulty)
        repeat = maker(seed, difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    maker = MAKERS[topic]
    for seed in SEEDS:
        question = maker(seed, difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_text_never_uses_comma_separator(topic, difficulty):
    maker = MAKERS[topic]
    for seed in SEEDS:
        question = maker(seed, difficulty)
        assert "," not in question.answer_text


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_correct_type(topic, difficulty):
    maker = MAKERS[topic]
    for seed in SEEDS:
        question = maker(seed, difficulty)
        assert isinstance(question.answer, (int, float))
        assert not isinstance(question.answer, bool)


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_formatted_alike(topic, difficulty):
    """No option is identifiable by its shape (decimal point present or not)."""
    maker = MAKERS[topic]
    for seed in SEEDS:
        question = maker(seed, difficulty)
        is_answer_decimal = "." in question.answer_text
        for distractor in question.distractors:
            is_distractor_decimal = "." in distractor
            assert is_answer_decimal == is_distractor_decimal, (
                f"{topic}/{difficulty} seed {seed}: answer '{question.answer_text}' "
                f"format mismatch with distractor '{distractor}'"
            )


# Pool-size disclosure.
#
# Skill 1 easy/medium/hard draw from fixed REAL, well-known place-pair tables (11 same-hemisphere,
# 10 opposite-hemisphere, 21 combined for hard); skill 2 easy/medium draw from fixed tables too (15
# same-side, 12 opposite-side pairs); hard is generative (36 places x 8 Δt choices x up to 2
# directions), not a fixed pool. A fixed pool of N rows drawn with replacement over 60 seeds is NOT
# guaranteed to show all N distinct texts (coupon-collector variance), so this asserts a RANGE -- at
# least the shared _MIN_DISTINCT=10 floor (proving the widening actually worked) and at most the
# table's own row count read live from the generator module (never hardcoded, so it tracks the table
# automatically) -- rather than an exact count, which would re-pin the pools by this test run's RNG
# draws instead of by genuine table size.
_MIN_DISTINCT_LOCAL = 10


def _table_size(topic, difficulty):
    import mathgen.generators.p9b_earth_distance_time_build as g

    if topic == TOPIC_1:
        return {
            "easy": len(g.SAME_HEMISPHERE),
            "medium": len(g.OPPOSITE_HEMISPHERE),
            "hard": len(g.SAME_HEMISPHERE) + len(g.OPPOSITE_HEMISPHERE),
        }[difficulty]
    return {
        "easy": len(g.SAME_SIDE_PAIRS),
        "medium": len(g.OPPOSITE_SIDE_PAIRS),
    }[difficulty]


@pytest.mark.parametrize(
    "topic,difficulty",
    [
        (TOPIC_1, "easy"),
        (TOPIC_1, "medium"),
        (TOPIC_1, "hard"),
        (TOPIC_2, "easy"),
        (TOPIC_2, "medium"),
    ],
)
def test_disclosed_pool_size_over_60_seeds(topic, difficulty):
    maker = MAKERS[topic]
    questions_rendered = {maker(seed, difficulty).text for seed in range(60)}
    distinct = len(questions_rendered)
    table_size = _table_size(topic, difficulty)
    assert _MIN_DISTINCT_LOCAL <= distinct <= table_size, (
        f"{topic}/{difficulty}: {distinct} distinct texts over 60 seeds, "
        f"expected between _MIN_DISTINCT={_MIN_DISTINCT_LOCAL} and the "
        f"table size {table_size}"
    )


def test_skill_2_hard_tier_at_least_20_distinct_over_60_seeds():
    """Skill 2's hard (inverse) tier is generative, not a fixed pool --
    holds the >=20-distinct bar directly."""
    questions_rendered = {_make2(seed, "hard").text for seed in range(60)}
    assert len(questions_rendered) >= 20


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    maker = MAKERS[topic]
    for seed in SEEDS:
        question = maker(seed, difficulty)
        assert grade_number(question, question.answer_text).correct, (
            f"{topic}/{difficulty} seed {seed}: real grader rejected its own canonical "
            f"answer_text {question.answer_text!r}"
        )
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct, (
                f"{topic}/{difficulty} seed {seed}: real grader accepted distractor {distractor!r} as correct"
            )


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    maker = MAKERS[topic]
    for seed in SEEDS:
        question = maker(seed, difficulty)
        if "." in question.answer_text:
            dp = len(question.answer_text.split(".")[1])
            offset = Fraction(1, 10**dp) * 5  # clears the stated-precision tolerance
        else:
            offset = 1
        wrong = Fraction(question.answer_text) + offset
        assert not grade_number(question, str(float(wrong))).correct


# --- decimal-answer stated-precision tolerance check -------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_decimal_answers_meet_stated_precision_tolerance(topic, difficulty):
    """Every distractor differs from the answer by more than half a unit in
    the LAST decimal place answer_text states."""
    maker = MAKERS[topic]
    for seed in SEEDS:
        question = maker(seed, difficulty)
        answer_text = question.answer_text
        if "." not in answer_text:
            tol = Fraction(1, 2)  # integer canonical: grader requires an exact match,
            # (stated_precision_tolerance(0) would be wrong here -- its own
            # docstring says dp=0 is NOT the real integer tolerance -- this is
            # a looser but sufficient margin for the distractor check)
        else:
            places = len(answer_text.split(".")[1])
            tol = stated_precision_tolerance(places)
        answer_frac = Fraction(answer_text)
        for distractor in question.distractors:
            distractor_frac = Fraction(distractor)
            gap = abs(distractor_frac - answer_frac)
            assert gap > tol, (
                f"{topic}/{difficulty} seed {seed}: distractor '{distractor}' too close "
                f"to answer '{answer_text}' (gap={gap}, tol={tol})"
            )


# --- independent-check-in-generator fires on corruption ---------------------


def test_earth_meridian_distance_verify_nm_fires_on_corrupted_total():
    from mathgen.generators.p9b_earth_distance_time_build import _verify_nm_via_decimal_degrees

    with pytest.raises(VerificationError):
        _verify_nm_via_decimal_degrees(37, 48, 16, 54, True, 9999)


def test_earth_meridian_distance_verify_km_fires_on_corrupted_total():
    from mathgen.generators.p9b_earth_distance_time_build import _verify_km_from_nm

    with pytest.raises(VerificationError):
        _verify_km_from_nm(1254, 1)


def test_earth_longitude_time_difference_verify_delta_fires_on_corrupted_total():
    from mathgen.generators.p9b_earth_distance_time_build import _verify_delta_lon

    with pytest.raises(VerificationError):
        _verify_delta_lon(Fraction(1512, 10), Fraction(1748, 10), Fraction(999))


def test_earth_longitude_time_difference_verify_inverse_fires_on_corrupted_dt():
    from mathgen.generators.p9b_earth_distance_time_build import _verify_inverse

    with pytest.raises(VerificationError):
        _verify_inverse(Fraction(1512, 10), Fraction(1897, 10), Fraction(1))


def _real_longitudes():
    """Real (approximate, well-known, 1 d.p.) longitude for every place in
    skill 1's tables -- distinct from the generator's single shared display
    string per pair, and NOT read from the generator module. This is the
    data-quality fact the plausibility test checks: that the two REAL places
    behind each displayed "both ~X° longitude" really are within 3 degrees
    of each other."""
    return {
        "Melbourne": Fraction(1449, 10),
        "Cairns": Fraction(1458, 10),
        "Hobart": Fraction(1473, 10),
        "Sydney": Fraction(1512, 10),
        "Brisbane": Fraction(1530, 10),
        "London": Fraction(-1, 10),
        "Accra": Fraction(-2, 10),
        "Adelaide": Fraction(1386, 10),
        "Tokyo": Fraction(1397, 10),
        "Perth": Fraction(1159, 10),
        "Beijing": Fraction(1164, 10),
        "Cairo": Fraction(312, 10),
        "Durban": Fraction(310, 10),
        "Lima": Fraction(-770, 10),
        "New York": Fraction(-740, 10),
        # Added for the widened SAME_HEMISPHERE/OPPOSITE_HEMISPHERE tables.
        "Paris": Fraction(23, 10),
        "Bogotá": Fraction(-741, 10),
        "Denpasar (Bali)": Fraction(1152, 10),
        "Rome": Fraction(125, 10),
        "Tripoli": Fraction(132, 10),
        "Berlin": Fraction(134, 10),
        "Toronto": Fraction(-794, 10),
        "Panama City": Fraction(-795, 10),
        "Quito": Fraction(-785, 10),
        "Ho Chi Minh City": Fraction(1066, 10),
        "Jakarta": Fraction(1068, 10),
        "Singapore": Fraction(1038, 10),
        "Maputo": Fraction(326, 10),
    }


# Widened to accept a decimal degree ("115.5°E", "79.5°W") -- the widened tables introduce displayed
# longitudes with 1 decimal place, not just whole degrees.
_LON_DISPLAY_RE = re.compile(r"^(\d+(?:\.\d+)?)°([EW]?)$")


def _parse_lon_display(lon_display: str) -> Fraction:
    """Parse the GENERATOR's own displayed longitude string ("145°E", "0°",
    "76°W", "115.5°E") -- the artefact a learner actually reads, not a value
    re-derived from this test's own lookup table."""
    match = _LON_DISPLAY_RE.match(lon_display)
    assert match, f"unparseable lon_display {lon_display!r}"
    deg, hemi = match.groups()
    value = Fraction(deg)
    return -value if hemi == "W" else value


def _plausibility_rows(same_hemisphere, opposite_hemisphere, longitudes):
    import mathgen.generators.p9b_earth_distance_time_build as g

    rows = []
    for row in same_hemisphere + opposite_hemisphere:
        place_a, deg_a, min_a, hemi_a, place_b, deg_b, min_b, hemi_b, lon_display = row
        same_hemi = hemi_a == hemi_b
        arcmin_a = g._arcmin_total(deg_a, min_a)
        arcmin_b = g._arcmin_total(deg_b, min_b)
        nm = abs(arcmin_a - arcmin_b) if same_hemi else arcmin_a + arcmin_b
        lat_mag_a = deg_a + Fraction(min_a, 60)
        lat_mag_b = deg_b + Fraction(min_b, 60)
        rows.append(
            dict(
                place_a=place_a,
                place_b=place_b,
                lat_a=lat_mag_a,
                lat_b=lat_mag_b,
                nm=nm,
                displayed_lon=_parse_lon_display(lon_display),
                real_lon_a=longitudes[place_a],
                real_lon_b=longitudes[place_b],
            )
        )
    return rows


def _assert_earth_meridian_distance_places_are_plausible(rows):
    for row in rows:
        assert row["lat_a"] < 90, row
        assert row["lat_b"] < 90, row
        assert 300 <= row["nm"] <= 4500, row
        # The GENERATOR-VISIBLE displayed longitude (what the learner reads
        # in question.text, parsed straight from the table row, not from
        # this test's own lookup) must itself be within 3 degrees of BOTH
        # real places -- so a corrupted `lon_display` in the generator's own
        # table fails here, not just a corrupted lookup in this test file.
        assert abs(row["displayed_lon"] - row["real_lon_a"]) <= 3, row
        assert abs(row["displayed_lon"] - row["real_lon_b"]) <= 3, row


def test_earth_meridian_distance_places_are_plausible_red_then_green():
    """Prove the constraint test fails against a corrupted table BEFORE proving it passes against the real one."""
    import mathgen.generators.p9b_earth_distance_time_build as g

    real_longitudes = _real_longitudes()
    real_rows = _plausibility_rows(g.SAME_HEMISPHERE, g.OPPOSITE_HEMISPHERE, real_longitudes)

    # RED: corrupt a SCRATCH COPY of the GENERATOR's own table -- Melbourne's
    # STATED longitude (the string a learner actually reads in question.text)
    # reads "45°E" against Cairns' "145°E", an 100-degree spread -- the
    # literal red scenario, applied to the generator-visible
    # datum rather than to this test's separate lookup dict.
    corrupted_same_hemisphere = list(g.SAME_HEMISPHERE)
    row0 = corrupted_same_hemisphere[0]
    assert row0[0] == "Melbourne" and row0[-1] == "145°E", row0
    corrupted_same_hemisphere[0] = row0[:-1] + ("45°E",)
    corrupted_rows = _plausibility_rows(corrupted_same_hemisphere, g.OPPOSITE_HEMISPHERE, real_longitudes)
    with pytest.raises(AssertionError):
        _assert_earth_meridian_distance_places_are_plausible(corrupted_rows)

    # GREEN: the real, uncorrupted generator table passes.
    _assert_earth_meridian_distance_places_are_plausible(real_rows)


def test_earth_longitude_time_difference_places_are_plausible():
    """No pole-adjacent longitude claim (N/A -- this skill has no latitude),
    every Δt over the whole table is one a person could actually experience
    (0 < Δt <= 12 h, since minimal circular separation never exceeds 180
    degrees = 12 hours), and no longitude is outside (-180, 180]."""
    import mathgen.generators.p9b_earth_distance_time_build as g

    for value in g.LONGITUDES.values():
        assert Fraction(-180) < value <= Fraction(180)

    for pairs in (g.SAME_SIDE_PAIRS, g.OPPOSITE_SIDE_PAIRS):
        for place_a, place_b in pairs:
            lon_a, lon_b = g.LONGITUDES[place_a], g.LONGITUDES[place_b]
            delta = g._delta_lon(lon_a, lon_b)
            dt = delta / 15
            assert 0 < dt <= 12, (place_a, place_b, dt)


# --- load through mathgen.skills --------------------------------------------------


def test_skill_1_loads_with_correct_prerequisites_and_placement():
    domains, skills = load_skills()
    skill = next(
        (s for s in skills if s.id == "senior-find-distances-between-two-places-on-earth-on"),
        None,
    )
    assert skill is not None
    assert skill.prerequisites == ["senior-calculate-lengths-of-arcs-and-areas-of-sectors"]
    assert skill.placement_eligible is False
    assert skill.answer == "number"
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}


def test_skill_2_loads_with_correct_prerequisites_and_placement():
    domains, skills = load_skills()
    skill = next(
        (s for s in skills if s.id == "senior-find-time-differences-between-two-places-on-earth"),
        None,
    )
    assert skill is not None
    assert skill.prerequisites == ["senior-use-units-of-time-conversions-between-units-fractional", "elapsed-time"]
    assert skill.placement_eligible is False
    assert skill.answer == "number"
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
