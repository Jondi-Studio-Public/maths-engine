"""Regression tests for the bug-hunt 2026-10-19 student-text fixes (early primary, statistics, measurement).

Every test renders through `display_question`, the text a learner reads, over several seeds.
"""

import re
from fractions import Fraction

import pytest

from mathgen import engine
from mathgen.core import article_for_number, display_question, equals_chain, exact_decimal_text

_SKILLS = None


def _skill(skill_id):
    global _SKILLS
    if _SKILLS is None:
        _SKILLS = {s.id: s for s in engine.load_skills()[1]}
    return _SKILLS[skill_id]


def _shown(skill_id, source, seeds):
    skill = _skill(skill_id)
    for seed in seeds:
        yield display_question(engine.make_for_skill(skill, seed, source=source))


def _blobs(d):
    out = [d["text"], d["answer_text"], *d["options"], *d["working"]]
    for p in d.get("parts", []):
        out += [p["text"], *p["options"], *p["working"]]
    return out


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Pack A: 3/4 = 3/4.", "Pack A: 3/4."),
        ("so 1/18 = 1/18", "so 1/18"),
        ("(c) 20/31 = 20/31.", "(c) 20/31."),
        ("6000 - 3000 = 3000 g", "6000 - 3000 = 3000 g"),
        ("x = 5 = 5.5", "x = 5 = 5.5"),
    ],
)
def test_repeated_step_collapses_only_when_identical(text, expected):
    from mathgen.core import _learner_text

    assert _learner_text(text, working=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Tie -- the same price", "Tie – the same price"),
        ("volume in cm^3", "volume in cm³"),
        ("16 cm2 of tiles", "16 cm² of tiles"),
        ("(1/3) x 4 x 5", "(1/3) × 4 × 5"),
        ("Find 2 x (x+1)", "Find 2 x (x+1)"),
    ],
)
def test_learner_text_typography(text, expected):
    from mathgen.core import _learner_text

    assert _learner_text(text) == expected


def test_core_helpers():
    assert article_for_number(8) == "an" and article_for_number(11) == "an" and article_for_number(7) == "a"
    assert equals_chain("19/35", "19/35") == "19/35"
    assert equals_chain("1 + 2", 3) == "1 + 2 = 3"
    assert exact_decimal_text(Fraction(1, 8)) == "0.125"


@pytest.mark.parametrize("source", [0, 1, 2])
def test_unit_price_working_is_decimal_cents(source):
    for d in _shown("compare-unit-prices", source, range(25)):
        for line in d["working"]:
            assert not re.search(r"\d{4,}/\d+", line), line


def test_take_one_part_is_singular():
    for source in (0, 1, 2):
        for d in _shown("fraction-of-quantity", source, range(60)):
            assert not any("Take 1 parts" in w for w in d["working"])


def test_expected_value_working_shows_the_sum():
    for d in _shown("discrete-expected-value", 1, range(30)):
        (line,) = d["working"]
        assert line.startswith("E[X] = (") and "×" in line
        assert not re.search(r"(\S+) = \1\.$", line), line


def test_compound_interest_working_has_no_float_noise():
    for d in _shown("senior-use-technology-to-investigate-the-effect-of-the", 3, range(30)):
        for blob in _blobs(d):
            assert not re.search(r"\d\.\d{9,}", blob), blob


def test_sample_sd_working_is_rounded():
    for d in _shown("senior-simulate-repeated-random-sampling-from-a-variety-of-acmsm139", 1, range(30)):
        for blob in _blobs(d):
            assert not re.search(r"\d\.\d{9,}", blob), blob


def test_money_change_options_have_consistent_decimals():
    for d in _shown("money-change", 0, range(60)):
        assert all(re.fullmatch(r"\d+\.\d\d", o) for o in d["options"]), d["options"]


def test_probability_distractors_are_probabilities():
    for sid in ("finite-sample-space-probability", "simple-event-probability"):
        for source in (0,) if sid.startswith("simple") else (0, 1, 2):
            for seed in range(40):
                q = engine.make_for_skill(_skill(sid), seed, source=source)
                for opt in q.distractors:
                    assert 0 < Fraction(opt) < 1, (sid, seed, opt)


def test_choose_display_working_is_plain_prose():
    for d in _shown("senior-determine-which-type-of-graph-is-best-used", 2, range(40)):
        joined = " ".join(_blobs(d))
        assert "Data class" not in joined and "compatibility matrix" not in joined and "m2" not in joined


def test_circle_area_easy_states_rounding():
    for d in _shown("circle-area-to-diameter-and-circumference", 0, range(20)):
        assert "decimal place" in d["text"]


def test_no_ascii_dash_in_lane_text():
    for sid, source in [("compare-unit-prices", 0), ("compare-unit-prices", 2)]:
        for d in _shown(sid, source, range(20)):
            assert not any(" -- " in b for b in _blobs(d))
