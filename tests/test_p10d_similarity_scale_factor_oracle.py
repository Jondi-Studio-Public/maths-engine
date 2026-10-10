"""Independent oracle for the `p10d_similarity_scale_factor` questions: it re-derives each answer from the
rendered question text.

Parses `question.text` with its own regexes and re-derives the answer from scratch -- it must NOT import
anything from `mathgen.generators.p10d_similarity_scale_factor_build`, not a helper, not a constant, not the
fixed MCQ option strings (those are copied here as literals because they are part of the skill's public
contract, the same way `mcq_options` is part of what a learner is given -- not a re-use of generator
internals).
"""

import math
import re
from fractions import Fraction

import pytest

import mathgen

CLASSIFY_TOPIC = "similarity_conditions_classify"
LENGTH_TOPIC = "similar_figure_scale_factor_length"

OPTION_AAA = "Similar by AAA (all pairs of corresponding angles are equal)"
OPTION_SSS = "Similar by SSS (all three pairs of corresponding sides are in the same ratio)"
OPTION_SAS = (
    "Similar by SAS (one pair of corresponding angles is equal and the sides that include it are in the same ratio)"
)
OPTION_NOT_SIMILAR = "Not similar (the given information does not establish similarity)"


# ============================================================================
# Skill 1 oracle: similarity_conditions_classify
# ============================================================================


def _classify_aaa(text: str) -> str:
    a1, a2, d1, d2 = (int(x) for x in re.findall(r"= (\d+) degrees", text))
    a3 = 180 - a1 - a2
    f3 = 180 - d1 - d2
    similar = sorted([a1, a2, a3]) == sorted([d1, d2, f3])
    return OPTION_AAA if similar else OPTION_NOT_SIMILAR


def _classify_sss(text: str) -> str:
    sides = [int(x) for x in re.findall(r"= (\d+) cm", text)]
    abc, defs = sorted(sides[:3]), sorted(sides[3:])
    ratios = {Fraction(d, a) for d, a in zip(defs, abc)}
    similar = len(ratios) == 1
    return OPTION_SSS if similar else OPTION_NOT_SIMILAR


def _classify_sas(text: str) -> str:
    side_nums = [int(x) for x in re.findall(r"(\d+) cm", text)]
    angle_nums = [int(x) for x in re.findall(r"= (\d+) degrees", text)]
    a, b = side_nums[0], side_nums[1]
    d, e = side_nums[2], side_nums[3]
    angle_c, angle_f = angle_nums

    def third_side(x, y, angle_deg):
        c2 = x * x + y * y - 2 * x * y * math.cos(math.radians(angle_deg))
        return math.sqrt(max(c2, 0.0))

    c = third_side(a, b, angle_c)
    f = third_side(d, e, angle_f)
    abc_sorted = sorted([a, b, c])
    def_sorted = sorted([d, e, f])
    ratios = [x / y for x, y in zip(def_sorted, abc_sorted)]
    similar = (max(ratios) - min(ratios)) <= 1e-6 * max(ratios)
    return OPTION_SAS if similar else OPTION_NOT_SIMILAR


_CLASSIFIERS = {"easy": _classify_aaa, "medium": _classify_sss, "hard": _classify_sas}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(400))
def test_classify_oracle_matches_across_full_draw_space(difficulty, seed):
    q = mathgen.make(CLASSIFY_TOPIC, seed=seed, difficulty=difficulty)
    expected = _CLASSIFIERS[difficulty](q.text)
    assert expected == q.answer_text, (
        f"oracle mismatch {difficulty} seed={seed}: expected {expected!r}, got {q.answer_text!r}\nQuestion: {q.text}"
    )


# ============================================================================
# Skill 2 oracle: similar_figure_scale_factor_length
# ============================================================================


def _oracle_easy_length(text: str) -> Fraction:
    ratio_match = re.search(r"scale factor from Figure A to Figure B is (\d+)/(\d+)", text)
    side_match = re.search(r"A side of Figure A measures (\d+) cm", text)
    assert ratio_match and side_match, f"cannot parse easy: {text}"
    p, q = int(ratio_match.group(1)), int(ratio_match.group(2))
    side_a = int(side_match.group(1))
    return Fraction(side_a) * Fraction(p, q)


def _oracle_medium_length(text: str) -> Fraction:
    pair_match = re.search(
        r"A side of Figure A measures (\d+) cm; the corresponding side of Figure B measures (\d+) cm", text
    )
    second_match = re.search(r"A second side of Figure A measures (\d+) cm", text)
    assert pair_match and second_match, f"cannot parse medium: {text}"
    length_a1, length_b1 = int(pair_match.group(1)), int(pair_match.group(2))
    length_a2 = int(second_match.group(1))
    k = Fraction(length_b1, length_a1)
    return Fraction(length_a2) * k


def _oracle_hard_length(text: str) -> Fraction:
    pair_match = re.search(
        r"A side of Figure A measures (\d+) cm; the corresponding side of Figure B measures (\d+) cm", text
    )
    perimeter_match = re.search(r"Figure A has a perimeter of (\d+) cm", text)
    assert pair_match and perimeter_match, f"cannot parse hard: {text}"
    length_a1, length_b1 = int(pair_match.group(1)), int(pair_match.group(2))
    perimeter_a = int(perimeter_match.group(1))
    k = Fraction(length_b1, length_a1)
    return Fraction(perimeter_a) * k


_LENGTH_ORACLES = {"easy": _oracle_easy_length, "medium": _oracle_medium_length, "hard": _oracle_hard_length}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(200))
def test_length_oracle_matches(difficulty, seed):
    q = mathgen.make(LENGTH_TOPIC, seed=seed, difficulty=difficulty)
    expected = _LENGTH_ORACLES[difficulty](q.text)
    actual = Fraction(q.answer_text)
    assert expected == actual, (
        f"oracle mismatch {difficulty} seed={seed}: expected {expected}, got {actual}\nQuestion: {q.text}"
    )
    for d in q.distractors:
        assert Fraction(d) != expected, f"distractor matches expected {difficulty} seed={seed}: {d}\nQuestion: {q.text}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(200))
def test_length_oracle_never_sees_area_or_volume_words(difficulty, seed):
    """The oracle's own read of the boundary: if the generator ever slipped an
    area/volume word into the text, no amount of correct arithmetic saves it."""
    q = mathgen.make(LENGTH_TOPIC, seed=seed, difficulty=difficulty)
    low = q.text.lower()
    assert "area" not in low and "volume" not in low
