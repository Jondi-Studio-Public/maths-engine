"""Grading fixes from the round-2 bug hunt: replies that the question's wording rules out are refused, and
plausible spellings of a right answer are accepted. Every case renders the skill's own question over several
seeds and grades replies built from that question's key (see `mathgen/grading/stem_checks.py`)."""

from __future__ import annotations

import re
from functools import lru_cache

import pytest

from mathgen import make
from mathgen.grading import typed
from mathgen.skills import load_skills

SEEDS = range(4)


@lru_cache(maxsize=1)
def _skills():
    return {skill.id: skill for skill in load_skills()[1]}


def _question(skill_id: str, source: int, seed: int):
    skill = _skills()[skill_id]
    src = skill.sources[source]
    params = {k: (tuple(x) if isinstance(x, list) else x) for k, x in (src.params or {}).items()}
    return skill, make(src.topic, seed=seed, difficulty=src.difficulty, **params)


def _ok(skill, question, reply: str) -> bool:
    return bool(typed.grade(skill.answer, question, reply, skill=skill).correct)


def _value(question) -> str:
    return re.search(r"-?\d+(?:\.\d+)?", question.answer_text.replace(",", "")).group(0)


# -- D1: angle units follow the stem, not only a degree sign in the key --------------------------------------------

DEGREE_SKILLS = [
    ("f10-enlargement-invariants", 0),
    ("f10-geometric-construction-algorithm", 1),
    ("angle-radians-to-degrees-decimal", 0),
    ("angle-radians-to-degrees-decimal", 3),
]


@pytest.mark.parametrize("skill_id,source", DEGREE_SKILLS)
@pytest.mark.parametrize("seed", SEEDS)
def test_degree_answers_refuse_other_units(skill_id, source, seed):
    skill, question = _question(skill_id, source, seed)
    if not re.search(r"(?:in|to) degrees", question.text):
        pytest.skip("this seed asks for something other than an angle")
    value = _value(question)
    for good in (value, f"{value}°", f"{value} degrees", f"{value} deg", f"x = {value}"):
        assert _ok(skill, question, good), good
    for bad in ("rad", "radians", "turns", "grad", "π", "pi", "cm", "m", "kg", "°C", "cm²"):
        assert not _ok(skill, question, f"{value} {bad}"), bad
        assert not _ok(skill, question, f"{value}{bad}"), bad
    assert not _ok(skill, question, f"${value}")


@pytest.mark.parametrize(
    "skill_id,source", [("angle-degrees-to-radians-decimal", 0), ("angle-degrees-to-radians-decimal", 2)]
)
@pytest.mark.parametrize("seed", SEEDS)
def test_radian_answers_refuse_degrees(skill_id, source, seed):
    skill, question = _question(skill_id, source, seed)
    value = _value(question)
    for good in (value, f"{value} rad", f"{value} radians"):
        assert _ok(skill, question, good), good
    for bad in ("°", " degrees", " deg", "π", " pi", " turns", " cm"):
        assert not _ok(skill, question, f"{value}{bad}"), bad


@pytest.mark.parametrize("seed", SEEDS)
def test_radius_in_cm_refuses_degrees_and_radians(seed):
    skill, question = _question("senior-calculate-lengths-of-arcs-and-areas-of-sectors", 2, seed)
    value = _value(question)
    assert _ok(skill, question, value) and _ok(skill, question, f"{value} cm")
    for bad in ("degrees", "rad", "radians"):
        assert not _ok(skill, question, f"{value} {bad}"), bad


@pytest.mark.parametrize("seed", SEEDS)
def test_angle_pi_pairs_with_the_degree_key_only_refused(seed):
    skill, question = _question("angle-degrees-radians-pi-form", 0, seed)
    assert _ok(skill, question, question.answer_text)
    assert _ok(skill, question, question.answer_text.replace("pi", "π"))


# -- D8: pi after a decimal ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id,source", [("cone-volume", 0), ("f10-circle-measures", 0)])
@pytest.mark.parametrize("seed", SEEDS)
def test_pi_after_a_decimal_is_refused(skill_id, source, seed):
    skill, question = _question(skill_id, source, seed)
    value = _value(question)
    assert _ok(skill, question, value)
    for bad in ("π", " π", " pi", "pi"):
        assert not _ok(skill, question, f"{value}{bad}"), bad


# -- D2: the unit named in a sentence that does not open with a verb ---------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_using_pi_find_volume_in_cm3(seed):
    skill, question = _question("cone-volume", 0, seed)
    value = _value(question)
    for good in (value, f"{value} cm³", f"{value} cm^3", f"{value} cubic centimetres"):
        assert _ok(skill, question, good), good
    for bad in ("cm", "cm²", "m³", "mL", "kg"):
        assert not _ok(skill, question, f"{value} {bad}"), bad


@pytest.mark.parametrize(
    "skill_id,source,good,bad",
    [
        ("senior-when-a-secant-meeting-the-circle-at-a", 0, ("cm", "centimetres"), ("m", "km", "mm")),
        ("senior-use-addition-of-the-area-of-the-faces", 0, ("cm²", "cm^2"), ("m²", "mm²", "cm", "kg")),
        ("pythagorean-leg", 1, ("mm",), ("kg", "dollars", "cm", "m")),
        ("pythagorean-hypotenuse", 1, ("mm",), ("kg", "cm", "m")),
        ("convert-metric-mass", 0, ("kg",), ("t", "g")),
    ],
)
@pytest.mark.parametrize("seed", SEEDS)
def test_unit_named_outside_an_asking_sentence(skill_id, source, good, bad, seed):
    skill, question = _question(skill_id, source, seed)
    value = _value(question)
    if skill_id == "convert-metric-mass" and not question.text.rstrip(".").endswith("to kg"):
        pytest.skip("this seed converts the other way")
    if skill_id == "pythagorean-leg" and "millimetres" not in question.text:
        pytest.skip("this tier states its lengths in another unit")
    if skill_id == "pythagorean-hypotenuse" and "millimetres" not in question.text:
        pytest.skip("this tier states its lengths in another unit")
    assert _ok(skill, question, value)
    for unit in good:
        assert _ok(skill, question, f"{value} {unit}"), unit
    for unit in bad:
        assert not _ok(skill, question, f"{value} {unit}"), unit


# -- D3: currency --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("seed", SEEDS)
def test_currency_conversion_refuses_the_other_currency(source, seed):
    skill, question = _question("senior-use-currency-exchange-rates-to-determine-the-cost", source, seed)
    value = _value(question)
    for good in (value, f"A${value}", f"${value}", f"{value} AUD", f"{value} dollars", question.answer_text):
        assert _ok(skill, question, good), good
    for bad in (f"US${value}", f"{value} USD", f"£{value}", f"{value} pounds", f"€{value}", f"{value} euros"):
        assert not _ok(skill, question, bad), bad


# -- D4: the question copied back, uncollected terms ---------------------------------------------------------------


def _stem_expression(question) -> str:
    return re.sub(r"^(?:Expand and simplify|Simplify|Expand)\s+", "", question.text).split(", giving")[0].rstrip(". ")


@pytest.mark.parametrize(
    "skill_id,source",
    [
        ("combine-like-terms", 0),
        ("combine-like-terms", 2),
        ("f10-exponent-laws-integer-variable", 0),
        ("expand-binomial-trinomial", 0),
        ("simplify-bracketed-expressions", 0),
    ],
)
@pytest.mark.parametrize("seed", SEEDS)
def test_simplify_refuses_the_stem_and_keeps_the_key(skill_id, source, seed):
    skill, question = _question(skill_id, source, seed)
    assert _ok(skill, question, question.answer_text)
    assert not _ok(skill, question, _stem_expression(question))
    assert not _ok(skill, question, _stem_expression(question).replace("×", "*"))


@pytest.mark.parametrize("seed", SEEDS)
def test_uncollected_terms_are_refused(seed):
    skill, question = _question("simplify-bracketed-expressions", 0, seed)
    key = question.answer_text.replace("*", "")
    assert _ok(skill, question, key)
    assert not _ok(skill, question, key + " + 0")
    assert not _ok(skill, question, key + " * 1")
    assert not _ok(skill, question, key.replace(" ", "") + "+0")


def test_uncollected_terms_literal_cases():
    skill, question = _question("combine-like-terms", 0, 0)
    assert question.answer_text == "-x"
    for good in ("-x", "- x", "-1x"):
        assert _ok(skill, question, good), good
    for bad in ("-2x - x + 2x", "-3x+2x", "x-2x", "-x-x+x", "-x+0"):
        assert not _ok(skill, question, bad), bad
    skill, question = _question("f10-exponent-laws-integer-variable", 0, 0)
    assert question.answer_text == "x^7"
    assert _ok(skill, question, "x^7")
    for bad in ("x^8 × x^(-1)", "x^8*x^-1", "x^8/x", "x^(5+2)", "x^(8-1)", "x^(14/2)"):
        assert not _ok(skill, question, bad), bad
    skill, question = _question("simplify-bracketed-expressions", 0, 0)
    assert question.answer_text.replace("*", "") == "74x - 45"
    for bad in ("72x+2x-45", "74x-45+0", "74x - 90/2", "74x - 45*1"):
        assert not _ok(skill, question, bad), bad
    for good in ("74x - 45", "74x-45", "-45 + 74x", "74*x - 45"):
        assert _ok(skill, question, good), good


# -- D12: a sum of fractions when the stem does not ask for one fraction -------------------------------------------


@pytest.mark.parametrize("source", [0, 1, 2])
@pytest.mark.parametrize("seed", SEEDS)
def test_sum_of_fractions_accepted_unless_a_single_fraction_is_asked(source, seed):
    skill, question = _question("algebraic-fraction-add-subtract", source, seed)
    if "single fraction" in question.text:
        pytest.skip("this stem asks for a single fraction")
    assert _ok(skill, question, question.answer_text)
    assert not _ok(skill, question, _stem_expression(question))


def test_split_fraction_accepted_and_refused_by_the_stem():
    skill, question = _question("algebraic-fraction-add-subtract", 0, 0)
    assert question.answer_text == "(61x + 72)/35"
    assert _ok(skill, question, "61x/35 + 72/35")
    assert _ok(skill, question, "(61/35)x + 72/35")
    assert not _ok(skill, question, "8x/5 + 6/5 + x/7 + 6/7")
    skill, question = _question("algebraic-fraction-add-subtract", 3, 0)
    assert "single fraction" in question.text
    assert _ok(skill, question, question.answer_text)


# -- D5: thousands separators on an expression key ---------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,source,seed",
    [
        ("senior-use-formulas-to-find-the-volume-of-pyramids", 0, 1),
        ("complex-polygon-vertices-and-roots-of-unity", 2, 1),
    ],
)
def test_expression_key_accepts_thousands_separators(skill_id, source, seed):
    skill, question = _question(skill_id, source, seed)
    value = int(_value(question))
    assert value >= 1000
    grouped = f"{value:,}"
    for good in (str(value), grouped, grouped.replace(",", " "), f"{grouped}.00"):
        assert _ok(skill, question, good), good
    assert not _ok(skill, question, f"{value + 1:,}")


# -- D6: a repeated root typed twice -------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_repeated_root_typed_twice(seed):
    skill, question = _question("solve-quadratics-factorable", 0, seed)
    if "," in question.answer_text or " or " in question.answer_text:
        pytest.skip("this equation has two roots")
    value = _value(question)
    for good in (f"x = {value}", value, f"x = {value} or x = {value}", f"x = {value}, {value}", f"{value}, {value}"):
        assert _ok(skill, question, good), good
    assert not _ok(skill, question, f"{value}, {int(float(value)) + 1}")


# -- D7: dollar sign for cents -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,source",
    [("f10-money-dollars-cents-representation", 0), ("f10-money-count-collection", 0)],
)
@pytest.mark.parametrize("seed", SEEDS)
def test_dollar_sign_is_refused_for_cents(skill_id, source, seed):
    skill, question = _question(skill_id, source, seed)
    if not re.search(r"(?:in|to) cents|how many cents", question.text, re.I):
        pytest.skip("this question is not asked in cents")
    value = _value(question)
    for good in (value, f"{value}c", f"{value} cents", f"{value}¢"):
        assert _ok(skill, question, good), good
    for bad in (f"${value}", f"${value}.00"):
        assert not _ok(skill, question, bad), bad


# -- D9: a percent sign on an answer that is not a percentage ------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,source",
    [
        ("distance-two-points", 0),
        ("senior-use-factorial-notation", 0),
        ("senior-determine-the-overall-change-in-a-quantity-following", 0),
    ],
)
@pytest.mark.parametrize("seed", SEEDS)
def test_percent_sign_is_refused_when_no_percentage_is_asked(skill_id, source, seed):
    skill, question = _question(skill_id, source, seed)
    value = question.answer_text
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        pytest.skip("this seed has an exact, non-decimal key")
    assert _ok(skill, question, value)
    for bad in (f"{value}%", f"{value} %", f"{value} percent"):
        assert not _ok(skill, question, bad), bad


def test_percent_sign_still_accepted_when_a_percentage_is_asked():
    skill, question = _question("senior-determine-the-overall-change-in-a-quantity-following", 0, 0)
    assert "percent" not in question.text.split("?")[-2].lower()
    for sid, src in (("f10-percentage-of-amount", 0),):
        if sid not in _skills():
            continue
        skill, question = _question(sid, src, 0)
        assert _ok(skill, question, question.answer_text)


# -- D10: a percent sign on every item of a list --------------------------------------------------------------------


@pytest.mark.parametrize("source", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", SEEDS)
def test_percent_sign_on_each_list_item(source, seed):
    skill, question = _question("senior-calculate-the-effective-annual-rate-of-interest-and", source, seed)
    items = [part.strip() for part in question.answer_text.split(",")]
    assert _ok(skill, question, ", ".join(items))
    assert _ok(skill, question, ", ".join(item + "%" for item in items))
    assert not _ok(skill, question, ", ".join(str(float(item) + 1) for item in items))


# -- D11: a mixed numeral ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,source",
    [("f10-mixed-numerals-number-line", 0), ("f10-mixed-numerals-number-line", 1), ("f10-written-division", 1)],
)
@pytest.mark.parametrize("seed", SEEDS)
def test_mixed_numeral_is_asked_for(skill_id, source, seed):
    skill, question = _question(skill_id, source, seed)
    if "mixed numeral" not in question.text:
        pytest.skip("not asked as a mixed numeral")
    whole, fraction = question.answer_text.split()
    numerator, denominator = map(int, fraction.split("/"))
    improper = f"{int(whole) * denominator + numerator}/{denominator}"
    decimal = f"{int(whole) + numerator / denominator}"
    unreduced = f"{whole} {numerator * 2}/{denominator * 2}"
    for good in (question.answer_text, f"{whole} and {fraction}", f"{whole}+{fraction}", f"{whole} + {fraction}"):
        assert _ok(skill, question, good), good
    for bad in (improper, decimal, unreduced):
        assert not _ok(skill, question, bad), bad


# -- D14: spaced thousands in a list, and the rads spelling --------------------------------------------------------


def test_spaced_thousands_inside_a_list():
    skill, question = _question("f10-large-number-naming", 0, 0)
    numbers = re.findall(r"\d{1,3}(?:,\d{3})+", question.answer_text)
    assert len(numbers) >= 2
    assert _ok(skill, question, ", ".join(numbers))
    assert _ok(skill, question, ", ".join(n.replace(",", " ") for n in numbers))


@pytest.mark.parametrize("reply", ["π/4", "pi/4", "¼π", "pi over 4", "π/4 rad", "π/4 rads", "π/4 radians"])
def test_pi_over_four_spellings(reply):
    skill, question = _question("angle-degrees-radians-pi-form", 0, 0)
    assert question.answer_text == "pi/4"
    assert _ok(skill, question, reply)


# -- r2-c3: pair equation with explicit multiplication, lower-case set letters ----------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pair_equation_accepts_explicit_multiplication(seed):
    skill, question = _question("equation-from-table", 0, seed)
    key = question.answer_text
    assert _ok(skill, question, key)
    assert _ok(skill, question, re.sub(r"(\d)x", r"\1*x", key))
    assert _ok(skill, question, re.sub(r"(\d)x", r"\1·x", key))
    assert not _ok(skill, question, "y = 9.5*x + 400")


@pytest.mark.parametrize("seed", SEEDS)
def test_set_of_letters_accepts_lower_case(seed):
    skill, question = _question("f10-graphical-inequality-regions", 0, seed)
    letters = re.findall(r"[A-Z]", question.answer_text)
    assert letters
    assert _ok(skill, question, question.answer_text)
    assert _ok(skill, question, "{" + ", ".join(letters).lower() + "}")
    assert _ok(skill, question, ", ".join(reversed(letters)).lower())
    assert not _ok(skill, question, "{" + ", ".join(letters[1:]).lower() + "}")


# -- D2: a unit followed by a bracket ("in litres (L)", "in cm² (1 d.p.)") is still the unit asked ---------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_unit_followed_by_a_bracketed_note(seed):
    skill, question = _question("prism-capacity-metric-units", 0, seed)
    value = _value(question)
    assert re.search(r"in litres \(L\)", question.text)
    for good in (value, f"{value} L", f"{value} litres"):
        assert _ok(skill, question, good), good
    for bad in ("cm", "cm²", "cm³", "mL", "mm²"):
        assert not _ok(skill, question, f"{value} {bad}"), bad
