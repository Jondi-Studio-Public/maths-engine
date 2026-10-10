"""Deliberately hands `phrase`/`prompt_shape` literal strings, not generator
output, so a bug in a generator can never mask a bug in the instrument
itself (and vice versa). `scripts/measure_tiers.py` is what exercises `prompt_shape`
against real generators.
"""

from __future__ import annotations

import random

import pytest

from mathgen.core import phrase
from mathgen.slots import CONTEXTS, NAMES, OBJECTS, UNITS
from mathgen.text_quality import prompt_shape


# ---------------------------------------------------------------------------
# mathgen.slots -- the banks themselves


def test_slots_banks_meet_their_stated_minimums():
    assert len(NAMES) >= 20
    assert len(OBJECTS) >= 15
    assert len(UNITS) >= 10
    assert len(CONTEXTS) >= 10


def test_slots_banks_are_plain_string_tuples():
    for bank in (NAMES, UNITS, CONTEXTS):
        assert isinstance(bank, tuple)
        for value in bank:
            assert isinstance(value, str) and value

    assert isinstance(OBJECTS, tuple)
    for pair in OBJECTS:
        assert isinstance(pair, tuple) and len(pair) == 2
        singular, plural = pair
        assert isinstance(singular, str) and singular
        assert isinstance(plural, str) and plural
        assert singular != plural


def test_slots_bank_values_have_no_cross_bank_collisions():
    """A value living in two banks would mask ambiguously (which placeholder
    does it become?). NAMES/UNITS/CONTEXTS are disjoint from each other and
    from every OBJECTS singular/plural form."""
    object_words = {w.lower() for pair in OBJECTS for w in pair}
    name_words = {n.lower() for n in NAMES}
    unit_words = {u.lower() for u in UNITS}
    context_words = {c.lower() for c in CONTEXTS}
    assert not (object_words & name_words)
    assert not (object_words & unit_words)
    assert not (name_words & unit_words)
    # CONTEXTS are multi-word phrases ("a bakery"); only check they don't
    # collide with a whole other bank value verbatim.
    assert not (context_words & name_words)
    assert not (context_words & unit_words)
    assert not (context_words & object_words)


# ---------------------------------------------------------------------------
# mathgen.core.phrase


def test_phrase_returns_one_pattern_filled():
    rng = random.Random(0)
    result = phrase(rng, ["Hello {name}", "Hi there, {name}"], name="Amir")
    assert result in {"Hello Amir", "Hi there, Amir"}


def test_phrase_is_deterministic_for_a_given_rng_state():
    a = phrase(random.Random(42), ["A{x}", "B{x}", "C{x}"], x=1)
    b = phrase(random.Random(42), ["A{x}", "B{x}", "C{x}"], x=1)
    assert a == b


def test_phrase_covers_every_pattern_given_enough_draws():
    patterns = ["one {n}", "two {n}", "three {n}"]
    seen = {phrase(random.Random(seed), patterns, n=7) for seed in range(200)}
    assert seen == {"one 7", "two 7", "three 7"}


def test_phrase_single_pattern_always_returns_it():
    rng = random.Random(0)
    for _ in range(10):
        assert phrase(rng, ["only {x}"], x=5) == "only 5"


def test_phrase_rejects_empty_patterns():
    with pytest.raises(ValueError):
        phrase(random.Random(0), [], x=1)


def test_phrase_raises_on_a_slot_the_chosen_pattern_needs_but_was_not_given():
    with pytest.raises(KeyError):
        phrase(random.Random(0), ["needs {missing}"])


# ---------------------------------------------------------------------------
# mathgen.text_quality.prompt_shape -- digit masking


def test_prompt_shape_masks_plain_digits():
    class Q:
        text = "There are 17 apples and 3 crates."

    assert "17" not in prompt_shape(Q())
    assert "3" not in prompt_shape(Q())


def test_prompt_shape_masks_negative_numbers_as_one_token():
    class Q1:
        text = "u = (-5, 0, 1), v = (1, -4, 2). Find u + v"

    class Q2:
        text = "u = (4, 4, -2), v = (2, 0, -2). Find u + v"

    # Only the sign PATTERN differs (which position is negative) -- this is
    # bare value variation, not a structural difference, so both collapse to
    # the same shape. See mathgen.text_quality._DIGIT_RUN_RE's docstring for
    # the regression this guards (an earlier version left the sign out of
    # the token and these two showed as different shapes).
    assert prompt_shape(Q1()) == prompt_shape(Q2())


def test_prompt_shape_does_not_swallow_a_spaced_minus_operator():
    class Q:
        text = "Work out 17 - 14."

    shape = prompt_shape(Q())
    assert shape == "Work out # - #."


def test_prompt_shape_preserves_a_fraction_numerator_of_one():
    """The false-outcome this instrument was built to fix: `radicals_fractional_indices`'s easy tier ("Evaluate
    N^(1/r).") is a single root extraction; its hard tier ("Evaluate N^(m/r).", m != 1) is a root then a power
    -- a genuine escalation the mask must not collapse."""

    class Easy:
        text = "Evaluate 25^(1/2)."

    class Hard:
        text = "Evaluate 81^(3/4)."

    assert prompt_shape(Easy()) == "Evaluate #^(1/#)."
    assert prompt_shape(Hard()) == "Evaluate #^(#/#)."
    assert prompt_shape(Easy()) != prompt_shape(Hard())


def test_prompt_shape_does_not_preserve_a_bare_one_outside_a_fraction():
    """The over-broad first attempt at the fix above preserved EVERY literal
    `1`, which inflated `vector-addition` (small integer components
    routinely include 1) to ~17 false "distinct shapes" in 20 seeds. A bare
    `1` that is not a fraction numerator must mask like any other digit."""

    class Q1:
        text = "u = (1, 0, -1), v = (1, 2, -4). Find u + v"

    class Q2:
        text = "u = (4, 4, -2), v = (2, 0, -2). Find u + v"

    assert prompt_shape(Q1()) == prompt_shape(Q2())
    assert "1" not in prompt_shape(Q1())


def test_prompt_shape_carries_are_genuinely_invisible_to_the_mask():
    """ACCEPTED limitation: whether a multiplication
    forces a carry is an arithmetic property of the drawn digits, not
    something present in the rendered text at all, so two questions with
    a different digit-count AND a different carry requirement still collapse
    to the same shape. This is by construction, not a bug to fix here."""

    class Easy:
        text = "Work out 81 × 4."

    class Hard:
        text = "Work out 701 × 75."

    assert prompt_shape(Easy()) == prompt_shape(Hard()) == "Work out # × #."


# ---------------------------------------------------------------------------
# mathgen.text_quality.prompt_shape -- slot-bank masking


def test_prompt_shape_masks_every_name_case_insensitively():
    class Lower:
        text = f"{NAMES[0].lower()} has 3 apples."

    class Title:
        text = f"{NAMES[0]} has 5 marbles."

    assert "§NAME§" in prompt_shape(Lower())
    assert prompt_shape(Lower()).replace("marbles", "apples") == prompt_shape(Title())


def test_prompt_shape_masks_both_singular_and_plural_object_forms():
    singular, plural = OBJECTS[0]

    class Singular:
        text = f"{NAMES[0]} has 1 {singular}."

    class Plural:
        text = f"{NAMES[0]} has 3 {plural}."

    shape_singular = prompt_shape(Singular())
    shape_plural = prompt_shape(Plural())
    assert "§OBJ§" in shape_singular and "§OBJ§" in shape_plural
    assert shape_singular == shape_plural  # only the count/form differs


def test_prompt_shape_masks_units_case_insensitively():
    class Q:
        text = f"The rope is 5 {UNITS[0].upper()} long."

    assert "§UNIT§" in prompt_shape(Q())


def test_prompt_shape_masks_contexts_as_one_placeholder():
    class Q:
        text = f"At {CONTEXTS[0]}, Amir bought 3 apples."

    shape = prompt_shape(Q())
    assert "§CTX§" in shape
    for word in CONTEXTS[0].split():
        if word != "a":
            assert word not in shape


def test_prompt_shape_collapses_whitespace():
    class Q:
        text = "Work  out   17   +  3."

    assert "  " not in prompt_shape(Q())


def test_prompt_shape_name_swap_collapses_to_the_same_shape():
    class A:
        text = f"{NAMES[0]} has 5 {OBJECTS[0][1]}. How many are left after giving away 2?"

    class B:
        text = f"{NAMES[1]} has 5 {OBJECTS[0][1]}. How many are left after giving away 2?"

    assert prompt_shape(A()) == prompt_shape(B())


def test_prompt_shape_real_structural_difference_does_not_collapse():
    class A:
        text = "What is the sum of 3 and 5?"

    class B:
        text = "What is the product of 3 and 5?"

    assert prompt_shape(A()) != prompt_shape(B())


# ---------------------------------------------------------------------------
# Edge cases


def test_prompt_shape_empty_text():
    class Q:
        text = ""

    assert prompt_shape(Q()) == ""


def test_prompt_shape_none_text_treated_as_empty():
    class Q:
        text = None

    assert prompt_shape(Q()) == ""


def test_prompt_shape_no_bank_entry_is_a_plain_english_word_risk_unchecked_by_text():
    """A name/object/unit/context that doubles as an ordinary English word
    used elsewhere in a prompt would over-mask. The known, accepted case is
    documented, not hidden: `mathgen.slots.UNITS` includes the bare unit "m"
    (metres), which also collides with algebra's conventional mass/gradient
    variable name "m" (e.g. `simple_harmonic_angular_frequency`'s "A mass m
    = 5 kg..."). Masking both alike is harmless for shape comparison (the
    position of the token is what matters, and it is the same in every
    render of the same source) -- recorded here as a known, deliberate
    imprecision rather than silently accepted."""

    class Q:
        text = "A mass m = 5 kg is attached to a spring with stiffness k = 20 N/m."

    shape = prompt_shape(Q())
    assert "§UNIT§" in shape
    assert "kg" not in shape  # kg is also masked, correctly
