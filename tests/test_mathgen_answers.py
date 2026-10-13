"""Answer quality: distractor counts, stated precision, fractions in lowest terms.

Part of the generator sweep, split across `tests/test_mathgen_*.py` so the CI
shards can balance it (`scripts/ci_shard.py` balances files by size). Verification
is built into the generators themselves (each independently re-checks its
answer with SymPy and raises VerificationError on mismatch), so constructing
thousands of questions IS the correctness test; the assertions here cover the
output contract on top of that. Set `MATHS_DEEP_SEEDS` for a deeper run of the
text checks.
"""

import math
import re

import pytest

from mathgen import make, topics
from mathgen.core import PRESETS, REGISTRY
from tests.mathgen_sweep_support import TEXT_SEEDS
from mathgen.text_quality import requested_rounding

TOPICS = [t for _lvl, t in topics()]
# Topics registered by a build module (`mathgen/generators/*_build.py`): these
# must meet the learner-facing text bar. Legacy topics keep their existing
# notation (their `latex` field is what the UI renders) and are out of scope
# until someone owns them.
BUILD_TOPICS = sorted(t for t, (_lvl, fn) in REGISTRY.items() if fn.__module__.rsplit(".", 1)[-1].endswith("_build"))

# Genuine 2-option classification topics (mcq answer, exactly one distractor
# -- "the other label" -- by design, e.g. odd/even parity prediction) can
# never reach the >=2-distractor threshold below; a strict binary answer
# space has only one possible wrong option, ever. They are still covered by
# test_distractors_are_valid (test_mathgen_contract.py) (no collision/duplication) and by their
# own generator-specific tests.
GENUINE_TWO_OPTION_TOPICS = {
    "odd_even_properties",
    # Both are genuine yes/no classifications
    # (divisibility; a single named prime/composite/perfect-square property)
    # with no third option, same structural shape as odd_even_properties.
    "factors_multiples_divisibility",
    "prime_composite_square_properties",
}

# Classifications that look binary but have three or more answers by design, so they must keep the
# >=2-distractor check: never add them to GENUINE_TWO_OPTION_TOPICS.
NEVER_TWO_OPTION_TOPICS = {
    "p19h_rate_usage_classification",
    "p19h_appropriate_activity_unit",
    "p19h_need_for_milligrams",
}


def test_multi_option_classifications_are_not_registered_as_two_option():
    assert not NEVER_TWO_OPTION_TOPICS & GENUINE_TWO_OPTION_TOPICS


@pytest.mark.parametrize("topic", TOPICS)
def test_all_topics_usually_have_enough_distractors(topic):
    if topic in GENUINE_TWO_OPTION_TOPICS:
        pytest.skip("genuine 2-option mcq classification: exactly 1 distractor by design")
    probe = make(topic, seed=0)
    if isinstance(probe.answer, dict) and "checks" in probe.answer:
        # `answer: rubric` (skills/README.md): the answer is a marking block
        # for a free-entry proof, marked by rubric, never offered as options.
        pytest.skip("rubric-assessed: no options by design")
    # Every topic now emits misconception distractors. Candidates can collide
    # with the answer for some parameters (that's expected and filtered); but
    # across seeds, most questions should keep at least 2 usable distractors.
    n = 60 if topic == "differential_equation" else 100

    # A multi-part question has no top-level options: each part offers its own, so count those instead. A part
    # answered yes or no is a genuine two-option classification, like the topics above, so it is left out.
    def _offers(part):
        return {part.answer_text.strip().lower(), *(d.strip().lower() for d in part.distractors)} != {"yes", "no"}

    counts = [
        min((len(part.distractors) for part in question.parts if _offers(part)), default=2)
        if question.parts
        else len(question.distractors)
        for question in (make(topic, seed=s) for s in range(n))
    ]
    assert sum(1 for c in counts if c >= 2) / n >= 0.85


# ---------------------------------------------------------------------------
# Three mechanical checks that were missing entirely. Each is a property
# the learner meets directly and nothing looked at:
#
#   * an answer that does not honour the precision its own prompt states
#     (cartesian_polar_convert answered "y = 2" under "rounded to 1 decimal
#     place"), or a fraction not in lowest terms;
#   * a tier producing four distinct questions in sixty seeds, so the learner
#     meets the same handful of items over and over
#     (apply_the_product_quotient_and_chain_rule_to shipped exactly that);
#   * an answer formatted unlike every distractor, pickable without doing the
#     arithmetic (midpoint_coordinate_scalar rendered "1.5" among "3", "-4"
#     and "-11/2" in a third of its questions).
#
# The regrade sweep cannot see any of these: it grades the generator against
# itself.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_FRACTION_RE = re.compile(r"^-?(\d+)/(\d+)$")


# A table answer names each blank before its value ("Month 3 Interest ($) = 67.66"). The
# label is given text, not an asked value, so its digits are not held to the precision; a
# blank labelled "(count)" is a whole number by its own label.
_COUNT_PAIR_RE = re.compile(r"(^|,)[^,=]*\(count\)\s*=\s*-?\d+(?=,|$)")
_LABEL_RE = re.compile(r"(^|,)[^,=]*=")


def _asked_values(answer_text):
    return _LABEL_RE.sub(r"\1=", _COUNT_PAIR_RE.sub(r"\1", answer_text))


def test_asked_values_drops_labels_and_counts_only():
    assert _asked_values("Month 3 Interest ($) = 67.66, Month 3 Closing ($) = 6394.01") == "= 67.66,= 6394.01"
    assert _asked_values("18 to 34 TV (count) = 60, 18 to 34 TV (%) = 48.0") == ",= 48.0"
    assert _asked_values("x = 0.2, y = 2") == "= 0.2,= 2"
    assert _asked_values("0.25 or 1.5") == "0.25 or 1.5"


@pytest.mark.parametrize("topic", BUILD_TOPICS)
def test_answer_honours_its_own_stated_precision(topic):
    """An answer asked for to N decimal places is shown to N decimal places.

    Every number in `answer_text` is checked, so pair and list answers count
    too: the generator this caught rendered `x = 0.2, y = 2` because its
    formatter used ":g", which strips the trailing zero that carries the
    stated precision.
    """
    for difficulty in [*sorted(PRESETS.get(topic, {})), None]:
        for seed in range(TEXT_SEEDS):
            question = make(topic, seed=seed, difficulty=difficulty)
            # A multi-part question states its precision in each part's prompt and answers each part
            # separately, so pair every part's prompt with its own answer.
            asked = [(p.prompt, p.answer_text) for p in question.parts] or [(question.text, question.answer_text)]
            for prompt, answer_text in asked:
                places = requested_rounding(prompt)
                if places is None:
                    continue
                if re.search(r"\bexact value\b", prompt):
                    # "Show h is 11.0 cm to 1 d.p. ... write down the exact value of h^2": the 1 d.p. is
                    # about the quantity shown in the prompt, and the answer asked for is exact.
                    continue
                tokens = _NUMBER_RE.findall(_asked_values(answer_text or ""))
                if re.search(r"\bas an integer\b", prompt):
                    # The prompt asks for one value as an integer and the rest to `places`
                    # ("R as an integer and alpha to 1 d.p."), so one whole number is allowed.
                    whole = [t for t in tokens if "." not in t]
                    tokens = tokens if len(whole) > 1 else [t for t in tokens if "." in t]
                for token in tokens:
                    shown = len(token.split(".")[1]) if "." in token else 0
                    assert shown == places, (
                        f"{topic} difficulty={difficulty} seed={seed}: prompt asks for "
                        f"{places} decimal places, answer {answer_text!r} shows "
                        f"{token!r} with {shown}"
                    )


# Topics whose questions ASK for a fraction that is deliberately not in lowest
# terms, so the rule below does not apply to them. Keep this set tiny and keep
# earning it: `test_exempt_topics_really_do_ask_for_an_unsimplified_form` below
# asserts the OPPOSITE property for every entry, so an exemption cannot silently
# hide a genuine unsimplified-answer defect the way a bare skip would.
#
# `equivalent_fraction_related_denominator` asks "Write 1/2 as an equivalent
# fraction with denominator 6". The answer is 3/6. Simplifying it to 1/2 would
# not answer the question -- which is exactly what the docstring below means by
# "the form they were asked to produce". The blanket gcd == 1 rule contradicted
# that principle rather than expressing it.
_ASKS_FOR_UNSIMPLIFIED = {"equivalent_fraction_related_denominator"}

_QUESTION_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


@pytest.mark.parametrize("topic", sorted(_ASKS_FOR_UNSIMPLIFIED))
def test_exempt_topics_really_do_ask_for_an_unsimplified_form(topic):
    """An exemption from the lowest-terms rule has to be earned, every run.

    Checks the property that makes the exemption legitimate, in a way that does
    not depend on how the question is phrased (this topic asks three ways:
    "with denominator 6", "1/2 = ?/8", and "in fifteenths"). For every draw the
    answer must be EQUAL IN VALUE to the fraction stated in the question but
    written over a LARGER denominator -- i.e. genuinely an equivalent fraction
    rather than a simplified one -- and across the topic at least one answer
    must actually be unsimplified, or it would not need the exemption at all.
    """
    checked = unsimplified = 0
    for difficulty in [*sorted(PRESETS.get(topic, {})), None]:
        for seed in range(TEXT_SEEDS):
            question = make(topic, seed=seed, difficulty=difficulty)
            asked = _QUESTION_FRACTION_RE.search(question.text)
            assert asked, (
                f"{topic} seed={seed}: exempt from the lowest-terms rule, but the "
                f"question states no fraction to be equivalent to: {question.text!r}"
            )
            match = _FRACTION_RE.match(str(question.answer_text).strip())
            assert match, f"{topic} seed={seed}: answer {question.answer_text!r} is not a fraction"
            qn, qd = int(asked.group(1)), int(asked.group(2))
            an, ad = int(match.group(1)), int(match.group(2))
            assert qd and ad, f"{topic} seed={seed}: zero denominator"
            assert an * qd == qn * ad, (
                f"{topic} seed={seed}: answer {question.answer_text!r} is not equal in "
                f"value to the question's {qn}/{qd}"
            )
            assert ad > qd, (
                f"{topic} seed={seed}: answer {question.answer_text!r} does not use a "
                f"larger denominator than the question's {qn}/{qd}, so it is not an "
                f"equivalent-fraction answer"
            )
            checked += 1
            if math.gcd(an, ad) != 1:
                unsimplified += 1
    assert checked, f"{topic}: generated nothing to check"
    assert unsimplified, (
        f"{topic}: every answer WAS in lowest terms, so it does not need the "
        f"exemption -- remove it from _ASKS_FOR_UNSIMPLIFIED"
    )


@pytest.mark.parametrize("topic", BUILD_TOPICS)
def test_fractions_are_in_lowest_terms(topic):
    """No option is a fraction the learner would never write.

    `grade_fraction` compares by value, so an unsimplified canonical answer is
    not a grading failure. It is a worked answer the learner is shown after
    responding, and it should be the form they were asked to produce.

    Topics in `_ASKS_FOR_UNSIMPLIFIED` are exempt because for them the form the
    learner was asked to produce IS unsimplified; see the note there.
    """
    if topic in _ASKS_FOR_UNSIMPLIFIED:
        pytest.skip(f"{topic} asks for a deliberately unsimplified fraction")
    for difficulty in [*sorted(PRESETS.get(topic, {})), None]:
        for seed in range(TEXT_SEEDS):
            question = make(topic, seed=seed, difficulty=difficulty)
            for option in [question.answer_text, *question.distractors]:
                if not isinstance(option, str):
                    continue
                match = _FRACTION_RE.match(option.strip())
                if match is None:
                    continue
                numerator, denominator = int(match.group(1)), int(match.group(2))
                if denominator == 0:
                    continue
                assert math.gcd(numerator, denominator) == 1, (
                    f"{topic} difficulty={difficulty} seed={seed}: {option!r} is not in lowest terms"
                )
