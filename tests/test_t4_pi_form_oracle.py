"""Independent oracle for `angle-degrees-radians-pi-form`.

Recomputes the answer from the rendered question text alone, with its own
small regexes over `question.text`/`question.answer_text`. Does NOT import
anything from `mathgen.generators.t4_pi_form_build` -- not the pools, not
`_format_pi_multiple`, not the topic's own construction of `coeff`.
"""

import re
from fractions import Fraction

from mathgen import make

TOPIC = "t4_degrees_radians_pi_form"

# A pi-form string, standalone: "pi", "2pi", "pi/6", "7pi/6".
_PI_STRING_RE = re.compile(r"^(\d*)pi(?:/(\d+))?$")
_DEG_IN_TEXT_RE = re.compile(r"(\d+) degrees")
_PI_IN_TEXT_RE = re.compile(r"The angle (\d*pi(?:/\d+)?) rad")


def _parse_pi_string(s: str) -> Fraction:
    match = _PI_STRING_RE.match(s.strip())
    assert match, f"could not parse pi-form string {s!r}"
    numerator = int(match.group(1)) if match.group(1) else 1
    denominator = int(match.group(2)) if match.group(2) else 1
    return Fraction(numerator, denominator)


def test_oracle_forward_common_easy_tier():
    checked = 0
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="easy")
        match = _DEG_IN_TEXT_RE.search(q.text)
        assert match, f"seed={seed}: no degree count found in {q.text!r}"
        deg = int(match.group(1))
        expected_coeff = Fraction(deg, 180)
        got_coeff = _parse_pi_string(q.answer_text)
        assert got_coeff == expected_coeff, (
            f"easy seed={seed}: {deg} degrees should reduce to {expected_coeff} * pi, "
            f"but answer_text {q.answer_text!r} implies {got_coeff}"
        )
        checked += 1
    assert checked == 40


def test_oracle_forward_fine_medium_tier():
    checked = 0
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="medium")
        match = _DEG_IN_TEXT_RE.search(q.text)
        assert match, f"seed={seed}: no degree count found in {q.text!r}"
        deg = int(match.group(1))
        expected_coeff = Fraction(deg, 180)
        got_coeff = _parse_pi_string(q.answer_text)
        assert got_coeff == expected_coeff, (
            f"medium seed={seed}: {deg} degrees should reduce to {expected_coeff} * pi, "
            f"but answer_text {q.answer_text!r} implies {got_coeff}"
        )
        checked += 1
    assert checked == 40


def test_oracle_reverse_hard_tier():
    checked = 0
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="hard")
        match = _PI_IN_TEXT_RE.search(q.text)
        assert match, f"seed={seed}: no pi-form angle found in {q.text!r}"
        coeff = _parse_pi_string(match.group(1))
        expected_deg = coeff * 180
        assert expected_deg.denominator == 1, (
            f"seed={seed}: {coeff} * 180 = {expected_deg} is not a whole number of degrees"
        )
        got_deg = int(q.answer_text)
        assert int(expected_deg) == got_deg, (
            f"seed={seed}: {coeff} rad should be {int(expected_deg)} degrees, but answer_text says {q.answer_text!r}"
        )
        checked += 1
    assert checked == 40
