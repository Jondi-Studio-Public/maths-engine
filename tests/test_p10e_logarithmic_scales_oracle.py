"""Independent oracle for f10-logarithmic-scales.
Parses ONLY what a learner is given (`question.text`, `question.answer_text`,
`question.distractors`) and never imports anything from
`mathgen.generators.p10e_logarithmic_scales_build`. Recomputes the
position/value relationship via actual base-10 logarithms
(`math.log10`) -- a genuinely different route from the generator's own
integer-exponent construction.
"""

import math
import re
from fractions import Fraction

from mathgen import make

TOPIC = "p10e_logarithmic_scales"

_FORWARD_RE = re.compile(
    r"On a logarithmic scale, A is marked at (\d+) and B is marked at (\d+), (\d+) equal "
    r"scale-steps apart \(each step means multiplying by 10\)\. A point lies "
    r"(\d+)/(\d+) of the way from A towards B"
)
_INVERSE_RE = re.compile(
    r"On a logarithmic scale, A is marked at (\d+) and B is marked at (\d+), (\d+) equal "
    r"scale-steps apart \(each step means multiplying by 10\)\. The value (\d+) "
    r"sits at some position between A and B"
)


def _log_ratio(A: int, B: int, value: float) -> float:
    return (math.log10(value) - math.log10(A)) / (math.log10(B) - math.log10(A))


def _re_derive(question):
    m = _FORWARD_RE.match(question.text)
    if m:
        A, B, k, j, k2 = (int(g) for g in m.groups())
        assert k == k2
        t = Fraction(j, k)
        expected_value = float(question.answer_text)
        computed_t = _log_ratio(A, B, expected_value)
        return "forward", abs(computed_t - float(t))
    m = _INVERSE_RE.match(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    A, B, k, value = (int(g) for g in m.groups())
    expected_t = float(question.answer_text)
    computed_t = _log_ratio(A, B, value)
    return "inverse", abs(computed_t - expected_t)


def test_oracle_rederives_answer_every_tier():
    n = 0
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(60):
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            kind, gap = _re_derive(question)
            assert gap < 1e-6, f"seed {seed} {difficulty}: oracle log-ratio gap {gap} (question: {question.text!r})"
            n += 1
    assert n == 180


def test_oracle_rejects_every_distractor():
    n = 0
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(60):
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            m = _FORWARD_RE.match(question.text)
            for distractor in question.distractors:
                if m:
                    A, B, k, j, _ = (int(g) for g in m.groups())
                    t = Fraction(j, k)
                    gap = abs(_log_ratio(A, B, float(distractor)) - float(t))
                else:
                    im = _INVERSE_RE.match(question.text)
                    A, B, k, value = (int(g) for g in im.groups())
                    gap = abs(_log_ratio(A, B, value) - float(distractor))
                assert gap > 1e-6, (
                    f"seed {seed} {difficulty}: distractor {distractor!r} matches the log-ratio relationship"
                )
                n += 1
    assert n > 0
