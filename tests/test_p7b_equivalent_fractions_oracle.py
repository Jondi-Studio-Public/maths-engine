"""Computational oracle: parses each rendered question's text and derives the
answer from scratch, using only what a learner is given (question.text,
question.answer_text, question.distractors). Deliberately does NOT import
anything from mathgen.generators.p7b_equivalent_fractions_build -- not the
_denominator_word helper, not the misconception formulas, nothing. The
"named" tier's ordinal-fraction word is decoded with an independently
authored, hardcoded table below rather than any shared word-generation
logic.
"""

import re
from fractions import Fraction

from mathgen import make

TOPIC = "equivalent_fraction_related_denominator"

# Independent decode table: every target denominator this generator's
# "named" tier can produce, mapped to its English ordinal-fraction word.
# The named tier is restricted to denominators a Year 4/5 learner has an
# ordinary word for -- 14ths/21sts/27ths/28ths/32nds/36ths are excluded by
# the generator's own (d, k) pool, so they are not listed here either.
# Built by hand, not derived from any shared word-formatting code.
_WORD_TO_DENOMINATOR = {
    "sixths": 6,
    "eighths": 8,
    "tenths": 10,
    "twelfths": 12,
    "fifteenths": 15,
    "sixteenths": 16,
    "eighteenths": 18,
    "twentieths": 20,
    "twenty-fourths": 24,
    "thirtieths": 30,
    "fortieths": 40,
}

_DIRECT_RE = re.compile(r"^Write (\d+)/(\d+) as an equivalent fraction with denominator (\d+)\.$")
_NAMED_RE = re.compile(r"^Write (\d+)/(\d+) as an equivalent fraction in ([a-z-]+)\.$")
_BACKWARD_RE = re.compile(r"^(\d+)/(\d+) = \?/(\d+)\. What is the missing equivalent fraction\?$")


def _parse(text: str):
    """Return (n, d, D) derived purely from the rendered text, or None."""
    m = _DIRECT_RE.match(text)
    if m:
        n, d, D = (int(x) for x in m.groups())
        return n, d, D
    m = _NAMED_RE.match(text)
    if m:
        n, d, word = m.groups()
        n, d = int(n), int(d)
        D = _WORD_TO_DENOMINATOR.get(word)
        return n, d, D
    m = _BACKWARD_RE.match(text)
    if m:
        n, d, D = (int(x) for x in m.groups())
        return n, d, D
    return None


def test_equivalent_fraction_oracle():
    challenged = []
    examined = 0
    value_equal_distractors = 0

    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            examined += 1
            parsed = _parse(q.text)
            if parsed is None:
                challenged.append(
                    {
                        "seed": seed,
                        "difficulty": difficulty,
                        "issue": "could not parse question text",
                        "text": q.text,
                    }
                )
                continue
            n, d, D = parsed
            if D is None:
                challenged.append(
                    {
                        "seed": seed,
                        "difficulty": difficulty,
                        "issue": "could not decode named target denominator",
                        "text": q.text,
                    }
                )
                continue
            if D % d != 0 or D // d < 2:
                challenged.append(
                    {
                        "seed": seed,
                        "difficulty": difficulty,
                        "issue": f"D={D} is not a related (multiple, factor>=2) denominator of d={d}",
                        "text": q.text,
                    }
                )
                continue

            expected_numerator = n * (D // d)
            try:
                answer_fraction = Fraction(q.answer_text)
            except (ValueError, ZeroDivisionError) as exc:
                challenged.append(
                    {
                        "seed": seed,
                        "difficulty": difficulty,
                        "issue": f"cannot parse answer_text as fraction: {exc}",
                        "text": q.text,
                        "answer_text": q.answer_text,
                    }
                )
                continue

            expected_fraction = Fraction(expected_numerator, D)
            if answer_fraction != expected_fraction:
                challenged.append(
                    {
                        "seed": seed,
                        "difficulty": difficulty,
                        "issue": f"answer {answer_fraction} != independently derived {expected_fraction}",
                        "text": q.text,
                        "answer_text": q.answer_text,
                    }
                )
                continue
            if answer_fraction != Fraction(n, d):
                challenged.append(
                    {
                        "seed": seed,
                        "difficulty": difficulty,
                        "issue": f"answer {answer_fraction} is not even equivalent to the supplied fraction {n}/{d}",
                        "text": q.text,
                        "answer_text": q.answer_text,
                    }
                )
                continue

            for dist in q.distractors:
                try:
                    dist_fraction = Fraction(dist)
                except (ValueError, ZeroDivisionError) as exc:
                    challenged.append(
                        {
                            "seed": seed,
                            "difficulty": difficulty,
                            "issue": f"cannot parse distractor {dist!r}: {exc}",
                            "text": q.text,
                        }
                    )
                    continue
                if dist_fraction == answer_fraction:
                    value_equal_distractors += 1
                    challenged.append(
                        {
                            "seed": seed,
                            "difficulty": difficulty,
                            "issue": f"distractor {dist} is value-equal to the answer {answer_fraction}",
                            "text": q.text,
                            "distractors": q.distractors,
                        }
                    )

    assert not challenged, f"{len(challenged)} of {examined} questions challenged: {challenged}"
    print(
        f"equivalent_fraction_related_denominator: examined {examined}, "
        f"challenged 0, value-equal distractors {value_equal_distractors}/{examined}"
    )
