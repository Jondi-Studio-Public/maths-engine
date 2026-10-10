"""Independent oracle for the `p9d_complex_plane_transform` questions: it re-derives each answer from the
rendered question text.

Parses only `question.text` and `question.answer_text` -- the artefact a
learner actually receives -- and recomputes the answer from scratch with a
hand-written complex-number parser and arithmetic, independent of both
`mathgen.generators.p9d_complex_plane_transform_build` (never imported here)
and `mathgen.generators.p5b_complex_plane_build` (also never imported here).
"""

from __future__ import annotations

import re

from mathgen.skills import load_skills
from mathgen import make

SEEDS = range(40)

# --- independent complex-number text parser --------------------------------
#
# Every complex number in these questions' text is written in exactly one of
# three student-facing shapes: a bare integer ("6", "-7"), a bare imaginary
# term ("-2i", "5i", "i", "-i"), or a two-term "a + bi"/"a - bi"/"a + i"/
# "a - i" expression -- the generator's own `_fmt_complex` reuses
# `mathgen.core.join_terms`, which elides a unit coefficient ("i", not "1i").
# Written fresh here, without reading the generator's own formatter.

_TWO_TERM_RE = re.compile(r"^(-?\d+)\s*([+-])\s*(\d*)i$")
_PURE_IMAG_RE = re.compile(r"^(-?)(\d*)i$")
_PURE_REAL_RE = re.compile(r"^(-?\d+)$")


def parse_complex(text: str) -> tuple[int, int]:
    """ "3 - 2i" -> (3, -2); "5i" -> (0, 5); "-6i" -> (0, -6); "-7" -> (-7, 0);
    "a + i" -> (a, 1); "a - i" -> (a, -1); "i" -> (0, 1); "-i" -> (0, -1)."""
    text = text.strip()
    m = _TWO_TERM_RE.match(text)
    if m:
        a = int(m.group(1))
        b = int(m.group(3)) if m.group(3) else 1
        if m.group(2) == "-":
            b = -b
        return a, b
    m = _PURE_IMAG_RE.match(text)
    if m:
        b = int(m.group(2)) if m.group(2) else 1
        if m.group(1) == "-":
            b = -b
        return 0, b
    m = _PURE_REAL_RE.match(text)
    if m:
        return int(m.group(1)), 0
    raise ValueError(f"cannot parse complex number from question text: {text!r}")


_ANSWER_PAIR_RE = re.compile(r"x = (-?\d+), y = (-?\d+)")


def parse_answer_pair(answer_text: str) -> tuple[int, int]:
    m = _ANSWER_PAIR_RE.match(answer_text.strip())
    assert m, f"cannot parse answer_text: {answer_text!r}"
    return int(m.group(1)), int(m.group(2))


def _skill(skill_id):
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == skill_id), None)
    assert skill is not None, f"{skill_id} not found in skills library"
    return skill


# ---------------------------------------------------------------------------
# Skill A: multiplication as a linear transformation
# ---------------------------------------------------------------------------


def test_mult_linear_transform_oracle():
    """Re-derive the image point from the question text alone: parse the
    multiplier w and the point z, multiply them by hand, and compare to
    answer_text."""
    skill = _skill("senior-examine-and-use-multiplication-as-a-linear-transformation")

    challenged = []
    examined = 0

    for source in skill.sources:
        for seed in SEEDS:
            q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **source.params)
            examined += 1

            w_text = q.text.split("w = ")[1].split(" is a linear")[0]
            z_text = q.text.split("z = ")[1].split(" under this")[0]

            a, b = parse_complex(w_text)
            x, y = parse_complex(z_text)

            expected_x = a * x - b * y
            expected_y = a * y + b * x

            got_x, got_y = parse_answer_pair(q.answer_text)
            if (got_x, got_y) != (expected_x, expected_y):
                challenged.append(
                    {
                        "seed": seed,
                        "variant": source.params.get("variant"),
                        "text": q.text,
                        "answer_text": q.answer_text,
                        "expected": (expected_x, expected_y),
                    }
                )

    assert not challenged, f"Found {len(challenged)} issues in {examined} questions: {challenged}"
    print(f"senior-examine-and-use-multiplication-as-a-linear-transformation: examined {examined}, challenged 0")


# ---------------------------------------------------------------------------
# Skill B: addition of complex numbers as vector addition
# ---------------------------------------------------------------------------


def test_addition_vector_oracle():
    """Re-derive the resultant (or the missing vector, for the inverse tier)
    from the question text alone, independent of the generator's own
    construction and check routes."""
    skill = _skill("senior-examine-addition-of-complex-numbers-as-vector-addition")

    challenged = []
    examined = 0

    for source in skill.sources:
        if source.difficulty == "exam":
            continue  # the exam source is covered by this skill's tests/test_exd_x_* file
        variant = source.params.get("variant")
        for seed in SEEDS:
            q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **source.params)
            examined += 1
            got_x, got_y = parse_answer_pair(q.answer_text)

            if variant == "two_term":
                z1_text = q.text.split("z1 = ")[1].split(" is represented")[0]
                z2_text = q.text.split("z2 = ")[1].split(" by the vector")[0]
                x1, y1 = parse_complex(z1_text)
                x2, y2 = parse_complex(z2_text)
                expected = (x1 + x2, y1 + y2)
            elif variant == "chained":
                z1_text = q.text.split("z1 = ")[1].split(", z2")[0]
                z2_text = q.text.split("z2 = ")[1].split(" and z3")[0]
                z3_text = q.text.split("z3 = ")[1].split(", each")[0]
                x1, y1 = parse_complex(z1_text)
                x2, y2 = parse_complex(z2_text)
                x3, y3 = parse_complex(z3_text)
                expected = (x1 + x2 + x3, y1 + y2 + y3)
            elif variant == "inverse":
                z1_text = q.text.split("z1 = ")[1].split(". The resultant")[0]
                r_text = q.text.split("R = ")[1].split(", where")[0]
                x1, y1 = parse_complex(z1_text)
                rx, ry = parse_complex(r_text)
                expected = (rx - x1, ry - y1)
            else:
                raise AssertionError(f"unexpected variant: {variant}")

            if (got_x, got_y) != expected:
                challenged.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "text": q.text,
                        "answer_text": q.answer_text,
                        "expected": expected,
                    }
                )

    assert not challenged, f"Found {len(challenged)} issues in {examined} questions: {challenged}"
    print(f"senior-examine-addition-of-complex-numbers-as-vector-addition: examined {examined}, challenged 0")
