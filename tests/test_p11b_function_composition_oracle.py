"""Independent oracle for the `p11b_function_composition` questions: it re-derives each answer from the rendered
question text.

Parses each rendered ``question.text`` and re-derives the answer with its own
regex and its own sympy/float arithmetic. Never imports
``mathgen.generators.p11b_function_composition_build`` -- per the conventions
file's "computational skills" oracle discipline for the compose skill (the
answer is directly computable from the parsed fixture), and the
"classification" discipline for the when_defined skill (the answer is one of
a small closed set, enumerated over the full draw space here at 400 seeds x 3
tiers, not a seed sample).
"""

import math
import re
from collections import Counter
from fractions import Fraction

from sympy import Symbol, sympify

from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import symbolically_equal

x = Symbol("x")
SEEDS = 400
TIERS = ("easy", "medium", "hard")

_FIXTURE_RE = re.compile(r"f\(x\) = 1/\(x ([+-]) (\d+)\) and g\(x\) = sqrt\(x - (\d+)\)")


def _parse_ak(text: str) -> tuple[int, int]:
    sign, kabs, a = _FIXTURE_RE.search(text).groups()
    k = -int(kabs) if sign == "+" else int(kabs)
    return int(a), k


def _oracle_compose_answer(a: int, k: int):
    return 1 / (sympify(f"sqrt(x - {a})") - k)


def _oracle_reverse_answer(a: int, k: int):
    """Derived fresh from a, k -- g(f(x)) = sqrt(f(x) - a) = sqrt(1/(x-k) - a).
    Not read from the generator's own formula string."""
    return sympify(f"sqrt(1/(x - {k}) - {a})")


def _oracle_when_defined_answer(a: int, k: int) -> str:
    if k < 0:
        return f"Defined for all x >= {a}"
    return f"Defined for all x >= {a}, except x = {a + k * k}"


def test_compose_oracle_matches_every_rendered_answer():
    """4 prompt shapes now render under mode="compose". Detect the shape from question.text by substring (never
    by importing the generator's shape names -- oracle discipline), independently re-derive the expected value
    per shape, and track a Counter per tier so an unrecognised shape is a hard failure, not a silent skip, and
    coverage of every shape available at each tier is asserted, not assumed."""
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "senior-find-the-composition-of-two-functions")

    total = 0
    shapes_by_tier = {source.difficulty: Counter() for source in skill.sources if source.difficulty != "exam"}
    for source in skill.sources:
        if source.difficulty == "exam":
            continue  # the exam source is covered by this skill's tests/test_exd_x_* file
        for seed in range(60):
            q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
            total += 1
            a, k = _parse_ak(q.text)
            text = q.text
            if "find g(f(x))" in text:
                shape = "reverse"
                expected = _oracle_reverse_answer(a, k)
                got = sympify(q.answer_text.replace("^", "**"))
                assert symbolically_equal(expected, got), (
                    f"seed={seed} difficulty={source.difficulty} reverse: oracle {expected} != "
                    f"rendered answer_text {q.answer_text!r}\ntext: {text}"
                )
            elif "evaluate f(g(" in text:
                shape = "point_eval"
                c = int(re.search(r"evaluate f\(g\((\d+)\)\)", text).group(1))
                m = math.isqrt(c - a)
                assert m * m == c - a, (
                    f"seed={seed} difficulty={source.difficulty} point_eval: c - a = {c - a} "
                    f"is not a perfect square\ntext: {text}"
                )
                expected_value = Fraction(1, m - k)
                got_value = Fraction(q.answer_text)
                assert got_value == expected_value, (
                    f"seed={seed} difficulty={source.difficulty} point_eval: oracle {expected_value} != "
                    f"rendered answer_text {q.answer_text!r}\ntext: {text}"
                )
            elif "as a single expression" in text:
                shape = "forward_b"
                expected = _oracle_compose_answer(a, k)
                got = sympify(q.answer_text.replace("^", "**"))
                assert symbolically_equal(expected, got), (
                    f"seed={seed} difficulty={source.difficulty}: oracle {expected} != "
                    f"rendered answer_text {q.answer_text!r}\ntext: {text}"
                )
            elif "find f(g(x))" in text:
                shape = "forward_a"
                expected = _oracle_compose_answer(a, k)
                got = sympify(q.answer_text.replace("^", "**"))
                assert symbolically_equal(expected, got), (
                    f"seed={seed} difficulty={source.difficulty}: oracle {expected} != "
                    f"rendered answer_text {q.answer_text!r}\ntext: {text}"
                )
            else:
                raise AssertionError(
                    f"unrecognised compose shape at seed={seed} difficulty={source.difficulty}: {text!r}"
                )
            shapes_by_tier[source.difficulty][shape] += 1
    print(f"\nsenior-find-the-composition-of-two-functions oracle: {total} questions re-derived, 0 challenged")
    for tier, counts in shapes_by_tier.items():
        print(f"  {tier}: {dict(counts)}")
    assert total == 60 * len([src for src in skill.sources if src.difficulty != "exam"])
    for tier, counts in shapes_by_tier.items():
        for shape in ("forward_a", "forward_b", "point_eval"):
            assert counts[shape] > 0, f"{tier}: shape {shape!r} never appeared over 60 seeds: {dict(counts)}"
    hard_counts = shapes_by_tier.get("hard", Counter())
    assert hard_counts["reverse"] > 0, f"hard: shape 'reverse' never appeared over 60 seeds: {dict(hard_counts)}"


def test_when_defined_oracle_matches_every_rendered_answer_over_full_draw_space():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "senior-determine-when-the-composition-of-two-functions-is")

    total = 0
    distribution = {tier: Counter() for tier in TIERS}
    for source in skill.sources:
        if source.difficulty == "exam":
            continue  # the exam source is covered by this skill's tests/test_exd_x_* file
        for seed in range(SEEDS):
            q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
            total += 1
            a, k = _parse_ak(q.text)
            expected = _oracle_when_defined_answer(a, k)
            assert expected == q.answer_text, (
                f"seed={seed} difficulty={source.difficulty}: oracle {expected!r} != "
                f"rendered answer_text {q.answer_text!r}\ntext: {q.text}"
            )
            distribution[source.difficulty][
                q.answer_text.startswith("Defined for all x >=") and "except" in q.answer_text
            ] += 1
    print(
        f"\nsenior-determine-when-the-composition-of-two-functions-is oracle: "
        f"{total} questions re-derived, 0 challenged"
    )
    for tier in TIERS:
        print(f"  {tier}: {dict(distribution[tier])}")
    assert total == SEEDS * len([src for src in skill.sources if src.difficulty != "exam"])
