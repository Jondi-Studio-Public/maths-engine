"""Exam tier of the six Cartesian complex-number skills (add, conjugate, multiply, real part, imaginary part,
modulus): prompt variety, an independent oracle that re-derives each key from the rendered prompt alone, and
the real graders accepting the key and rejecting every distractor.

The oracle shares no code with ``mathgen.generators.exd_x_complex_arith``: it turns the printed expression
into a Python complex number and evaluates it.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import mcq_options
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

# topic -> (answer kind, (existing topic, existing params, existing difficulty) of the skill's hardest practice tier)
TOPICS = {
    "exd_x_complex_add": ("mcq", ("complex_number", {"kind": "add"}, "easy")),
    "exd_x_complex_conjugate": ("mcq", ("complex_number", {"kind": "conjugate"}, "easy")),
    "exd_x_complex_multiply": ("mcq", ("complex_number", {"kind": "multiply"}, "easy")),
    "exd_x_complex_real_part": ("number", ("complex_number", {"kind": "real_part"}, "easy")),
    "exd_x_complex_imaginary_part": ("number", ("complex_number", {"kind": "imaginary_part"}, "easy")),
    "exd_x_complex_modulus": ("number", ("complex_number", {"kind": "modulus"}, "easy")),
}

_C = r"(?<![A-Za-z])(?:-?\d*i|-?\d+(?: [+-] \d*i)?)(?![A-Za-z])"


def _parse_c(token: str) -> complex:
    """'3 - 2i', '-i', '5', '7i' -> complex."""
    token = token.strip()
    m = re.fullmatch(r"(-?\d+) ([+-]) (\d*)i", token)
    if m:
        im = int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
        return complex(int(m.group(1)), im)
    m = re.fullmatch(r"(-?)(\d*)i", token)
    if m:
        return complex(0, (-1 if m.group(1) else 1) * int(m.group(2) or 1))
    return complex(int(token), 0)


def _eval_expr(expr: str) -> complex:
    """Evaluate a printed expression such as '(3 + i)^2(1 - 2i)' or '(a)/(b)' with Python complex arithmetic."""
    py = re.sub(r"(\d*)i", lambda m: (m.group(1) or "1") + "j", expr)
    py = py.replace("^", "**").replace(")(", ")*(").replace(")**2(", ")**2*(")
    py = re.sub(r"\)(\d)", r")*\1", py)
    return complex(eval(py, {"__builtins__": {}}, {}))  # noqa: S307 -- test-only, input is our own prompt


def _q(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def _expr(text: str) -> str:
    for pat in (
        r"^Find (?:Re|Im)\((.*)\)\.$",
        r"^Find the (?:real|imaginary) part of (.*)\. Give a whole number\.$",
        r"^What is the (?:real|imaginary) part of the complex number (.*)\? Give the coefficient only, a whole number\.$",
        r"^Find the modulus of (.*)\.$",
        r"^Calculate \|(.*)\|\.$",
        r"^What is the value of \|(.*)\|\?$",
        r"^Find the conjugate of (.*)\.$",
        r"^What is the complex conjugate of (.*)\?$",
        r"^Write the conjugate of (.*) in the form a \+ bi\.$",
        r"^Calculate (.*)\.$",
        r"^Expand and simplify (.*)\.$",
        r"^Write (.*) in the form a \+ bi\.$",
    ):
        m = re.match(pat, text)
        if m:
            return m.group(1)
    raise AssertionError(f"unrecognised prompt: {text!r}")


def _oracle(topic: str, text: str):
    """The key re-derived from the prompt text alone."""
    if topic == "exd_x_complex_add":
        q = _parse_c(re.search(rf"(?:Let q|Given q|where q) = ({_C})", text).group(1))
        s = _parse_c(re.search(rf"conjugate of q\)? = ({_C})", text).group(1))
        k = int(m.group(1)) if (m := re.search(r"(\d+) times", text)) else 1
        sign = 1 if re.search(r"p \+ ", text) else -1
        return s - sign * k * q.conjugate()
    value = _eval_expr(_expr(text))
    if topic == "exd_x_complex_conjugate":
        return value.conjugate()
    if topic == "exd_x_complex_multiply":
        return value
    if topic == "exd_x_complex_real_part":
        return value.real
    if topic == "exd_x_complex_imaginary_part":
        return value.imag
    if topic == "exd_x_complex_modulus":
        return abs(value)
    raise AssertionError(topic)


@pytest.mark.parametrize("topic", sorted(TOPICS))
def test_key_matches_independent_oracle_over_400_seeds(topic):
    kind = TOPICS[topic][0]
    bad = []
    for seed in range(400):
        q = _q(topic, seed)
        expected = _oracle(topic, q.text)
        if kind == "mcq":
            got = _parse_c(q.answer_text)
            ok = abs(got - expected) < 1e-6
        else:
            ok = abs(int(q.answer_text) - expected) < 1e-6 and q.answer == int(q.answer_text)
        if not ok:
            bad.append((seed, q.text, q.answer_text, expected))
    assert bad == [], bad[:5]


@pytest.mark.parametrize("topic", sorted(TOPICS))
def test_prompt_variety_and_new_shapes_over_hard_tier(topic):
    shapes = {prompt_shape(_q(topic, s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    old_topic, old_params, old_diff = TOPICS[topic][1]
    old_shapes = {prompt_shape(make(old_topic, seed=s, difficulty=old_diff, **old_params)) for s in range(60)}
    assert shapes - old_shapes, (shapes, old_shapes)


@pytest.mark.parametrize("topic", sorted(TOPICS))
def test_no_text_defects_and_deterministic(topic):
    for seed in range(60):
        q = _q(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == _q(topic, seed).to_json()


@pytest.mark.parametrize("topic", sorted(TOPICS))
def test_real_grader_accepts_key_and_rejects_every_distractor(topic):
    kind = TOPICS[topic][0]
    grader = GRADERS[kind]
    bad = []
    for seed in range(150):
        q = _q(topic, seed)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        if kind == "mcq":
            options = mcq_options(q)
            for idx, opt in enumerate(options):
                correct = grader(q, idx).correct
                if correct != (opt == q.answer_text):
                    bad.append((seed, idx, opt))
        else:
            if not grader(q, q.answer_text).correct:
                bad.append(("key rejected", seed, q.answer_text))
            for d in q.distractors:
                if grader(q, d).correct:
                    bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]


def test_exam_preset_only_and_unknown_tier_rejected():
    from mathgen.core import REGISTRY

    for topic in TOPICS:
        with pytest.raises(ValueError):
            REGISTRY[topic][1](__import__("random").Random(0), tier="easy")
