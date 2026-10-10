"""Independent oracle for BOTH topics built: `linear_transform_composition` (ACMSM057) and
`linear_transform_inverse` (ACMSM058).
"""

import re

from mathgen import make

SEEDS = range(100)


# --------------------------------------------------------------------------
# Fresh (re-derived, not imported) transformation tables.
# --------------------------------------------------------------------------


def _dilation_direct(l1, l2, x, y):
    return l1 * x, l2 * y


def _rotation_direct(theta, x, y):
    if theta == 90:
        return -y, x
    if theta == 180:
        return -x, -y
    if theta == 270:
        return y, -x
    raise AssertionError(f"unexpected rotation angle {theta!r}")


_LINE_DESC_TO_KEY = {
    "the x-axis": "x-axis",
    "the y-axis": "y-axis",
    "the line y = x": "y=x",
    "the line y = -x": "y=-x",
}


def _reflection_direct(line_desc, x, y):
    key = _LINE_DESC_TO_KEY[line_desc]
    if key == "x-axis":
        return x, -y
    if key == "y-axis":
        return -x, y
    if key == "y=x":
        return y, x
    return -y, -x  # y=-x


_LINE_ALTS = r"(the x-axis|the y-axis|the line y = x|the line y = -x)"

_ANSWER_RE = re.compile(r"^x = (-?\d+), y = (-?\d+)$")


def _parse_answer_text(answer_text):
    m = _ANSWER_RE.match(answer_text)
    assert m, f"unexpected answer_text shape: {answer_text!r}"
    return int(m.group(1)), int(m.group(2))


# --------------------------------------------------------------------------
# Skill A oracle: linear_transform_composition
# --------------------------------------------------------------------------

_POINT_RE = re.compile(r"to the point \((-?\d+), (-?\d+)\)")
_DILATION_STEP_RE = re.compile(r"the dilation \(x, y\) -> \((-?\d+)x, (-?\d+)y\)")
_ROTATION_STEP_RE = re.compile(r"a rotation of (\d+) degrees about the origin")
_REFLECTION_STEP_RE = re.compile(r"a reflection in " + _LINE_ALTS)


def _ordered_composition_steps(text):
    """Every transformation phrase in `text`, in the order it appears --
    this always matches the STATED order in the question, whatever that
    order happened to be, because the generator's own text always reads
    'First apply X ... Then apply Y ...' with X and Y in the order to use."""
    matches = []
    for m in _DILATION_STEP_RE.finditer(text):
        matches.append((m.start(), "dilation", (int(m.group(1)), int(m.group(2)))))
    for m in _ROTATION_STEP_RE.finditer(text):
        matches.append((m.start(), "rotation", (int(m.group(1)),)))
    for m in _REFLECTION_STEP_RE.finditer(text):
        matches.append((m.start(), "reflection", (m.group(1),)))
    matches.sort(key=lambda item: item[0])
    return [(kind, params) for _start, kind, params in matches]


def _apply_step(kind, params, x, y):
    if kind == "dilation":
        return _dilation_direct(params[0], params[1], x, y)
    if kind == "rotation":
        return _rotation_direct(params[0], x, y)
    return _reflection_direct(params[0], x, y)


def _oracle_composition_answer(text):
    point_match = _POINT_RE.search(text)
    assert point_match, f"could not find the starting point in {text!r}"
    x, y = int(point_match.group(1)), int(point_match.group(2))
    steps = _ordered_composition_steps(text)
    assert len(steps) == 2, f"expected exactly 2 transformation steps, found {steps} in {text!r}"
    for kind, params in steps:
        x, y = _apply_step(kind, params, x, y)
    return x, y


class TestCompositionOracle:
    def test_easy_matches_oracle(self):
        n = 0
        for seed in SEEDS:
            q = make("linear_transform_composition", seed=seed, difficulty="easy")
            expected = _oracle_composition_answer(q.text)
            assert expected == _parse_answer_text(q.answer_text), (
                f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r} (question: {q.text!r})"
            )
            n += 1
        assert n == len(SEEDS)

    def test_medium_matches_oracle(self):
        for seed in SEEDS:
            q = make("linear_transform_composition", seed=seed, difficulty="medium")
            expected = _oracle_composition_answer(q.text)
            assert expected == _parse_answer_text(q.answer_text), (
                f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r} (question: {q.text!r})"
            )

    def test_hard_matches_oracle(self):
        for seed in SEEDS:
            q = make("linear_transform_composition", seed=seed, difficulty="hard")
            expected = _oracle_composition_answer(q.text)
            assert expected == _parse_answer_text(q.answer_text), (
                f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r} (question: {q.text!r})"
            )


# --------------------------------------------------------------------------
# Skill B oracle: linear_transform_inverse
# --------------------------------------------------------------------------

_INV_DILATION_RE = re.compile(
    r"A dilation \(x, y\) -> \((-?\d+)x, (-?\d+)y\) maps a point to the image \((-?\d+), (-?\d+)\)"
)
_INV_ROTATION_RE = re.compile(
    r"A rotation of (\d+) degrees about the origin maps a point to the image \((-?\d+), (-?\d+)\)"
)
_INV_REFLECTION_RE = re.compile(r"A reflection in " + _LINE_ALTS + r" maps a point to the image \((-?\d+), (-?\d+)\)")


def _oracle_inverse_answer(text):
    m = _INV_DILATION_RE.search(text)
    if m:
        l1, l2, p, q = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        assert l1 != 0 and l2 != 0, f"degenerate dilation factor in {text!r}"
        assert p % l1 == 0 and q % l2 == 0, f"non-exact dilation image in {text!r}"
        return p // l1, q // l2
    m = _INV_ROTATION_RE.search(text)
    if m:
        theta, p, q = int(m.group(1)), int(m.group(2)), int(m.group(3))
        inverse_theta = (360 - theta) % 360
        return _rotation_direct(inverse_theta, p, q)
    m = _INV_REFLECTION_RE.search(text)
    if m:
        line_desc, p, q = m.group(1), int(m.group(2)), int(m.group(3))
        # A reflection through the origin is its own inverse.
        return _reflection_direct(line_desc, p, q)
    raise AssertionError(f"could not parse inverse question: {text!r}")


class TestInverseOracle:
    def test_easy_matches_oracle(self):
        for seed in SEEDS:
            q = make("linear_transform_inverse", seed=seed, difficulty="easy")
            expected = _oracle_inverse_answer(q.text)
            assert expected == _parse_answer_text(q.answer_text), (
                f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r} (question: {q.text!r})"
            )

    def test_medium_matches_oracle(self):
        for seed in SEEDS:
            q = make("linear_transform_inverse", seed=seed, difficulty="medium")
            expected = _oracle_inverse_answer(q.text)
            assert expected == _parse_answer_text(q.answer_text), (
                f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r} (question: {q.text!r})"
            )

    def test_hard_matches_oracle(self):
        for seed in SEEDS:
            q = make("linear_transform_inverse", seed=seed, difficulty="hard")
            expected = _oracle_inverse_answer(q.text)
            assert expected == _parse_answer_text(q.answer_text), (
                f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r} (question: {q.text!r})"
            )
