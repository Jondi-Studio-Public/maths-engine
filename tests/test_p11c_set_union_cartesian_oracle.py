"""Independent oracle for the `p11c_set_union_cartesian` questions: it re-derives each answer from the rendered
question text.

Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p11c_set_union_cartesian_build` and recomputes the
expected answer from first principles, then compares it against
`answer_text`/`question.answer`'s own displayed value.

Generates through each skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""

import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

UNION_TOPIC = "p11c_set_union_cardinality"
CARTESIAN_TOPIC = "p11c_cartesian_form_equating"

_UNION_ID = "senior-determine-and-use-the-formulas-for-finding-the"
_CARTESIAN_ID = "senior-review-cartesian-form"

SEEDS = range(60)


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, skill_id, difficulty, seed):
    skill = skills_dict[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# =============================================================================
# Skill 1: p11c_set_union_cardinality (ACMSM005)
# =============================================================================

_UNION_EASY_RE = re.compile(
    r"In a class of (\d+) students, (\d+) study \w+, (\d+) study \w+, and (\d+) study both "
    r"\w+ and \w+\. How many study neither\?"
)

# Renamed from the old `_UNION_HARD_RE`.
_UNION_TWO_SET_OVERLAP_RE = re.compile(
    r"In a class of (\d+) students, (\d+) study \w+ and (\d+) study \w+\. (\d+) study neither "
    r"\w+ nor \w+\. How many study both \w+ and \w+\?"
)

# The old three-set union text (always asks "how many like at least one",
# never states the union total, never omits a region) is no longer reachable
# through the skill's shipping sources -- it stays live only as the `medium`
# preset direct-call regression pin in test_p11c_set_union_cartesian_build.py
# (`test_union_legacy_presets_backward_compatible_outputs_pinned`). The
# new hard tier states the union total and omits exactly one of
# A, B, C, AB, AC, BC, ABC -- three structurally distinct shapes, one regex
# per unknown kind, each independently re-deriving the answer from the
# seven-term inclusion-exclusion identity.

_UNION_THREE_SET_ABC_RE = re.compile(
    r"In a survey of students, (\d+) like \w+, (\d+) like \w+ and (\d+) like \w+\. "
    r"(\d+) like both \w+ and \w+ \(whether or not they also like \w+\), "
    r"(\d+) like both \w+ and \w+ \(whether or not they also like \w+\), and "
    r"(\d+) like both \w+ and \w+ \(whether or not they also like \w+\)\. "
    r"(\d+) students like at least one of \w+, \w+ or \w+\. How many students like all three\?"
)

_UNION_THREE_SET_C_RE = re.compile(
    r"In a survey of students, (\d+) like \w+ and (\d+) like \w+\. "
    r"(\d+) like both \w+ and \w+ \(whether or not they also like \w+\), "
    r"(\d+) like both \w+ and \w+ \(whether or not they also like \w+\), "
    r"(\d+) like both \w+ and \w+ \(whether or not they also like \w+\), and "
    r"(\d+) like all three\. "
    r"(\d+) students like at least one of \w+, \w+ or \w+\. "
    r"How many students like \w+ in total \(whether or not they also like \w+ or \w+\)\?"
)

_UNION_THREE_SET_AB_RE = re.compile(
    r"In a survey of students, (\d+) like \w+, (\d+) like \w+ and (\d+) like \w+\. "
    r"(\d+) like both \w+ and \w+ \(whether or not they also like \w+\), "
    r"(\d+) like both \w+ and \w+ \(whether or not they also like \w+\), and "
    r"(\d+) like all three\. "
    r"(\d+) students like at least one of \w+, \w+ or \w+\. "
    r"How many students like both \w+ and \w+ \(whether or not they also like \w+\)\?"
)


def _oracle_union_easy(text: str) -> int:
    m = _UNION_EASY_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable easy-tier union question: {text!r}")
    N, a, b, c = (int(g) for g in m.groups())
    # From scratch: |A union B| = a + b - c; neither = N - |A union B|.
    return N - (a + b - c)


def _oracle_union_two_set_overlap(text: str) -> int:
    m = _UNION_TWO_SET_OVERLAP_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable two_set_overlap union question: {text!r}")
    N, a, b, neither = (int(g) for g in m.groups())
    # From scratch: |A union B| = N - neither; |A intersect B| = a + b - |A union B|.
    union = N - neither
    return a + b - union


def _oracle_union_three_set_unknown(text: str) -> int:
    m = _UNION_THREE_SET_ABC_RE.search(text)
    if m:
        A, B, C, AB, AC, BC, union_total = (int(g) for g in m.groups())
        # From scratch: rearrange the seven-term identity for the triple.
        return union_total - A - B - C + AB + AC + BC

    m = _UNION_THREE_SET_C_RE.search(text)
    if m:
        A, B, AB, AC, BC, ABC, union_total = (int(g) for g in m.groups())
        # From scratch: rearrange the seven-term identity for the third set's total.
        return union_total - A - B + AB + AC + BC - ABC

    m = _UNION_THREE_SET_AB_RE.search(text)
    if m:
        A, B, C, AC, BC, ABC, union_total = (int(g) for g in m.groups())
        # From scratch: rearrange the seven-term identity for the AB pairwise total.
        return A + B + C - AC - BC + ABC - union_total

    raise AssertionError(f"unparseable three_set_unknown union question: {text!r}")


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_union_easy),
        ("medium", _oracle_union_two_set_overlap),
        ("hard", _oracle_union_three_set_unknown),
    ],
)
def test_union_oracle(skills_dict, difficulty, oracle_fn):
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _UNION_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = int(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# =============================================================================
# Skill 2: p11c_cartesian_form_equating (ACMSM078)
# =============================================================================

_TOKEN = r"-?\d+"

_CARTESIAN_EASY_RE = re.compile(r"Let z = x \+ yi\. If \((.+?)\) \+ \((.+?)\)i = (.+?), find x and y\.")

_CARTESIAN_MEDIUM_RE = re.compile(r"Let z = x \+ yi\. If z \+ \((.+?)\) = (.+?), find x and y\.")

_CARTESIAN_HARD_RE = re.compile(
    r"Let z = x \+ yi\. The conjugate of z is x - yi\. If z ([+-]) (\d+) times the conjugate "
    r"of z equals (.+?), find x and y\."
)


def _parse_linear_expr(expr: str, var: str) -> tuple[int, int]:
    """Parse a linear expression in `var` written the way `join_terms` emits
    it ("2x - 1", "-x + 4", "y", "-3") into (coefficient, constant). Built
    from scratch via a small hand tokenizer -- no import from the generator."""
    expr = expr.strip()
    coef, const = 0, 0
    # Split on whitespace around the top-level +/- that
    # join_terms always inserts, by scanning char by char for a +/- that is
    # preceded and followed by a space (join_terms's own separator style).
    parts = re.split(r"(?<=\S)\s+(?=[+-]\s)", expr)
    normalised = []
    sign = "+"
    for part in parts:
        part = part.strip()
        if part.startswith("+"):
            sign = "+"
            part = part[1:].strip()
        elif part.startswith("-"):
            sign = "-"
            part = part[1:].strip()
        normalised.append((sign, part))
        sign = "+"
    for sign, term in normalised:
        signval = 1 if sign == "+" else -1
        if term == var:
            coef += signval * 1
        elif term.endswith(var):
            coef += signval * int(term[: -len(var)])
        elif term == "":
            continue
        else:
            const += signval * int(term)
    return coef, const


def _parse_complex(expr: str) -> tuple[int, int]:
    """Parse "3 - 2i" / "i" / "-6i" / "-7" (join_terms's own output shape for
    a complex number) into (real, imag)."""
    coef, const = _parse_linear_expr(expr, "i")
    return const, coef


def _oracle_cartesian_easy(text: str) -> tuple[int, int]:
    m = _CARTESIAN_EASY_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable easy-tier cartesian question: {text!r}")
    real_expr, imag_expr, rhs_expr = m.groups()
    a, b = _parse_linear_expr(real_expr, "x")
    c, d = _parse_linear_expr(imag_expr, "y")
    e, f = _parse_complex(rhs_expr)
    # From scratch: solve the two separated linear equations directly.
    x = Fraction(e - b, a)
    y = Fraction(f - d, c)
    assert x.denominator == 1 and y.denominator == 1, (text, x, y)
    return int(x), int(y)


def _oracle_cartesian_medium(text: str) -> tuple[int, int]:
    m = _CARTESIAN_MEDIUM_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable medium-tier cartesian question: {text!r}")
    pq_expr, rhs_expr = m.groups()
    p, q = _parse_complex(pq_expr)
    r, s = _parse_complex(rhs_expr)
    # From scratch: z = (r + si) - (p + qi).
    return r - p, s - q


def _oracle_cartesian_hard(text: str) -> tuple[int, int]:
    m = _CARTESIAN_HARD_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable hard-tier cartesian question: {text!r}")
    sign_word, k_str, rhs_expr = m.groups()
    k = int(k_str) if sign_word == "+" else -int(k_str)
    r, s = _parse_complex(rhs_expr)
    # From scratch: z + k*conjugate(z) = (1+k)x + (1-k)y*i = r + si.
    x = Fraction(r, 1 + k)
    y = Fraction(s, 1 - k)
    assert x.denominator == 1 and y.denominator == 1, (text, x, y)
    return int(x), int(y)


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_cartesian_easy),
        ("medium", _oracle_cartesian_medium),
        ("hard", _oracle_cartesian_hard),
    ],
)
def test_cartesian_oracle(skills_dict, difficulty, oracle_fn):
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _CARTESIAN_ID, difficulty, seed)
        expected_x, expected_y = oracle_fn(q.text)
        actual_x, actual_y = q.answer["x"], q.answer["y"]
        assert (expected_x, expected_y) == (actual_x, actual_y), (
            f"seed {seed} {difficulty}: expected x={expected_x}, y={expected_y}, "
            f"got x={actual_x}, y={actual_y} -- {q.text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)
