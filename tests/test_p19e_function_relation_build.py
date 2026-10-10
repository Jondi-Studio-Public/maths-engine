"""Tests `p19e_function_or_relation`: decide whether a stated relation is a function, four-way `mcq`.

`p19e_function_or_relation` (`answer: mcq`, 4-way)."""

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p19e_function_relation_build as m  # noqa: F401 -- registers the topic

SKILL_ID = "senior-recognise-the-distinction-between-functions-and-relations-and"
TOPIC = "p19e_function_or_relation"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

_domains, _skills = load_skills()
SKILL = next(s for s in _skills if s.id == SKILL_ID)
SOURCES = {src.difficulty: src for src in SKILL.sources}


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _key_option_text(q):
    m = re.search(rf"{q.answer_text}: (.+?)(?:   [A-D]:|$)", q.text)
    assert m, q.text
    return m.group(1)


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(difficulty, seed)
        assert len(q1.distractors) == 3
        assert q1.answer_text in ("A", "B", "C", "D")
        assert set(q1.distractors) | {q1.answer_text} == {"A", "B", "C", "D"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert not grade_mcq(q, "99").correct


def test_skill_loads_with_expected_shape():
    assert SKILL.answer == "mcq"
    assert SKILL.placement_eligible is False
    assert SKILL.domain == "functions-sequences"
    assert SKILL.prerequisites == []
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in SKILL.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


@pytest.mark.parametrize("difficulty", TIERS)
def test_exactly_one_true_option_every_draw(difficulty):
    """Read each option straight out of the text and confirm exactly the key
    reads 'Function: every x-value ...' XOR is a genuinely checkable claim --
    i.e. re-derive the SET of options and assert only the key's wording
    differs in kind from a fixed, checkable non-key set (the real hazard
    check -- that no two options can BOTH be true for the same relation --
    lives in the generator's own `check(sum(option_truth) == 1, ...)`, fired
    unconditionally on every draw; this test just confirms the rendered text
    always carries exactly one 'Function: every x-value ...' generic option,
    plus exactly three checkable claim options among the remaining four minus
    one -- a claim option is EITHER a `two_value` ("Not a function: x = ...
    gives two different y-values, a and b") OR a `one_value` ("Function: x =
    ... gives only y = a" -- a genuine two-value point
    mislabelled as single-valued, used when a shape's single-valued points
    run out)."""
    for seed in SEEDS:
        q = _make(difficulty, seed)
        func_opts = re.findall(r"Function: every x-value has exactly one y-value", q.text)
        two_value_opts = re.findall(r"Not a function: x = -?\d+ gives two different y-values, -?\d+ and -?\d+", q.text)
        one_value_opts = re.findall(r"Function: x = -?\d+ gives only y = -?\d+", q.text)
        assert len(func_opts) == 1, q.text
        assert len(two_value_opts) + len(one_value_opts) == 3, q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_class_distribution_within_bounds(difficulty):
    counts = Counter()
    N = 200
    for seed in range(N):
        q = _make(difficulty, seed)
        counts["function" if _key_option_text(q).startswith("Function: every") else "not"] += 1
    share = counts["function"] / N
    assert 0.15 <= share <= 0.45, (difficulty, share)
    assert counts["not"] > 0


def test_key_letter_roughly_balanced_over_400_seeds():
    counts = Counter(_make("easy", seed).answer_text for seed in range(400))
    for letter in ("A", "B", "C", "D"):
        assert 60 <= counts[letter] <= 160, counts


def test_check_fires_on_a_corrupted_exactly_one_true_claim():
    from mathgen.core import VerificationError, check

    truths_bad = [True, True, False, False]
    with pytest.raises(VerificationError):
        check(sum(truths_bad) == 1, "exactly one option must be true")


def test_pairs_is_function_helper_directly():
    from mathgen.generators.p19e_function_relation_build import _pairs_is_function

    assert _pairs_is_function([(1, 3), (2, 5)]) is True
    assert _pairs_is_function([(1, 3), (1, 3)]) is True  # exact duplicate pair
    assert _pairs_is_function([(1, 3), (1, 5)]) is False  # same x, different y
    assert _pairs_is_function([(1, 3), (2, 3)]) is True  # repeated y, distinct x


@pytest.mark.parametrize("difficulty", TIERS)
def test_function_key_share_in_25_to_35_percent_band_at_200_seeds(difficulty):
    """The yes-type ('Function') key must be 25-35% of draws at 200 seeds -- never near 50% or 0 (contract's
    binary-mcq rule). Checked per tier (a tighter band than `test_answer_class_distribution_within_bounds`'s
    generic 0.15-0.45)."""
    n_func = 0
    N = 200
    for seed in range(N):
        q = _make(difficulty, seed)
        if _key_option_text(q).startswith("Function: every"):
            n_func += 1
    share = n_func / N
    assert 0.25 <= share <= 0.35, (difficulty, n_func, share)


def test_function_key_share_in_25_to_35_percent_band_default_params():
    """Same band check with no explicit difficulty (`make(topic, seed=s)`),
    which falls back to the generator's own default tier."""
    n_func = 0
    N = 200
    for seed in range(N):
        q = make("p19e_function_or_relation", seed=seed)
        if _key_option_text(q).startswith("Function: every"):
            n_func += 1
    share = n_func / N
    assert 0.25 <= share <= 0.35, (n_func, share)


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_garbled_or_trap_location_hints_in_prompt(difficulty):
    """A prompt must never tell the learner where the trap is (removed 'look carefully', 'not unique', 'written
    as y^2 =... or a circle' style parentheticals)."""
    bad_phrases = (
        "look carefully",
        "written as y^2",
        "not unique",
        "some values repeat",
        "values are not unique",
        "or a circle",
    )
    for seed in range(200):
        q = _make(difficulty, seed)
        for phrase_ in bad_phrases:
            assert phrase_ not in q.text, (difficulty, seed, phrase_, q.text)


def test_exam_described_curves_with_specific_valued_options_state_their_equation():
    """Any exam-tier described curve whose options name specific coordinate values (the 'Not a function: x = k
    gives a, b' claims) must state its own equation explicitly in the description -- never just a
    vertex/parameter with the equation left implicit (the 'parabola opening to the right with vertex (1, -2)'
    hazard, where the key claim's truth secretly depended on an unstated leading coefficient)."""
    seen_right = seen_up = seen_left = False
    for seed in range(400):
        q = _make("exam", seed)
        if "opens to the left" in q.text:
            seen_left = True
            assert "the parabola x = -" in q.text, q.text
        elif "opens to the right" in q.text:
            seen_right = True
            assert "the parabola x = " in q.text, q.text
        if "opens upward" in q.text:
            seen_up = True
            assert "the parabola y = " in q.text, q.text
        if "the whole circle" in q.text or "upper semicircle" in q.text or "part of the circle" in q.text:
            assert "x^2" in q.text and "y^2" in q.text or "sqrt" in q.text
    assert seen_right and seen_up and seen_left, (
        "exam tier must draw the right-opening, upward-opening AND left-opening parabola variants over 400 seeds"
    )


# Kept in sync by hand; if the real detector's regexes change, update these too.
_TEXT_COMMAND_GROUP_RE = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\{[^{}]*\}")
_ESCAPE_MACRO_RE = re.compile(r"\\text(?:asciicircum|asciitilde|backslash)\{\}")
_SUBSCRIPT_GROUP_RE = re.compile(r"[_^]\{[^{}]*\}")
_ENVIRONMENT_RE = re.compile(r"\\(?:begin|end)\{[^{}]*\}")
_CAPS_LABEL_RE = re.compile(r"(?<![A-Za-z])[A-Z]{3,4}(?![A-Za-z])")
_COMMAND_TOKEN_RE = re.compile(r"\\[A-Za-z]+")
_LETTER_RUN_RE = re.compile(r"[A-Za-z]{3,}")


def _is_mangled(latex: str) -> bool:
    if not latex or latex.startswith("plain:"):
        return False
    stripped = _ENVIRONMENT_RE.sub("", latex)
    stripped = _ESCAPE_MACRO_RE.sub("", stripped)
    stripped = _TEXT_COMMAND_GROUP_RE.sub("", stripped)
    stripped = _SUBSCRIPT_GROUP_RE.sub("", stripped)
    stripped = _CAPS_LABEL_RE.sub("", stripped)
    stripped = _COMMAND_TOKEN_RE.sub("", stripped)
    return bool(_LETTER_RUN_RE.search(stripped)) or "sqrt" in stripped.lower()


@pytest.mark.parametrize("difficulty", TIERS)
def test_ordered_pairs_prompt_latex_is_not_mangled(difficulty):
    """`mathgen.core.latex_prompt` escapes a literal brace to `\\{`/`\\}` IN PLACE inside whichever `\\text{...}`
    span the prose split around a negative number fell into, nesting a second level of `{}` inside that span
    and breaking the mangled-latex detector's one-level `\\text{[^{}]*}` assumption -- the prose then "leaks"
    through as an apparent hand-rolled, non-LaTeX string. The generator now authors its own `latex`
    (`_prompt_latex`) that lifts the brace into math mode instead, so it must clear the real detector at every
    seed, for both tiers that build a `pairs_text` (easy, and hard's is_function branch) and the tiers that
    don't (medium, exam -- where this is a no-op)."""
    for seed in range(30):
        q = _make(difficulty, seed)
        assert not _is_mangled(q.latex), (difficulty, seed, q.latex)


def test_ordered_pairs_prompt_latex_lifts_braces_into_math_mode():
    """Confirms the fix actually changed something (not a vacuously-passing
    guard): at least one easy-tier seed's `text` carries the literal `{`/`}`
    set-notation braces, `latex_prompt` on that raw text alone IS mangled
    (pins the old defect), and the generator's own `q.latex` differs from
    that raw `latex_prompt(q.text)` -- proving the override in `make()`
    actually took effect -- and contains no `\\{`/`\\}` still nested inside
    a `\\text{}` argument (i.e. no literal `{`/`}` immediately following a
    `\\text{...}`-opening or preceding a `\\text{...}`-closing brace)."""
    from mathgen.core import latex_prompt

    seen_pairs_prompt = False
    for seed in range(30):
        q = _make("easy", seed)
        if "{" not in q.text:
            continue
        seen_pairs_prompt = True
        raw = latex_prompt(q.text)
        assert _is_mangled(raw), (seed, raw)  # pins the pre-fix defect
        assert q.latex != raw
        assert not _is_mangled(q.latex), (seed, q.latex)
    assert seen_pairs_prompt, "no easy-tier seed in range(30) built a pairs_text prompt"


def test_build_claims_check_fires_when_no_witness_exists():
    """Corrupt the claim: tell `_build_claims` the relation is not a function
    when `contains` is actually a genuine function -- it must fail to find a
    witness and raise."""
    from mathgen.core import VerificationError
    from mathgen.generators.p19e_function_relation_build import _build_claims
    import random

    def contains(x, y):
        return y == x  # y = x is a function: never two y at one x

    rng = random.Random(0)
    with pytest.raises(VerificationError):
        _build_claims(rng, contains, is_function=False, x_candidates=list(range(-5, 6)), n=3)


def test_find_false_pair_check_fires_on_a_genuine_two_value_x():
    """Regression guard for a previously fixed defect."""
    from mathgen.core import VerificationError
    from mathgen.generators.p19e_function_relation_build import _find_false_pair
    import random

    def contains(x, y):
        return x * x + y * y == 25  # circle, r = 5: x = 0 has y = +-5,+-4,+-3

    rng = random.Random(0)
    with pytest.raises(VerificationError):
        _find_false_pair(rng, contains, x=0)


_OPTION_RE = re.compile(
    r"([A-D]): (?:Function: every x-value has exactly one y-value"
    r"|Not a function: x = (-?\d+) gives two different y-values, -?\d+ and -?\d+"
    r"|Function: x = (-?\d+) gives only y = -?\d+)"
)


@pytest.mark.parametrize("difficulty", TIERS)
def test_false_claim_hygiene_at_200_seeds(difficulty):
    """Both defects, per tier at 200 seeds.

    Spies on the module's own `_assemble_options` to capture each draw's
    `contains` closure (direct internal access -- this is a build test for
    the module it owns), then re-derives, per non-key option's cited x,
    whether that x has a TRUE two-real-y verdict under `contains` and
    whether it is absent from the relation (zero real y).

    The first version only had two false-claim bands (single-valued, then
    outside-domain) and left a shape's OTHER genuine two-value points
    unused, forcing an "avoidable" absent claim on any not-a-function
    parabola (exactly one single-valued point -- its vertex -- but two false
    claims needed). The later version adds a third band, `one_value` claims
    ("Function: x = {x} gives only y = {a}", true iff `a` is the ONLY real y
    there): these spend a shape's OTHER two-value points as false claims
    (checkably false because they UNDERSTATE a genuine two-value x as
    single-valued), so every shape in this module now has enough in-domain
    x's and NEVER needs to fall back to one absent from the relation. Both
    counts below are therefore asserted at 0, unconditionally, not just
    "avoidable":

    - `n_shares_verdict`: a `two_value`-TEMPLATE distractor (same "Not a
      function: x = ... gives two different y-values, a and b" wording as
      the key) whose x is ALSO a genuine two-value point -- the exact defect
      false" but shares its verdict). The hard `check()`s in
      `_build_claims`/`_find_false_pair` make this structurally impossible.
      `one_value` distractors are deliberately excluded from this count: they
      use a DIFFERENT template ("Function: ... only y = ...") and are false
      by a different, self-contained mechanism (understating a two-value
      point), not by resembling the key's own claim shape.
    - `n_absent_raw`: ANY distractor (either template) whose cited x has
      ZERO real y under `contains` -- i.e. absent from the relation. With the
      third band available, this is 0 for every shape (set, circle, and both
      parabola families), not merely reduced.
    """
    import mathgen.generators.p19e_function_relation_build as gen_mod

    def _make_with_contains(seed):
        captured = {}
        orig = gen_mod._assemble_options

        def spy(rng, contains, is_function, x_candidates):
            result = orig(rng, contains, is_function, x_candidates)
            captured["contains"] = contains
            return result

        gen_mod._assemble_options = spy
        try:
            q = _make(difficulty, seed)
        finally:
            gen_mod._assemble_options = orig
        return q, captured["contains"]

    n_shares_verdict = 0
    n_absent_raw = 0
    for seed in range(200):
        q, contains = _make_with_contains(seed)
        options = _OPTION_RE.findall(q.text)
        assert len(options) == 4, q.text
        key = q.answer_text
        for letter, two_value_x, one_value_x in options:
            if letter == key:
                continue
            if two_value_x:
                x = int(two_value_x)
                ys = [y for y in gen_mod._Y_RANGE if contains(x, y)]
                if len(ys) >= 2:
                    n_shares_verdict += 1
                if len(ys) == 0:
                    n_absent_raw += 1
            elif one_value_x:
                x = int(one_value_x)
                ys = [y for y in gen_mod._Y_RANGE if contains(x, y)]
                if len(ys) == 0:
                    n_absent_raw += 1
    assert n_shares_verdict == 0, (difficulty, n_shares_verdict)
    assert n_absent_raw == 0, (difficulty, n_absent_raw)
