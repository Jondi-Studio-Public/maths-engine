"""Generates `p13c_trig_identity_integrate` questions (ACMSM116): integrate an expression by first applying a
trigonometric identity.

Kernel: integrate an integrand that CONTAINS ``sin^2(kx)``, ``cos^2(kx)`` or
``tan^2(kx)`` by first rewriting it via the three stated identities --
power-reduction (``sin^2 x = (1 - cos 2x)/2``, ``cos^2 x = (1 + cos 2x)/2``)
for the squared sin/cos terms, ``tan^2 x = sec^2 x - 1`` for the tan term --
and THEN integrating the rewritten form. The identity substitution is the
content the descriptor names; a bare linear ``sin(kx)``/``cos(kx))`` integrand
(no squaring, no identity) is never offered here.

**The boundary against ``integrate-trig``**
(``skills/integration/integrate-trig.yaml``): rendered directly, that skill's
own questions are shaped ``Find ∫ (25cos(5x)) dx`` -- a BARE first-power
sin/cos integrand, answered by the direct ``∫sin``/``∫cos`` rule with no
identity step at all. Every question this generator makes contains a squared
trig term (``sin^2``, ``cos^2``) or a ``tan^2`` term; none is ever swappable
into ``integrate-trig``'s territory.

**The ``sec``/``tan`` parsing finding, confirmed directly (not assumed)**:
`mathgen/grading/typed.py::_parse_expression` calls `sympy.parsing.sympy_parser.parse_expr`
with NO restricted `local_dict`/`global_dict`, so every sympy function name
(including `sec`) parses through it correctly -- confirmed directly:
`_parse_expression("sec(x)**2")` returns `sec(x)**2` and
`sympy.simplify(_parse_expression("sec(x)**2") - (1 + tan(x)**2)) == 0`. The
worry that `sec` might not be a recognised sympy function name in that
parser's allowed symbol table does NOT apply: `sec` IS recognised. This generator still never needs
`sec` in any canonical answer or distractor regardless, because every
antiderivative in the ladder below (power-reduction and
``tan^2 x = sec^2 x - 1``) integrates to a `sin`/`cos`/`tan`/`x` expression,
never a `sec` -- so the finding is recorded for the record, not load-bearing
for this module's own text.

**The ``+C`` grading finding, proven not assumed** (same convention as
``integrate-polynomial``/``integrate-trig`` and the
``p13c_antideriv_notation_build`` generator, confirmed again here with a direct
``grade_expression`` call in this module's own test file):
``question.answer`` holds the antiderivative WITHOUT ``+ C``;
``question.answer_text`` holds the same expression WITH ``" + C"`` appended.
`mathgen/grading/typed.py::_expected_expression` treats the DISPLAYED ``answer_text``
as authoritative whenever it differs from the internal ``answer`` by exactly
one free symbol (the constant of integration), so a learner who drops
``+ C`` is compared against the full antiderivative and correctly REJECTED,
while a learner who types the full canonical is ACCEPTED. Every
constant-of-integration distractor in this module is spelled with the same
letter ``C`` the canonical uses.

**Difficulty ladder** (every tier ``answer: expression``, one grader kind):
- easy: ``∫sin^2(kx) dx`` or ``∫cos^2(kx) dx`` (single term, integer ``k``).
  Direct application of the stated power-reduction identity.
- medium: ``∫tan^2(kx) dx`` using ``1 + tan^2 x = sec^2 x``. A genuinely
  different identity and a different resulting antiderivative shape (no
  ``sin(2kx)`` term at all) -- not a bigger version of easy.
- hard: a TWO-TERM combination ``a*sin^2(kx) + b*cos^2(kx)`` (``a != b``,
  shared ``k``), forcing the learner to apply the identity to each term
  separately before combining -- two applications of the identity plus
  combining, not one.
- exam: a squared term combined with the ``tan^2`` identity,
  ``a*cos^2(kx) + b*tan^2(mx)``, with TWO DIFFERENT frequencies ``k != m`` --
  mixing TWO DIFFERENT identity types in one antiderivative AND tracking two
  independent inner frequencies, a genuine escalation over hard (which
  shares one frequency between its two sin^2/cos^2 terms).

**Misconceptions** (checked distinct via a real sympy ``Symbol('C')``, every
seed, unconditionally, from the named list):
- forgets the ``1/2`` power-reduction factor (uses the un-halved identity
  ``sin^2 x = 1 - cos 2x`` / ``cos^2 x = 1 + cos 2x``, doubling the whole
  power-reduction contribution to the antiderivative);
- sign error on the ``cos(2kx)`` term (swaps the sin^2/cos^2 identity signs);
- forgets the ``1/k`` chain-rule factor from integrating
  ``sin(2kx)``/``cos(2kx)``/``tan(kx)``;
- (medium/hard/exam-tan-part) applies ``1 + tan^2 x = sec^2 x`` but forgets
  the ``- x`` term (integrates only ``sec^2(kx)`` and omits the ``-1`` half
  of the identity);
- drops ``+ C`` (grader-enforced, confirmed directly).
"""

from __future__ import annotations

from sympy import Integer, cos, diff, sin, symbols, tan

from ..core import (
    check,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    phrase,
    pick_distractors,
    symbolically_equal,
)

x = symbols("x")


# --- Distinctness, worked around a real `mathgen.core._probe_differs` bug --
#
# `check_distinct`/`symbolically_equal` (compared as `expr + C` vs `other_expr + C`, the pattern
# `p13c_antideriv_notation_build.py` and this module both need for the "dropped +C" misconception)
# hit a genuine performance bug in `_probe_differs`: it computes `symbols_used` from `(left -
# right).free_symbols`, which is correctly EMPTY of `C` once it cancels out of the subtraction --
# but then substitutes only THOSE symbols into `left`/`right` themselves (not just the difference)
# to compute `scale`, so `left.subs({x...})` still contains the un-substituted free symbol `C`,
# `complex(...)` raises `TypeError`, and EVERY probe point is skipped via the `except: continue` --
# `_probe_differs` returns `False` (inconclusive) even for two expressions that a numeric probe
# would prove unequal in one line, forcing the full `simplify(left - right)` path on every single
# distinctness check. Confirmed directly: profiling this generator's `exam` tier showed one `make`
# call spending ~700ms in `symplify`/`trigsimp`, all from candidate-vs-answer distinctness checks
# that should each cost <1ms via the numeric probe.
#
# The workaround needs no fix to the shared helper: `expr1 + C` and `expr2 + C` are equal iff
# `expr1` and `expr2` are (the shared symbol `C` cancels identically, with no numeric substitution
# needed at all), and `expr1 + C` is equal to a plain `expr2` (no `+ C`) only in the degenerate case
# of a `C`-free difference identically zero, which never happens here by construction (`C` is a bare
# free symbol, never assigned a value) -- so a `has_c` mismatch is always distinct, and a `has_c`
# match reduces to comparing the two BARE expressions, never touching `C` at all.
def _values_equal(expr1, has_c1: bool, expr2, has_c2: bool) -> bool:
    if has_c1 != has_c2:
        return False
    return symbolically_equal(expr1, expr2)


def _add_distinct(pool: list, answer_expr, *candidates) -> None:
    """Append each ``(expr, has_c)`` candidate only if it is genuinely a
    different VALUE from ``(answer_expr, True)`` (the true canonical is
    always ``F + C``) and from every candidate already kept -- some
    collisions are expected structural coincidences (e.g. a=b would collapse
    the "sign error" candidate onto the answer, but a != b is enforced by
    construction below), not construction bugs.
    """
    kept = [(answer_expr, True)]
    for expr, has_c in candidates:
        if expr is None:
            continue
        if any(_values_equal(expr, has_c, k_expr, k_hc) for k_expr, k_hc in kept):
            continue
        kept.append((expr, has_c))
        pool.append((expr, has_c))


def _check_all_distinct(answer_expr, cands, msg: str) -> None:
    """Final unconditional safety net (the same discipline `check_distinct`
    is named for) over every kept candidate, using the bug-free
    ``_values_equal`` comparison above rather than `mathgen.core.check_distinct`
    itself, which would re-trigger the `_probe_differs` slowdown this module
    works around."""
    for expr, has_c in cands:
        check(not _values_equal(answer_expr, True, expr, has_c), msg)


def _value(expr) -> str:
    """Canonical VALUE string for `answer_text`/a distractor: Python's "**"
    power operator becomes "^" (forbidden anywhere in a learner-facing field
    by `double_star`); a bare "*" is fine outside `Question.text` itself
    (`star_operator` only checks the prompt). Matches
    `p13c_antideriv_notation_build.py`'s own `_value`."""
    return str(expr).replace("**", "^")


def _display(expr, has_c: bool) -> str:
    return f"{_value(expr)} + C" if has_c else _value(expr)


def _rand_nonzero(rng, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _sq_disp(name: str, k: int) -> str:
    """Prompt-facing display of a squared trig term: `sin^2(3x)`, `cos^2(x)`
    (k=1 drops the redundant "1x"). Never built from `str(sympy_expr)` --
    sympy's own ordering for `sin(k*x)**2` is `sin(k*x)**2`, not the
    `sin^2(kx)` shape the descriptor itself uses -- so this is hand-built,
    matching the descriptor's own notation exactly."""
    inner = f"{k}x" if k != 1 else "x"
    return f"{name}^2({inner})"


_EASY_TEMPLATES = (
    "Find ∫ ({integrand}) dx",
    "Evaluate ∫ ({integrand}) dx using the power-reduction identity",
    "Determine ∫ ({integrand}) dx",
)

_MEDIUM_TEMPLATES = (
    "Find ∫ ({integrand}) dx",
    "Evaluate ∫ ({integrand}) dx using 1 + tan^2(x) = sec^2(x)",
    "Determine ∫ ({integrand}) dx",
)

_HARD_TEMPLATES = (
    "Find ∫ ({integrand}) dx",
    "Evaluate ∫ ({integrand}) dx, applying the power-reduction identity to each term",
)

_EXAM_TEMPLATES = (
    "Find ∫ ({integrand}) dx",
    "Evaluate ∫ ({integrand}) dx",
)


def _finish(F, cands, text):
    answer_text = f"{_value(F)} + C"
    distractor_strs = pick_distractors(answer_text, [_display(e, hc) for e, hc in cands])
    return dict(
        text=text,
        latex=text,
        answer=str(F),
        answer_text=answer_text,
        answer_latex=latex_answer(str(F)) + " + C",
        distractors=distractor_strs,
    )


def _easy(rng):
    kind = rng.choice(["sin", "cos"])
    k = rng.randint(1, 20)
    if kind == "sin":
        integrand = sin(k * x) ** 2
        F = x / Integer(2) - sin(2 * k * x) / (4 * k)
    else:
        integrand = cos(k * x) ** 2
        F = x / Integer(2) + sin(2 * k * x) / (4 * k)
    check_equivalent(diff(F, x), integrand, f"easy[{kind}]: derivative mismatch k={k}")

    cands: list = []
    # forgets the 1/2 power-reduction factor -> uses sin^2=1-cos2kx (or
    # cos^2=1+cos2kx) un-halved, doubling the whole contribution.
    dropped_half = 2 * F
    # sign error on the cos(2kx) term -> swaps the sin^2/cos^2 identity signs.
    sign_swapped = (
        x / Integer(2) + sin(2 * k * x) / (4 * k) if kind == "sin" else x / Integer(2) - sin(2 * k * x) / (4 * k)
    )
    # forgets the 1/k chain-rule factor from integrating sin(2kx).
    dropped_k = x / Integer(2) - sin(2 * k * x) / 4 if kind == "sin" else x / Integer(2) + sin(2 * k * x) / 4
    # `pick_distractors` caps at n=3, and 4 candidates compete for 3 slots
    # here -- shuffle the three power-reduction/sign/chain-factor
    # misconceptions (keeping "dropped +C" first, always exercised) so no
    # one candidate is permanently squeezed out over many seeds, the same
    # discipline `p13c_antideriv_notation_build.py`'s exam tier uses.
    extra = [(dropped_half, True), (sign_swapped, True), (dropped_k, True)]
    rng.shuffle(extra)
    _add_distinct(cands, F, (F, False), *extra)
    _check_all_distinct(F, cands, f"easy[{kind}]: distractor equals answer")

    integrand_str = _sq_disp(kind, k)
    text = phrase(rng, _EASY_TEMPLATES, integrand=integrand_str)
    return _finish(F, cands, text)


def _medium(rng):
    k = rng.randint(1, 30)
    integrand = tan(k * x) ** 2
    F = tan(k * x) / k - x
    check_equivalent(diff(F, x), integrand, f"medium: derivative mismatch k={k}")

    cands: list = []
    # forgets the 1/k chain-rule factor integrating tan(kx).
    dropped_k = tan(k * x) - x
    # forgets the -x term: integrates only sec^2(kx), omits the -1 half of
    # the identity tan^2 x = sec^2 x - 1.
    dropped_minus_x = tan(k * x) / k
    _add_distinct(cands, F, (F, False), (dropped_k, True), (dropped_minus_x, True))
    _check_all_distinct(F, cands, "medium: distractor equals answer")

    integrand_str = _sq_disp("tan", k)
    text = phrase(rng, _MEDIUM_TEMPLATES, integrand=integrand_str)
    return _finish(F, cands, text)


def _hard(rng):
    k = rng.randint(1, 4)
    a, b = rng.sample([v for v in range(-6, 7) if v != 0], 2)
    integrand = a * sin(k * x) ** 2 + b * cos(k * x) ** 2
    F = (a + b) * x / Integer(2) + (b - a) * sin(2 * k * x) / (4 * k)
    check_equivalent(diff(F, x), integrand, f"hard: derivative mismatch a={a} b={b} k={k}")

    cands: list = []
    # forgets the 1/2 power-reduction factor on BOTH terms -> doubles the
    # whole antiderivative.
    dropped_half = 2 * F
    # sign error on the cos(2kx) term -> flips the sign of the sin(2kx)
    # coefficient (equivalent to swapping which identity's sign is used).
    sign_swapped = (a + b) * x / Integer(2) - (b - a) * sin(2 * k * x) / (4 * k)
    # forgets the 1/k chain-rule factor from integrating sin(2kx).
    dropped_k = (a + b) * x / Integer(2) + (b - a) * sin(2 * k * x) / 4
    # Same 4-candidates-for-3-slots shuffle as the easy tier (see its
    # comment) -- "dropped +C" always first and always exercised.
    extra = [(dropped_half, True), (sign_swapped, True), (dropped_k, True)]
    rng.shuffle(extra)
    _add_distinct(cands, F, (F, False), *extra)
    _check_all_distinct(F, cands, "hard: distractor equals answer")

    integrand_str = join_terms([(a, _sq_disp("sin", k)), (b, _sq_disp("cos", k))])
    text = phrase(rng, _HARD_TEMPLATES, integrand=integrand_str)
    return _finish(F, cands, text)


def _exam(rng):
    k, m = rng.sample(range(1, 4), 2)  # k != m: two independent frequencies
    a = _rand_nonzero(rng, 2, 6)
    b = _rand_nonzero(rng, 2, 6)
    integrand = a * cos(k * x) ** 2 + b * tan(m * x) ** 2
    F = (a * Integer(2) ** -1 - b) * x + a * sin(2 * k * x) / (4 * k) + b * tan(m * x) / m
    check_equivalent(diff(F, x), integrand, f"exam: derivative mismatch a={a} b={b} k={k} m={m}")

    cands: list = []
    # forgets the 1/2 power-reduction factor on the cos^2 term -> doubles
    # just that term's contribution (the cos-part), tan-part unchanged.
    dropped_half = (a - b) * x + a * sin(2 * k * x) / (2 * k) + b * tan(m * x) / m
    # sign error on the cos(2kx) term.
    sign_swapped = (a * Integer(2) ** -1 - b) * x - a * sin(2 * k * x) / (4 * k) + b * tan(m * x) / m
    # forgets the 1/k chain-rule factor from integrating sin(2kx) (cos-part).
    dropped_k = (a * Integer(2) ** -1 - b) * x + a * sin(2 * k * x) / 4 + b * tan(m * x) / m
    # forgets the -x term from the tan^2 identity: integrates only sec^2(mx),
    # omits the -1 half (drops the tan-part's -b*x contribution).
    dropped_minus_x = a * Integer(2) ** -1 * x + a * sin(2 * k * x) / (4 * k) + b * tan(m * x) / m
    # 5 candidates compete for 3 slots -- shuffle the four
    # non-"dropped-+C" misconceptions so all four rotate through over many
    # seeds (see the easy/hard tiers' comment for the same discipline).
    extra = [(dropped_half, True), (sign_swapped, True), (dropped_k, True), (dropped_minus_x, True)]
    rng.shuffle(extra)
    _add_distinct(cands, F, (F, False), *extra)
    _check_all_distinct(F, cands, "exam: distractor equals answer")

    integrand_str = join_terms([(a, _sq_disp("cos", k)), (b, _sq_disp("tan", m))])
    text = phrase(rng, _EXAM_TEMPLATES, integrand=integrand_str)
    return _finish(F, cands, text)


@generator(
    "p13c_trig_identity_integrate",
    level=15,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13c_trig_identity_integrate(rng, tier="easy"):
    """See module docstring for the full ladder and the boundary against
    `integrate-trig`.

    easy:   sin^2(kx) or cos^2(kx), single term -- direct power-reduction.
    medium: tan^2(kx) -- the 1 + tan^2 x = sec^2 x identity, a different
            resulting shape (no sin(2kx) term).
    hard:   a*sin^2(kx) + b*cos^2(kx), a != b, shared k -- two applications
            of the power-reduction identity plus combining.
    exam:   a*cos^2(kx) + b*tan^2(mx), k != m -- two DIFFERENT identity
            types, two independent frequencies.
    """
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    return _exam(rng)
