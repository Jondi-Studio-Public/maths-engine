"""Secondary-number cluster: one new
generator for one genuinely new graded operation in the ``secondary-number``
domain, framework ``acara-f10-v9``.

``p10e_exact_approx_repeated_calc`` (AC9M10N01, prerequisite
``f10-measurement-error-interpretation``) -- given a stated chain of
arithmetic operations and a stated rounding rule to apply after every
intermediate step, calculate the discrepancy between the exact final result
and the result obtained by applying that repeated rounding. Per the
Scope: "Given a stated sequence of
arithmetic operations and a stated rounding rule to apply after each
intermediate step, calculate the discrepancy between the exact final result
and the result obtained by applying that repeated rounding."

**This is the one row in the whole library that must NOT round early**, and it is the natural home for a
fixed-rational-pi distractor as CONTENT. Construction uses ``math.pi`` (a "true pi" per the rule -- the rule
explicitly names ``sympy.pi``/``math.pi`` as the two acceptable routes) throughout, and rounds only when the
STATED rounding rule says to (every intermediate step for the approximate pathway; once, at the very end, for
the exact pathway). The independent check rebuilds the exact pathway from scratch as a fresh ``sympy`` string
expression (built from exact ``Fraction`` literals, never a decimal literal, so ``sympify`` never silently
produces a ``Float`` in place of a ``Rational``) and evaluates it with ``evalf(30)`` -- a genuinely different,
symbolic route from this module's own fast ``(pi-coefficient, constant)`` tracking (see ``_exact_state``
below).

**Construction, every seed**: a starting value ``v0`` (a 2-decimal-place
Fraction, e.g. 3.45 -- 2dp, not 1dp, so that even a single "multiply by a
1dp k" step at ``easy``/``medium`` genuinely overflows the stated rounding
precision ``d=2`` and produces real rounding error from the first step
onward; a 1dp start would make ``easy``'s single rounding event a no-op on
every seed, since a 1dp x 1dp product is still <= 2dp). A chain of
operations, each one of ``mul`` (multiply by a stated 1dp decimal, e.g. 1.5,
2.2), ``add`` (add a stated integer or 1dp decimal), or -- hard tier only,
exactly once, at a uniformly random position in the chain (never forced to
be first or last) -- ``mul_pi`` (multiply by pi). ``easy``/``medium`` force
the FIRST operation to be ``mul`` (guaranteeing the early rounding-error
seed above); ``hard`` does not need this forcing since pi's own irrational
expansion already guarantees genuine rounding error wherever it lands.

At fixed ``n_ops=2, d=2`` the easy tier's discrepancy was almost always exactly one unit in the last place
(``0.01``, 91% of 400 seeds): the step-1 rounding error is bounded at half a ULP regardless of which multiplier
is drawn, and with only one further step there was too little room for that bound to translate into a range of
final discrepancies. The fix varies two things for the easy tier ONLY (``has_pi=False`` and the received ``n_ops == 2``, i.e. the
bare easy preset): the stated rounding precision ``d`` now varies over ``{1, 2}`` instead of being fixed at 2,
and the SECOND op (the only other op at ``n_ops=2``) draws its multiplier from a wider pool
(``_MUL_POOL_EASY_SECOND``, up to 9.5 rather than 4.5) so the step-1 error, scaled by that multiplier, spreads
across several grid units rather than collapsing back onto one almost every time. ``n_ops`` itself is
deliberately left at 2 for easy: a d=3 precision would make the (2dp v0) x (1dp k1) first step exactly 3dp --
lossless at d=3 -- so "round after every step" and "round only at the end" would coincide on EVERY draw and
the discrepancy would always be zero, `_make_seed` would retry every draw until it raised. Widening ``d`` to
include 3 would need a wider ``n_ops`` too (so a later step can carry the divergence past the always-lossless
first one, as it already does at ``medium``/``hard``); that combination was judged out of scope here since ``d
in {1, 2}`` plus the wider second-op pool already meets the acceptance bar (>= 6 distinct answers, no
answer over ~35% share, 400 seeds) while keeping the easy tier's own 2-step shape intact. ``hard`` passes no
widened pool and is unaffected -- see ``_draw_ops_easy_medium``'s ``later_mul_pool`` default.

The same defect was measured at the MEDIUM tier too (the easy-tier fix did not reach it):
still 73% ``0.01`` at 400 seeds (7 distinct), re-confirmed
before the medium fix. At medium (``n_ops=3``), every surviving draw already has
at least one `mul` op after the forced-first one (an all-`add` tail is a
mathematically forced zero discrepancy here too, at every chain length, so
`_make_seed`'s own tolerance check rejects it every time), so the tail
multiplier pool is live on 100% of draws. Medium's own `d` cannot split its
mass across two different rendered strings the way easy's `d in {1, 2}`
does: a frozen test (``test_medium_answer_carries_exactly_the_stated_precision``)
pins ``d == 2`` on every medium seed, out of scope to weaken here (an
empirical check found `d=3` at medium's own `n_ops=3` draws fine on its own
merits, 0/200 `_MAX_TRIES` exhaustions -- the frozen test, not
draw-starvation, is what keeps `d` fixed here). Without a second `d` value
to split across, reaching the acceptance bar (>= 6 distinct answers, no
answer over ~35% share, 400 seeds) took a genuinely wider pool than easy's
own: a first attempt reusing `_MUL_POOL_EASY_SECOND` unmodified only reached
19 distinct / 42% top share, so a new, wider pool
(``_MUL_POOL_MEDIUM_LATER``, up to 14.5, not 9.5) is applied to every `mul`
op after the forced-first one via the new `medium_wide_mul` flag. Measured
after (400 seeds): 32 distinct answers, top ``0.01`` at 30% -- clears the
bar.

Every rounding EVENT in both pathways (each approximate-pathway intermediate
step, and the exact pathway's own single final rounding) is checked with
``mathgen.core.rounds_unambiguously`` before being accepted; a draw that puts
any rounding on (or within its default 0.05 margin of) a ".5" boundary is
resampled, bounded at ``_MAX_TRIES`` attempts. This is a stronger guard than
the more obvious wording ("resample on a zero/near-zero draw"), covers
every rounding event, not just the graded discrepancy, and was added because
a tie is a coin-flip a real calculator's tie-break convention could legally
disagree with this generator's ``round_half_up`` on.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import sympy as sp

from ..core import (
    check,
    check_distinct,
    exact_decimal_string,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

_MAX_TRIES = 1500

_V0_RANGE = (150, 950)  # scaled by 100 -> 1.50..9.50, 2dp
_MUL_POOL = [Fraction(n, 10) for n in (12, 15, 18, 22, 25, 35, 45)]
_ADD_POOL = [Fraction(n) for n in (3, 5, 7, 9, 11)] + [Fraction(n, 10) for n in (15, 25, 35, 45, 65)]

# The easy tier's SECOND op only (never the forced-first op, which keeps drawing from `_MUL_POOL` so
# the module docstring's "always overflows d=2" guarantee is untouched) draws from this wider pool
# instead, so the step-1 rounding error -- bounded at half a ULP no matter what multiplies it --
# gets scaled by a genuinely varied second factor (up to 9.5, not 4.5) rather than collapsing back
# onto "one unit in the last place" almost every time. Never used by hard (see `later_mul_pool`
# default below). Still easy's own pool only, unchanged, wider pool instead
# (`_MUL_POOL_MEDIUM_LATER` below), because a first attempt reusing this pool unmodified for medium
# was not wide enough to clear medium's own concentration bar.
_MUL_POOL_EASY_SECOND = [Fraction(n, 10) for n in (12, 15, 18, 22, 25, 35, 45, 55, 65, 75, 85, 95)]

# Every op AFTER the forced-first one, at MEDIUM only (`n_ops == 3, has_pi == False`), draws its
# multiplier from this pool instead of `_MUL_POOL_EASY_SECOND`. Medium's own `d` cannot vary (a
# frozen assertion, `test_medium_answer_carries_exactly_the_stated_precision`, pins `d == 2` on
# every medium seed -- confirmed empirically safe to lift at `d=3, n_ops=3`, 0/200 `_MAX_TRIES`
# exhaustions, but the frozen test still forbids it), so unlike easy, medium cannot split its
# concentration across two different `d` string representations; a wider pool than easy's own was
# needed to reach the same numeric bar working with `d` fixed. Measured (400 seeds, this pool): 32
# distinct answers, top `0.01` at 30% -- `_MUL_POOL_EASY_SECOND` alone (up to 9.5) only reached 19
# distinct / 42% top share, still over the ~35% bar.
_MUL_POOL_MEDIUM_LATER = [
    Fraction(n, 10) for n in (12, 15, 18, 22, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145)
]


def _draw_v0(rng) -> Fraction:
    return Fraction(rng.randint(*_V0_RANGE), 100)


def _draw_op(rng, kind: str, pool=None):
    if kind == "mul":
        return "mul", rng.choice(pool or _MUL_POOL)
    if kind == "add":
        return "add", rng.choice(_ADD_POOL)
    raise ValueError(f"unknown op kind {kind!r}")


def _draw_ops_easy_medium(rng, n_ops: int, later_mul_pool=None):
    """`add`'s own pool is entirely 2dp-grid-aligned (integers and 1dp decimals), so an `add` as the LAST op
    after a chain of length 1 is a mathematically forced no-op: with only one prior step, "round once at the
    end" and "round after every step" are the same single rounding operation by definition, and adding a
    grid-aligned constant afterwards cannot reintroduce a difference. `_make_seed`'s own discrepancy>=
    tolerance check correctly resamples every such draw away (measured: 2dp tier, n_ops=2, `add` as the 2nd op
    is rejected 100% of the time, 500/500 seeds), so at `n_ops == 2` (easy) the 2nd op is drawn as `mul` only,
    rather than wastefully drawing and discarding `add` half the time. From `n_ops == 3` (medium) onward the
    chain accumulates real rounding error over 2+ prior steps before the final op, so `add` is genuinely live
    there.

    `later_mul_pool`: every `mul` op
    AFTER the forced-first one uses this pool instead of `_MUL_POOL` when
    given. Defaults to `None` (meaning "use `_MUL_POOL`", via `_draw_op`'s
    own `pool or _MUL_POOL`). The easy tier's call site passes
    `_MUL_POOL_EASY_SECOND`; the medium tier's call site passes
    `_MUL_POOL_MEDIUM_LATER`, its own wider pool (see module docstring). The
    hard tier's call site (`_draw_ops_hard`, a separate function) never uses
    this at all."""
    ops = [_draw_op(rng, "mul")]  # forced first, see module docstring
    kinds = ["mul"] if n_ops == 2 else ["mul", "add"]
    for _ in range(n_ops - 1):
        ops.append(_draw_op(rng, rng.choice(kinds), pool=later_mul_pool))
    return ops


def _draw_ops_hard(rng, n_ops: int):
    ops = [_draw_op(rng, rng.choice(["mul", "add"])) for _ in range(n_ops - 1)]
    pi_index = rng.randint(0, n_ops - 1)
    ops.insert(pi_index, ("mul_pi", None))
    return ops


def _op_text(op) -> str:
    kind, k = op
    if kind == "mul":
        return f"multiply by {exact_decimal_string(k, 1)}"
    if kind == "add":
        return f"add {exact_decimal_string(k, 1)}"
    return "multiply by pi (its true, unrounded value)"


# -- exact pathway: value = a*pi + b, tracked as exact Fraction coefficients --


def _exact_state(v0: Fraction, ops):
    a, b = Fraction(0), v0
    for kind, k in ops:
        if kind == "mul":
            a, b = a * k, b * k
        elif kind == "add":
            b = b + k
        else:  # mul_pi
            a, b = b, Fraction(0)
    return a, b


def _exact_value_fraction(a: Fraction, b: Fraction) -> Fraction:
    """`a*pi + b` evaluated with `math.pi` (a true, unrounded pi to float
    precision) and converted to an exact `Fraction` of that float's own exact
    binary value -- never a rational substitute for pi anywhere in this
    route."""
    return Fraction(float(a) * math.pi + float(b))


def _sympy_frac(f: Fraction) -> str:
    return f"({f.numerator}/{f.denominator})"


def _build_expr_str(v0: Fraction, ops) -> str:
    s = _sympy_frac(v0)
    for kind, k in ops:
        if kind == "mul":
            s = f"({s})*{_sympy_frac(k)}"
        elif kind == "add":
            s = f"({s})+{_sympy_frac(k)}"
        else:
            s = f"({s})*pi"
    return s


def _verify_exact_final(v0: Fraction, ops, d: int, has_pi: bool, exact_final: Fraction) -> None:
    """Independent check: rebuild the exact pathway from scratch as a fresh
    sympy string expression (exact Fraction literals only, never a decimal
    literal -- `sympify` would silently give a Float, not a Rational, for
    "4.5") and evaluate with `evalf(30)`. A genuinely different route from
    this module's own `_exact_state` coefficient tracking."""
    expr_str = _build_expr_str(v0, ops)
    val = sp.sympify(expr_str).evalf(30)
    recomputed = round_half_up(Fraction(Decimal(str(val))), d)
    if has_pi:
        check(
            abs(recomputed - exact_final) <= stated_precision_tolerance(d),
            f"exact pathway mismatch: sympy route got {recomputed}, construction got {exact_final}",
        )
    else:
        check(
            recomputed == exact_final,
            f"exact pathway mismatch: sympy route got {recomputed}, construction got {exact_final}",
        )


def _approx_run(v0: Fraction, ops, d: int, skip_index: int | None = None):
    """One full run of the approximate pathway: round the running value to
    `d` places after every step EXCEPT `skip_index` (used by the
    skips-one-rounding-step misconception; `None` means round after every
    step, the correctly-applied rule). Returns the list of pre-rounding
    values seen (for the unambiguous-rounding guard) and the final value."""
    val = v0
    pre_round_values = []
    for i, (kind, k) in enumerate(ops):
        if kind == "mul":
            val = val * k
        elif kind == "add":
            val = val + k
        else:  # mul_pi
            val = Fraction(float(val) * math.pi)
        pre_round_values.append(val)
        if i != skip_index:
            val = round_half_up(val, d)
    return pre_round_values, val


def _verify_approx_final(v0: Fraction, ops, d: int, expected: Fraction) -> None:
    """Independent check: a fresh loop (not the one construction ran) that
    re-simulates the round-after-every-step rule with `round_half_up`, per
    a permitted route for this pathway."""
    val = v0
    for kind, k in ops:
        if kind == "mul":
            val = val * k
        elif kind == "add":
            val = val + k
        else:
            val = Fraction(float(val) * math.pi)
        val = round_half_up(val, d)
    check(val == expected, f"approximate pathway mismatch: fresh loop got {val}, construction got {expected}")


def _make_seed(rng, n_ops: int, has_pi: bool, d: int, easy_wide_mul: bool = False, medium_wide_mul: bool = False):
    """Draw a non-degenerate seed AND its full misconception candidate set
    together, retrying the WHOLE draw (never just the misconceptions) on any
    degeneracy: a rounding event sitting on a tie, a near-zero discrepancy,
    or -- found empirically, not anticipated -- a misconception distractor
    that happens to coincide with the true discrepancy for this specific
    draw (`skips-one-rounding-step` skipping a rounding event that turns out
    not to change the final rounded bucket is a genuine, occasional no-op,

    `easy_wide_mul`: forwarded to `_draw_ops_easy_medium` as its
    widened `later_mul_pool` (`_MUL_POOL_EASY_SECOND`) when set; only ever
    passed by the easy tier's own call site below
    `p10e_exact_approx_repeated_calc`. `medium_wide_mul`: the same
    mechanism, but forwarding the medium tier's own, wider pool
    (`_MUL_POOL_MEDIUM_LATER` -- see module docstring for why medium needs a
    wider pool than easy's) and only ever passed by the medium tier's own
    call site. Both default to `False` so every other call site (hard, and
    the existing tests that call `_make_seed` directly with 4 positional
    args) is unaffected, and the two are mutually exclusive in practice (one
    generator call is never both the easy and medium preset)."""
    for _ in range(_MAX_TRIES):
        v0 = _draw_v0(rng)
        if has_pi:
            ops = _draw_ops_hard(rng, n_ops)
        else:
            if easy_wide_mul:
                later_mul_pool = _MUL_POOL_EASY_SECOND
            elif medium_wide_mul:
                later_mul_pool = _MUL_POOL_MEDIUM_LATER
            else:
                later_mul_pool = None
            ops = _draw_ops_easy_medium(rng, n_ops, later_mul_pool=later_mul_pool)

        a, b = _exact_state(v0, ops)
        exact_pre_round = _exact_value_fraction(a, b)
        if not rounds_unambiguously(float(exact_pre_round), d):
            continue
        exact_final = round_half_up(exact_pre_round, d)

        pre_round_values, approx_final = _approx_run(v0, ops, d)
        if any(not rounds_unambiguously(float(v), d) for v in pre_round_values):
            continue

        discrepancy = abs(exact_final - approx_final)
        if discrepancy < stated_precision_tolerance(d):
            continue  # genuinely nonzero and clears its own stated precision

        # Order matches the documented misconception list, and matters: with 4 candidates at hard
        # tier but `pick_distractors`' universal n=3 cap, whichever candidate sorts last is
        # systematically dropped, not merely occasionally. rounds-pi-early is the flagship, on-theme
        # distractor at hard tier (the only tier it can ever appear on) so it is placed ahead of
        # skips-one-rounding-step, which is deliberately the one that starves at hard tier instead
        # -- documented, not accidental.
        reports_approx = approx_final
        reports_exact = exact_final
        skips_one = _skips_one_rounding_step_value(v0, ops, d, exact_final, approx_final)
        candidates = [reports_approx, reports_exact]
        if has_pi:
            candidates.append(_rounds_pi_early_value(v0, ops, d, approx_final, rng))
        candidates.append(skips_one)

        if any(abs(c - discrepancy) <= stated_precision_tolerance(d) for c in candidates):
            continue  # a misconception distractor coincides with the true discrepancy

        # Independent checks: real bugs must still raise, never be resampled away.
        _verify_exact_final(v0, ops, d, has_pi, exact_final)
        _verify_approx_final(v0, ops, d, approx_final)

        return v0, ops, exact_final, approx_final, discrepancy, candidates
    check(False, "could not draw a non-degenerate exact/approximate repeated-calculation seed")


def _rounds_pi_early_value(v0: Fraction, ops, d: int, approx_final: Fraction, rng) -> Fraction:
    """The on-theme misconception: re-run the ENTIRE exact pathway
    substituting a fixed rational pi (22/7 or the decimal literal 3.14) for
    true pi from the first
    operation onward, then compute the discrepancy against the (still
    correctly rounded, true-pi) approximate pathway. Only ever called at
    hard tier (the only tier using pi)."""
    wrong_pi = rng.choice([Fraction(22, 7), Fraction(314, 100)])
    val = v0
    for kind, k in ops:
        if kind == "mul":
            val = val * k
        elif kind == "add":
            val = val + k
        else:  # mul_pi, using the WRONG fixed rational pi throughout
            val = val * wrong_pi
    wrong_pi_rounded = round_half_up(val, d)
    return abs(wrong_pi_rounded - approx_final)


def _skips_one_rounding_step_value(
    v0: Fraction, ops, d: int, exact_final: Fraction, approx_final: Fraction
) -> Fraction:
    """Skip ONE intermediate rounding event (never the final step, so the
    misconception's own value is still displayable at `d` places) and report
    the resulting discrepancy against the true exact final.

    Tries skip positions 0..n-2 IN ORDER and takes the first that actually
    changes the result (`alt_approx != approx_final`): `round(x + k) ==
    round(x) + k` for a <=1dp `k` whenever the running value's fractional
    part is unaffected by the addition's own rounding boundary, so skipping
    the rounding immediately before an `add` step is very often a no-op --
    measured (500-seed sweep, easy/medium): fixing the skip position at
    `n - 2` made this misconception a no-op whenever ops[-1] was `add`
    (n - 2 is always a `mul` step in that draw shape), starving `add` out of
    the built questions entirely (0/500 `add` occurrences at both tiers).
    Trying every position and keeping the first live one fixes this without
    changing `_draw_ops_easy_medium`'s own op-kind mix."""
    n = len(ops)
    for skip_index in range(n - 1):  # never the final step (n - 1)
        _, alt_approx = _approx_run(v0, ops, d, skip_index=skip_index)
        if alt_approx != approx_final:
            return abs(exact_final - alt_approx)
    # Every skip position was a no-op for this specific draw (found
    # empirically to be possible, not merely hypothetical) -- the caller's
    # own retry loop in `_make_seed` resamples the whole draw when this
    # coincides with the true discrepancy or fails `check_distinct`, so
    # returning the last (still legitimate, just degenerate-for-this-draw)
    # candidate here is safe: it will simply trigger that resample.
    _, alt_approx = _approx_run(v0, ops, d, skip_index=n - 2)
    return abs(exact_final - alt_approx)


def _dp_words(d: int) -> str:
    """ "1 decimal place" vs "2 decimal places" -- needed, once the easy tier's own `d` can be 1, not just always
    plural."""
    return "decimal place" if d == 1 else "decimal places"


def _build_question(
    rng, n_ops: int, has_pi: bool, d: int, easy_wide_mul: bool = False, medium_wide_mul: bool = False
) -> dict:
    v0, ops, exact_final, approx_final, discrepancy, candidates = _make_seed(
        rng, n_ops, has_pi, d, easy_wide_mul=easy_wide_mul, medium_wide_mul=medium_wide_mul
    )

    # Belt-and-braces: _make_seed only accepts a draw once every candidate
    # already clears this same margin, so these can never fail here -- kept
    # as unconditional checks per convention (every candidate, always).
    check_distinct(discrepancy, *candidates, msg="a misconception distractor equals the true discrepancy")
    for candidate in candidates:
        check(
            abs(candidate - discrepancy) > stated_precision_tolerance(d),
            f"distractor {candidate} is within the stated-precision tolerance of {discrepancy}",
        )

    op_sentence = "; then ".join(_op_text(op) for op in ops)
    op_sentence = op_sentence[0].upper() + op_sentence[1:]
    v0_text = exact_decimal_string(v0, 2)
    pi_clause = " (using the true, unrounded value of pi where it appears)" if has_pi else ""
    dp_words = _dp_words(d)
    text = (
        f"Start with {v0_text}. {op_sentence}, in that order. Work out the result two ways: "
        f"(1) using full precision throughout{pi_clause}, rounding only the FINAL result to "
        f"{d} {dp_words}; (2) rounding the running value to {d} {dp_words} after every "
        f"single step above. What is the discrepancy (the positive difference) between the two "
        f"final results, to {d} {dp_words}?"
    )

    answer_text = stated_precision_string(discrepancy, d)
    distractor_texts = [stated_precision_string(v, d) for v in candidates]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=numeric_answer(discrepancy),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Full precision, final result rounded once: {stated_precision_string(exact_final, d)}.",
            f"Rounding after every step: {stated_precision_string(approx_final, d)}.",
            f"Discrepancy: {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, distractor_texts),
    )


@generator(
    "p10e_exact_approx_repeated_calc",
    level=9,
    presets={
        "easy": dict(n_ops=2, has_pi=False, d=2),
        "medium": dict(n_ops=3, has_pi=False, d=2),
        "hard": dict(n_ops=3, has_pi=True, d=2),
    },
)
def p10e_exact_approx_repeated_calc(rng, n_ops=2, has_pi=False, d=2):
    """Compare an exact repeated calculation with the same calculation
    rounded after every step. See the module docstring for the pi
    handling and the misconception derivations. ``has_pi=True`` additionally
    randomises `n_ops` over {3, 4} and `d` over {2, 3} per the hard-tier
    contract; the preset's own `n_ops=3`/`d=2` are only the bare-call
    defaults.

    When ``has_pi`` is `False` and the received `n_ops` is 2 (the easy preset, and also a bare call with no
    args at all -- same branch, same fix, since the easy preset's own values ARE the function's defaults) this
    additionally randomises `d` over {1, 2} and widens the pool the SECOND op's multiplier draws from, so the
    discrepancy is not almost always one unit in the last place. See the module docstring for why `d=3`/wider
    `n_ops` were not also added here.

    When ``has_pi`` is `False` and the received `n_ops` is 3 (the medium preset) every `mul` op AFTER the
    forced-first one draws from its own, wider pool (`_MUL_POOL_MEDIUM_LATER`) the same way -- see the module
    docstring for the measured before/after concentration and why medium needed a wider pool than easy's own.
    Unlike easy, `d` stays fixed at 2 for medium (a frozen test pins it; see the module docstring)."""
    easy_wide_mul = False
    medium_wide_mul = False
    if has_pi:
        n_ops = rng.choice([3, 4])
        d = rng.choice([2, 3])
    elif n_ops == 2:
        d = rng.choice((1, 2))
        easy_wide_mul = True
    elif n_ops == 3:
        # `d` stays fixed at 2 here (unlike easy): a frozen assertion in
        # test_p10e_exact_approx_repeated_calc_build.py
        # (`test_medium_answer_carries_exactly_the_stated_precision`) pins medium's `d == 2` on
        # every seed, and an empirical check found d=3/n_ops=3 draws fine on its own (0/200
        # `_MAX_TRIES` exhaustions) -- so it is the frozen test, not a draw-starvation problem, that
        # keeps `d` fixed here. Both ops after the forced-first one draw their multiplier from
        # medium's own wider pool (`_MUL_POOL_MEDIUM_LATER`, NOT `_MUL_POOL_EASY_SECOND` -- measured
        # insufficient alone at medium, see module docstring) via `medium_wide_mul`, exactly as
        # `_draw_ops_easy_medium`'s own `later_mul_pool` already supports.
        medium_wide_mul = True
    return _build_question(rng, n_ops, has_pi, d, easy_wide_mul=easy_wide_mul, medium_wide_mul=medium_wide_mul)
