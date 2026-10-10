"""Generates six university-core topics (`p15d_automaton_acceptance`, `p15d_formal_derivation_check`,
`p15d_dominated_convergence`, `p15d_chart_jacobian`, `p15d_validation_calibration` and `p15d_shape_sorting`)
from one module.

Six NEW topics, one module, per-skill sections below:

- ``p15d_automaton_acceptance``      -- ucv1-finite-automaton-acceptance (mcq)
- ``p15d_formal_derivation_check``   -- ucv1-formal-derivation-check (mcq)
- ``p15d_dominated_convergence``     -- ucv1-dominated-convergence-check (mcq)
- ``p15d_chart_jacobian``            -- ucv1-manifold-chart-jacobian (list)
- ``p15d_validation_calibration``    -- ucv1-validation-calibration-check (mcq)
- ``p15d_shape_sorting``             -- f10-shape-sorting-algorithm (mcq)

Every helper reused from ``mathgen/core.py`` (``check``, ``check_equivalent``, ``check_distinct``,
``check_pairwise_distinct``, ``pick_distractors``, ``latex_answer``, ``stated_precision_string``,
``stated_precision_tolerance``). No generator sets ``latex``: ``make`` derives safe LaTeX from ``text`` via
``latex_prompt`` automatically (mathgen/core.py:126-128), so every prompt here is authored as plain text only,
sidestepping the English-inside-maths hazard entirely for the prompt side. Word-shaped
``answer_text`` values use ``latex_answer(answer_text, plain=True)`` for the same reason on the answer side
(the ``propositional_formula_equivalence`` precedent, university_algebra.py).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import (
    check,
    check_distinct,
    check_pairwise_distinct,
    generator,
    latex_answer,
    pick_distractors,
    stated_precision_string,
    stated_precision_tolerance,
)

# ---------------------------------------------------------------------------
# 1. ucv1-finite-automaton-acceptance -- p15d_automaton_acceptance
# ---------------------------------------------------------------------------
#
# Accept/reject alone is binary. The contract's own "by
# following its transition table" supports a COMPOSITE answer: the verdict
# AND the state the run ends in. Answer space is every (state, verdict) pair
# actually reachable by the automaton's own accepting-set membership (state
# fixes the verdict, so the answer space size equals ``num_states``, never
# fewer than 5 here -- guarantees >= 3 distractors of the form
# "<verdict>, ends in <other state>" on every draw, satisfying the "three
# distractors every draw" boundary without relying on chance).

_STATE_NAMES = ["q0", "q1", "q2", "q3", "q4"]
_SYMBOLS = ["a", "b", "c"]


def _simulate_dfa(transitions: dict, start: str, word: str) -> str:
    current = start
    for ch in word:
        current = transitions[(current, ch)]
    return current


@generator(
    "p15d_automaton_acceptance",
    level=17,
    presets={
        "easy": dict(alphabet_size=2, num_accepting=1, word_length=3, compact_word=False),
        "medium": dict(alphabet_size=2, num_accepting=2, word_length=5, compact_word=False),
        "hard": dict(alphabet_size=3, num_accepting=2, word_length=6, compact_word=False),
        "exam": dict(alphabet_size=3, num_accepting=3, word_length=9, compact_word=True),
    },
)
def p15d_automaton_acceptance(rng, alphabet_size=2, num_accepting=1, word_length=4, compact_word=False):
    """Trace a deterministic finite automaton on a supplied word."""
    states = _STATE_NAMES
    symbols = _SYMBOLS[:alphabet_size]
    transitions = {(s, sym): rng.choice(states) for s in states for sym in symbols}
    accepting = set(rng.sample(states, num_accepting))

    if compact_word:
        # EXAM ONLY: the word is given as a repeated block raised to a
        # power (e.g. "a^40b" or "(ab)^25c") -- 40-60 symbols long once
        # expanded, far too many to trace one transition at a time by hand.
        # The learner must instead find the CYCLE the automaton falls into
        # once it starts repeating the block, and reason by period rather
        # than by brute-force tracing -- a step the hard tier never asks
        # for (there the word is short and spelled out in full).
        block_len = rng.choice([1, 2, 3])
        block = "".join(rng.choice(symbols) for _ in range(block_len))
        repeats = rng.randint(15, 25)
        suffix_len = rng.randint(0, 2)
        suffix = "".join(rng.choice(symbols) for _ in range(suffix_len))
        word = block * repeats + suffix
        block_notation = block if block_len == 1 else f"({block})"
        word_display = f"{block_notation}^{repeats}" + (suffix if suffix else "")
    else:
        word = "".join(rng.choice(symbols) for _ in range(word_length))
        word_display = word

    final_state = _simulate_dfa(transitions, states[0], word)
    # Independent route: fold over the word with functools.reduce instead of
    # the for-loop used above, walking the SAME table a structurally
    # different way.
    import functools

    final_state_check = functools.reduce(lambda cur, ch: transitions[(cur, ch)], word, states[0])
    check(final_state == final_state_check, "automaton trace mismatch")

    verdict = "Accepted" if final_state in accepting else "Rejected"
    answer_text = f"{verdict}, ends in {final_state}"

    table_lines = [f"from {s} on {sym} go to {transitions[(s, sym)]}" for s in states for sym in symbols]
    accepting_text = ", ".join(sorted(accepting))
    text = (
        f"A deterministic finite automaton has states {', '.join(states)}, "
        f"starting state {states[0]}, and accepting states {accepting_text}. "
        f"Its transition table is: {'; '.join(table_lines)}. "
        f'Trace the automaton on the word "{word_display}". State the verdict '
        "(Accepted or Rejected) and the state the run ends in, in the form "
        "'Accepted, ends in qN' or 'Rejected, ends in qN'."
    )

    alternatives = []
    for s in states:
        if s == final_state:
            continue
        other_verdict = "Accepted" if s in accepting else "Rejected"
        alternatives.append(f"{other_verdict}, ends in {s}")
    check(len(alternatives) >= 3, "not enough automaton-state distractors")
    check(answer_text not in alternatives, "an automaton distractor equals the answer")
    check(len(set(alternatives)) == len(alternatives), "automaton distractors collide")

    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f'Starting from {states[0]}, the word "{word_display}" drives the '
            f"automaton through its transition table to {final_state}.",
            f"{final_state} is {'an accepting' if final_state in accepting else 'a non-accepting'} "
            "state, so the word is " + verdict.lower() + ".",
        ],
        distractors=pick_distractors(answer_text, alternatives, n=3),
    )


# ---------------------------------------------------------------------------
# 2. ucv1-formal-derivation-check -- p15d_formal_derivation_check
# ---------------------------------------------------------------------------
#
# "Decide whether EACH line follows" -> "the first
# line that does not follow from the pinned rules, or every line is valid".
# Pinned rule set (stated in every prompt): Premise, Modus Ponens (MP),
# Conjunction Introduction (AndI), Conjunction Elimination (AndE-L/AndE-R),
# Disjunction Introduction (OrI). Each derived line's validity is a purely
# STRUCTURAL check (`_line_is_valid`) against whatever is actually written on
# its cited earlier lines -- so a corruption at line i is caught at line i
# regardless of what any later line, correctly re-applying its own rule to
# i's (possibly wrong) content, does afterwards; the FIRST invalid line is
# always well-defined. >= 4 options every draw: >= 3 derived lines (any one
# could be the first failure) + "every line is valid".

# "Conjunction Elimination" alone does not say WHICH conjunct is claimed --
#: a rendered rule name that cannot be told apart from its mirror image
#: means a "wrong side" corruption is not actually checkable (any conjunct
#: is a valid elimination under the unqualified name), so the two
#: elimination directions are named distinctly here; a line labelled
#: "left conjunct" that shows the right one is then genuinely, uncontestably
#: invalid, and the independent oracle checks
#: exactly this qualified name, never the unqualified one.
_RULE_NAMES = {
    "MP": "Modus Ponens",
    "AndI": "Conjunction Introduction",
    "AndE-L": "Conjunction Elimination (left conjunct)",
    "AndE-R": "Conjunction Elimination (right conjunct)",
    "OrI": "Disjunction Introduction",
}


def _fmt_formula(f) -> str:
    if f[0] == "atom":
        return f[1]
    if f[0] == "and":
        return f"{_fmt_formula(f[1])} & {_fmt_formula(f[2])}"
    if f[0] == "or":
        return f"{_fmt_formula(f[1])} | {_fmt_formula(f[2])}"
    if f[0] == "implies":
        return f"{_fmt_formula(f[1])} -> {_fmt_formula(f[2])}"
    raise ValueError(f)


def _line_is_valid(lines, i) -> bool:
    formula, rule, refs = lines[i]
    cited = [lines[j][0] for j in refs]
    if rule == "Premise":
        return True
    if rule == "MP":
        a, aimpb = cited
        return aimpb[0] == "implies" and aimpb[1] == a and aimpb[2] == formula
    if rule == "AndI":
        a, b = cited
        return formula == ("and", a, b)
    if rule == "AndE-L":
        (aandb,) = cited
        return aandb[0] == "and" and formula == aandb[1]
    if rule == "AndE-R":
        (aandb,) = cited
        return aandb[0] == "and" and formula == aandb[2]
    if rule == "OrI":
        (a,) = cited
        return formula[0] == "or" and formula[1] == a
    raise ValueError(rule)


@generator(
    "p15d_formal_derivation_check",
    level=17,
    presets={
        # corrupt_prob = num_derived / (num_derived + 1) so "every line is valid"
        # has the SAME 1/(num_derived+1) share as each individual corrupted-line
        # class under a uniform draw -- measured imbalanced at a flat 3/4 for
        # medium/hard/exam (the pool-variety check, 400 seeds); this is the fix.
        "easy": dict(num_derived=3, corrupt_prob=Fraction(3, 4), require_cascade=False),
        "medium": dict(num_derived=4, corrupt_prob=Fraction(4, 5), require_cascade=False),
        "hard": dict(num_derived=5, corrupt_prob=Fraction(5, 6), require_cascade=False),
        "exam": dict(num_derived=6, corrupt_prob=Fraction(6, 7), require_cascade=True),
    },
)
def p15d_formal_derivation_check(rng, num_derived=3, corrupt_prob=Fraction(3, 4), require_cascade=False):
    """Decide whether each line of a supplied derivation follows a pinned rule."""
    letters = rng.sample(["p", "q", "r", "s", "t"], 3)
    A, B, C = (("atom", letters[0]), ("atom", letters[1]), ("atom", letters[2]))

    lines = [
        (A, "Premise", ()),
        (("implies", A, B), "Premise", ()),
        (("and", B, C), "Premise", ()),
    ]
    n_prem = len(lines)

    # Fixed derived-line chain (a prefix is used per tier); index i below is
    # relative to `lines` (0-based, absolute).
    def add_mp():
        lines.append((B, "MP", (0, 1)))

    def add_andi():
        lines.append((("and", A, B), "AndI", (0, len(lines) - 1)))

    def add_ande_l():
        lines.append((B, "AndE-L", (2,)))

    def add_ori():
        # arbitrary extra disjunct: the OTHER premise atom (A)
        lines.append((("or", B, A), "OrI", (len(lines) - 1,)))

    def add_ande_r():
        lines.append((C, "AndE-R", (2,)))

    def add_andi2():
        lines.append((("and", C, B), "AndI", (len(lines) - 1, len(lines) - 3)))

    builders = [add_mp, add_andi, add_ande_l, add_ori, add_ande_r, add_andi2]
    # Vary the skeleton so the key
    # cannot be learned as "look at line 4". add_mp and add_ande_l both
    # derive the SAME formula B, from different premises (0,1) vs (2), and
    # every downstream builder that cites "whichever line is at position 0
    # / 2 / the immediately-preceding line" only ever depends on that
    # position holding formula B -- never on which of the two rules put it
    # there. Swapping them is therefore safe at every num_derived (the
    # other four builders use positional, not rule-specific, references),
    # unlike a full reorder of all six, which would require re-deriving
    # valid formulas for add_ori (needs literally B immediately before it)
    # and add_andi2 (needs len(lines)-3 to hold formula B); a same-position
    # swap keeps that guarantee without touching either of them.
    if rng.random() < 0.5:
        builders[0], builders[2] = builders[2], builders[0]
    for k in range(num_derived):
        builders[k]()

    derived_start = n_prem
    corrupt_idx = None
    if rng.random() < float(corrupt_prob):
        if require_cascade:
            # EXAM ONLY: restrict the corrupted line to one that at least
            # one LATER line actually cites, so the corruption's effect is
            # only visible by tracking what that earlier line established
            # -- a genuine two-step chain, never just the corrupted line
            # itself looked at in isolation (which is all the hard tier
            # ever needs, since its corrupt_idx may land on the final
            # derived line, with nothing built on it afterward).
            cascading = [
                i for i in range(derived_start, len(lines)) if any(i in lines[j][2] for j in range(i + 1, len(lines)))
            ]
            check(cascading, "no cascading derivation index available for the exam escalation")
            corrupt_idx = rng.choice(cascading)
        else:
            corrupt_idx = rng.randrange(derived_start, len(lines))
        formula, rule, refs = lines[corrupt_idx]
        cited = [lines[j][0] for j in refs]
        if rule == "MP":
            a, _ = cited
            wrong = a  # names the antecedent again instead of the consequent
        elif rule == "AndI":
            a, b = cited
            wrong = ("and", a, C if b != C else A)
        elif rule == "AndE-L":
            (aandb,) = cited
            wrong = aandb[2]  # names the wrong (right) conjunct
        elif rule == "AndE-R":
            (aandb,) = cited
            wrong = aandb[1]  # names the wrong (left) conjunct
        elif rule == "OrI":
            (a,) = cited
            wrong = ("or", C if a != C else A, formula[2])
        else:
            raise ValueError(rule)
        check(wrong != formula, "corruption produced the original formula")
        lines[corrupt_idx] = (wrong, rule, refs)

    # Independent re-derivation: scan every line from scratch with the
    # purely structural `_line_is_valid`, never consulting `corrupt_idx`.
    first_invalid = next((i for i in range(len(lines)) if not _line_is_valid(lines, i)), None)
    check(first_invalid == corrupt_idx, "first-invalid-line mismatch")

    rule_pool = ", ".join(sorted({_RULE_NAMES[r] for _, r, _ in lines if r != "Premise"}))
    line_texts = []
    for i, (formula, rule, refs) in enumerate(lines):
        cite = f" (lines {', '.join(str(j + 1) for j in refs)})" if refs else ""
        line_texts.append(f"Line {i + 1}: {_fmt_formula(formula)} -- {_RULE_NAMES.get(rule, rule)}{cite}")

    text = (
        f"Using only the pinned rule set (Premise, {rule_pool}), check the "
        f"following finite derivation line by line: {' | '.join(line_texts)}. "
        "Identify the first line that does not correctly follow from its "
        "cited earlier lines by the pinned rules, or state that every line is valid."
    )

    if first_invalid is None:
        answer_text = "Every line is valid"
    else:
        answer_text = f"Line {first_invalid + 1}"

    alternatives = [f"Line {i + 1}" for i in range(derived_start, len(lines))]
    alternatives.append("Every line is valid")
    alternatives = [a for a in alternatives if a != answer_text]
    check(len(alternatives) >= 3, "not enough derivation-line options")
    check(len(set(alternatives)) == len(alternatives), "derivation-line distractors collide")

    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            "Check each derived line's stated rule against its cited earlier "
            "lines in order; the first mismatch is the answer.",
            f"Result: {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, alternatives, n=3),
    )


# ---------------------------------------------------------------------------
# 3. ucv1-dominated-convergence-check -- p15d_dominated_convergence
# ---------------------------------------------------------------------------
#
# On a FINITE sample space with counting measure,
# every real-valued function is automatically integrable (a finite sum of
# finite values is always finite) -- there is no genuine way to fail
# integrability here, so the original fourth class was only ever
# constructed by printing the non-function-value string "g(x)=unbounded"
# and made the key spottable by that one word, with no reasoning required.
# The honest key space on a finite domain is exactly three classes: {all
# hypotheses met, pointwise convergence fails, the stated bound is violated
# at a supplied point}. "The bound is not integrable" stays as a fourth,
# PERMANENTLY WRONG option -- offered as a distractor on every draw (it is
# always false on a finite domain, which is itself worth testing), never
# constructed as the key. Construct-to-target: the target
# class is chosen FIRST, from the three honest classes, and the data built
# to realise it so each is drawn with real, checkable frequency.

_POINT_NAMES = ["x1", "x2", "x3", "x4"]


@generator(
    "p15d_dominated_convergence",
    level=17,
    presets={
        "easy": dict(domain_size=2, num_functions=3, hide_index=False),
        "medium": dict(domain_size=3, num_functions=4, hide_index=False),
        "hard": dict(domain_size=3, num_functions=5, hide_index=False),
        "exam": dict(domain_size=4, num_functions=6, hide_index=True),
    },
)
def p15d_dominated_convergence(rng, domain_size=3, num_functions=4, hide_index=False):
    """Check the dominated-convergence hypotheses for a finite sequence."""
    points = _POINT_NAMES[:domain_size]
    N = num_functions
    # Tail length grows with N (>= 2 always, so "stabilises" is never a
    # self-comparison): ceil(N/2), floored at 2.
    tail_len = max(2, -(-N // 2))
    tail_start = N - tail_len  # 0-based index of the first "tail" term

    target = rng.choice(["met", "no_converge", "no_dominate"])

    values = {p: [Fraction(rng.randint(-3, 3), rng.choice([1, 2])) for _ in range(N)] for p in points}
    # Force the tail (from tail_start on) to stabilise, for every point.
    for p in points:
        for n in range(tail_start, N):
            values[p][n] = values[p][tail_start]

    if target == "no_converge":
        broken = rng.choice(points)
        values[broken][N - 1] = values[broken][tail_start] + Fraction(1)

    bound = {}
    for p in points:
        bound[p] = max(abs(v) for v in values[p]) + Fraction(1)

    violated_point = None
    if target == "no_dominate":
        violated_point = rng.choice(points)
        # shrink the bound below the largest |f_n(x)| at that point.
        bound[violated_point] = (
            max(abs(v) for v in values[violated_point][:tail_start] or [Fraction(0)]) if tail_start > 0 else Fraction(0)
        )
        largest = max(abs(v) for v in values[violated_point])
        if bound[violated_point] >= largest:
            bound[violated_point] = largest - Fraction(1, 2)

    # Independent re-derivation from the constructed data (never from `target`).
    converges = all(values[p][n] == values[p][tail_start] for p in points for n in range(tail_start, N))
    dominates = all(all(abs(v) <= bound[p] for v in values[p]) for p in points)
    # Always True here: a finite sum of finite stated values is always
    # finite, so integrability never genuinely fails on this finite domain.
    integrable = True

    if not converges:
        derived = "Pointwise convergence fails"
    elif not dominates:
        derived = "The stated bound is violated at a supplied point"
    else:
        derived = "All hypotheses are met"

    expected = {
        "met": "All hypotheses are met",
        "no_converge": "Pointwise convergence fails",
        "no_dominate": "The stated bound is violated at a supplied point",
    }[target]
    check(derived == expected, "dominated-convergence classification mismatch")
    answer_text = derived

    func_lines = []
    for p in points:
        row = ", ".join(f"f{n + 1}({p})={values[p][n]}" for n in range(N))
        func_lines.append(row)
    bound_line = ", ".join(f"g({p})={bound[p]}" for p in points)
    if hide_index:
        # EXAM ONLY: an earlier wording here
        # ("becomes constant from SOME point onward") was undefined -- that
        # is trivially true of every finite sequence, from its own last
        # term alone, so a careful learner could never fail it and the
        # class was not actually checkable. State the generator's REAL
        # convention explicitly instead: constant across (at least) the
        # last `tail_len` listed values, named, at every sample point --
        # still a genuine step over hard (which states a single starting
        # index; here the learner must read off and check every one of the
        # `tail_len` named terms, not just the last one, at every point).
        tail_names = ", ".join(f"f{n + 1}" for n in range(tail_start, N))
        pointwise_clause = (
            f"the pointwise-limit test (the convention: the sequence is constant "
            f"across at least the last {tail_len} listed values, {tail_names}, at "
            "every sample point)"
        )
    else:
        pointwise_clause = (
            f"the pointwise-limit test (the sequence stabilises from f{tail_start + 1} onward at every point)"
        )
    text = (
        f"A finite sequence of functions f1, ..., f{N} on the finite sample points "
        f"{', '.join(points)} is given by: {'; '.join(func_lines)}. A stated bound g "
        f"is given by: {bound_line}. Using {pointwise_clause} and the domination "
        "test (|f_n(x)| <= g(x) for every n and every point) -- g is a finite list of "
        "stated values on this finite sample space, so it is always integrable -- "
        "decide whether the dominated-convergence hypotheses are met."
    )

    all_classes = [
        "All hypotheses are met",
        "Pointwise convergence fails",
        "The stated bound is violated at a supplied point",
        "The bound is not integrable",
    ]
    alternatives = [v for v in all_classes if v != answer_text]
    check(len(set(alternatives)) == len(alternatives), "dominated-convergence classes collide")
    check(len(alternatives) == 3, "dominated-convergence distractor count changed unexpectedly")
    check(
        "The bound is not integrable" != answer_text,
        "the bound is not integrable is a permanently-wrong decoy on a finite domain",
    )

    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Tail stabilisation from f{tail_start + 1} onward: {'holds' if converges else 'fails'}.",
            f"Domination |f_n(x)| <= g(x) for every n, x: {'holds' if dominates else 'fails'}.",
            f"g integrable over the finite sample points: {'yes' if integrable else 'no'}.",
            f"Result: {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, alternatives, n=3),
    )


# ---------------------------------------------------------------------------
# 4. ucv1-manifold-chart-jacobian -- p15d_chart_jacobian
# ---------------------------------------------------------------------------
#
# Sharp duplicate risk vs jacobian-two-variable (mcq): the boundary's own
# escape is "the list is the four entries followed by the determinant". This
# row is graded `list` (a typed 5-number list, not a choice among options)
# and the determinant is a NEW quantity the neighbour's answer never states
# -- computing it, and reading its sign/zero-ness, is exactly the contract's
# second clause ("decide whether it is locally nonsingular") without needing
# a separate boolean field. `singular_bias` forces a genuine escalation in
# KIND: easy/medium charts are always nonsingular at the stated point; hard
# mixes singular and nonsingular charts 50/50 (by making the two component
# gradients exactly proportional when a singular draw is chosen); exam is
# ALWAYS a singular chart (a strictly different kind of question -- every
# exam draw asks the learner to recognise and report a zero determinant).

_X, _Y, _H = sympy.symbols("x y h")


def _chart_jacobian_at(first, second, x0, y0):
    J = [[sympy.diff(expr, var).subs({_X: x0, _Y: y0}) for var in (_X, _Y)] for expr in (first, second)]
    return J


def _finite_difference_jacobian(first, second, x0, y0):
    substitutions = []
    for expr in (first, second):
        dx = sympy.expand(expr.subs({_X: x0 + _H, _Y: y0}))
        dy = sympy.expand(expr.subs({_X: x0, _Y: y0 + _H}))
        substitutions.append([sympy.diff(dx, _H).subs(_H, 0), sympy.diff(dy, _H).subs(_H, 0)])
    return substitutions


def _list_text(values) -> str:
    return ", ".join(str(v) for v in values)


@generator(
    "p15d_chart_jacobian",
    level=18,
    presets={
        "easy": dict(coefficient_hi=3, point_hi=2, singular_bias=Fraction(0)),
        "medium": dict(coefficient_hi=5, point_hi=3, singular_bias=Fraction(0)),
        "hard": dict(coefficient_hi=5, point_hi=3, singular_bias=Fraction(1, 2)),
        "exam": dict(coefficient_hi=6, point_hi=4, singular_bias=Fraction(1)),
    },
)
def p15d_chart_jacobian(rng, coefficient_hi=5, point_hi=3, singular_bias=Fraction(0)):
    """Compute a 2D chart's Jacobian at a point and its determinant."""
    want_singular = rng.random() < float(singular_bias)
    for _ in range(200):
        a = rng.randint(-coefficient_hi, coefficient_hi) or 1
        b = rng.randint(-coefficient_hi, coefficient_hi)
        c = rng.randint(-coefficient_hi, coefficient_hi)
        d = rng.randint(-coefficient_hi, coefficient_hi)
        first = a * _X**2 + b * _X * _Y + c * _Y + d
        if want_singular:
            # k == 1 would make the second component identical to the
            # first (up to a constant), which makes the swapped-rows
            # distractor collide with the answer -- excluded.
            k = rng.choice([-2, -1, 2, 3])
            offset = rng.randint(-3, 3)
            second = k * first + offset
        else:
            e = rng.randint(-coefficient_hi, coefficient_hi)
            f = rng.randint(-coefficient_hi, coefficient_hi)
            g = rng.randint(-coefficient_hi, coefficient_hi)
            kk = rng.randint(-3, 3)
            second = e * _X + f * _Y**2 + g * _X * _Y + kk
        x0 = rng.randint(-point_hi, point_hi)
        y0 = rng.randint(-point_hi, point_hi)
        J = _chart_jacobian_at(first, second, x0, y0)
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        if want_singular and det != 0:
            continue
        if not want_singular and det == 0:
            continue
        break
    else:
        raise RuntimeError("could not construct a chart with the requested singularity")

    # Independent route: finite-difference substitution (construction used
    # sympy.diff directly), and sympy's own Matrix.det (a different
    # determinant algorithm from the hand expansion above).
    fd = _finite_difference_jacobian(first, second, x0, y0)
    check(J == fd, "chart Jacobian finite-difference mismatch")
    det_check = sympy.Matrix(J).det()
    check(det == det_check, "chart Jacobian determinant mismatch")

    answer = [int(J[0][0]), int(J[0][1]), int(J[1][0]), int(J[1][1]), int(det)]
    answer_text = _list_text(answer)

    def poly_display(expr):
        # `**`/bare `*` read as raw Python/text-quality defects; same fix as
        # `university_foundations.py::_poly_display`.
        return str(sympy.expand(expr)).replace("**", "^").replace("*", "")

    text = (
        f"A 2-dimensional chart is given by u(x,y) = {poly_display(first)} and "
        f"v(x,y) = {poly_display(second)}. At the point (x, y) = ({x0}, {y0}), "
        "give the chart's Jacobian matrix as [du/dx, du/dy, dv/dx, dv/dy] in that "
        "row-major order, followed by its exact determinant, as five numbers."
    )

    # ``ordered_list`` (mathgen.grading.grade_ordered_list), not ``list`` -- `grade_list` is
    # order-insensitive on the SET of five values, so a transpose/swapped-rows candidate (a real,
    # common misconception: naming the same five numbers in the wrong slots) graded CORRECT under it
    # even though it names the same numbers as the answer. `ordered_list` compares position by
    # position, so those exact-same-values-wrong-order candidates are now genuinely wrong, and are
    # restored below as distractors since they now clear the grader honestly.
    j11, j12, j21, j22 = answer[0], answer[1], answer[2], answer[3]
    transpose = [j11, j21, j12, j22, det]  # swap du/dy <-> dv/dx; det(A) = det(A^T)
    swapped_rows = [j21, j22, j11, j12, -det]  # swap the two rows; swapping rows negates det

    origin_J = _chart_jacobian_at(first, second, 0, 0)
    origin_det = origin_J[0][0] * origin_J[1][1] - origin_J[0][1] * origin_J[1][0]
    origin = [int(origin_J[0][0]), int(origin_J[0][1]), int(origin_J[1][0]), int(origin_J[1][1]), int(origin_det)]

    perturbed_candidates = []
    flat = [J[0][0], J[0][1], J[1][0], J[1][1]]
    for idx in range(4):
        cand = flat[:]
        cand[idx] = cand[idx] + 1
        p_det = cand[0] * cand[3] - cand[1] * cand[2]
        perturbed_candidates.append([int(v) for v in cand] + [int(p_det)])
    neg_perturbed_candidates = []
    for idx in range(4):
        cand = flat[:]
        cand[idx] = cand[idx] - 1
        p_det = cand[0] * cand[3] - cand[1] * cand[2]
        neg_perturbed_candidates.append([int(v) for v in cand] + [int(p_det)])

    candidate_lists = [transpose, swapped_rows, origin, *perturbed_candidates, *neg_perturbed_candidates]
    seen = set()
    deduped = []
    for c in candidate_lists:
        # NOTE: local name deliberately not `text` -- this function's outer
        # `text` (the rendered question prompt) lives in the same scope.
        cand_text = _list_text(c)
        if cand_text == answer_text or cand_text in seen:
            continue
        seen.add(cand_text)
        deduped.append(cand_text)
    alternatives = deduped
    check(len(alternatives) >= 3, "not enough chart-Jacobian distractors")
    check(len(set(alternatives)) == len(alternatives), "chart-Jacobian distractors collide")
    check_distinct(answer_text, *alternatives)

    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Differentiate each component with respect to x and y, then substitute ({x0}, {y0}).",
            f"det = ({answer[0]})({answer[3]}) - ({answer[1]})({answer[2]}) = {det}, "
            f"so the chart is {'singular' if det == 0 else 'nonsingular'} there.",
        ],
        distractors=pick_distractors(answer_text, alternatives, n=3),
    )


# ---------------------------------------------------------------------------
# 5. ucv1-validation-calibration-check -- p15d_validation_calibration
# ---------------------------------------------------------------------------
#
# "Compute calibration error AND select over- or
# under-confident" -> composite options (error value to a stated precision,
# direction); >= 2 error values x 2 directions; the convention (equal-count
# bins, absolute-difference, unweighted mean per bin) is stated in every
# prompt.


@generator(
    "p15d_validation_calibration",
    level=18,
    presets={
        "easy": dict(num_points=4, num_bins=2, dp=2, equal_width=True),
        "medium": dict(num_points=6, num_bins=2, dp=2, equal_width=True),
        "hard": dict(num_points=6, num_bins=3, dp=3, equal_width=True),
        "exam": dict(num_points=8, num_bins=3, dp=3, equal_width=False),
    },
)
def p15d_validation_calibration(rng, num_points=4, num_bins=2, dp=2, equal_width=True):
    """Compute a binned calibration error and its over/under-confidence direction."""
    p_pool = [Fraction(k, 8) for k in range(1, 8)]
    for _ in range(200):
        if equal_width:
            edges = [Fraction(i, num_bins) for i in range(num_bins + 1)]
        else:
            # EXAM ONLY: unequal-width bins, with the boundaries STATED
            # explicitly rather than computed from num_bins -- the learner
            # must read the stated edges, not assume an equal division, a
            # step the hard tier never needs (its bins are always [0,
            # 1/3), [1/3, 2/3), [2/3, 1] etc).
            interior = sorted(rng.sample(p_pool, num_bins - 1))
            edges = [Fraction(0), *interior, Fraction(1)]
            gaps = {edges[i + 1] - edges[i] for i in range(num_bins)}
            if len(gaps) < 2:  # accidentally equal-width; resample
                continue

        def _bin_of(p):
            for i in range(num_bins):
                if edges[i] <= p < edges[i + 1]:
                    return i
            return num_bins - 1  # p == edges[-1] == 1

        preds = [rng.choice(p_pool) for _ in range(num_points)]
        outcomes = [rng.choice([0, 1]) for _ in range(num_points)]
        avg_pred_overall = Fraction(sum(preds), num_points)
        avg_obs_overall = Fraction(sum(outcomes), num_points)
        if avg_pred_overall == avg_obs_overall:
            continue
        bins = [[] for _ in range(num_bins)]
        for p, o in zip(preds, outcomes):
            bins[_bin_of(p)].append((p, o))
        if any(not b for b in bins):
            continue
        total_error = Fraction(0)
        for b in bins:
            n_b = len(b)
            avg_p = Fraction(sum(p for p, _ in b), n_b)
            avg_o = Fraction(sum(o for _, o in b), n_b)
            total_error += Fraction(n_b, num_points) * abs(avg_p - avg_o)
        # A common misconception: skip binning entirely (one global bin).
        # Resample if it happens to coincide with the binned answer within
        # the grader's own stated-precision tolerance -- a real distractor
        # must clear that gap by construction, on every draw shipped.
        unbinned_error = abs(avg_pred_overall - avg_obs_overall)
        if abs(unbinned_error - total_error) <= stated_precision_tolerance(dp):
            continue
        break
    else:
        raise RuntimeError("could not construct a balanced calibration draw")

    ece_rounded = _round_half_up_frac(total_error, dp)
    direction = "over-confident" if avg_pred_overall > avg_obs_overall else "under-confident"
    answer_text = f"ECE = {stated_precision_string(ece_rounded, dp)}, {direction}"

    # Independent route: recompute total error as a weighted sum over
    # bins built by re-scanning the (point, bin) assignment from scratch,
    # never reusing `total_error`'s running accumulation. Re-derives each
    # point's bin from the stated `edges` boundaries directly (not via
    # `_bin_of`, which `total_error` already used) so the check is a
    # genuinely independent route, correct for both equal- and
    # unequal-width edges.
    check_bins = [[] for _ in range(num_bins)]
    for p, o in zip(preds, outcomes):
        edge = num_bins - 1
        for i in range(num_bins):
            if edges[i] <= p < edges[i + 1]:
                edge = i
                break
        check_bins[edge].append((p, o))
    recomputed = Fraction(0)
    for b in check_bins:
        if not b:
            continue
        n_b = len(b)
        recomputed += Fraction(n_b, num_points) * abs(
            Fraction(sum(p for p, _ in b), n_b) - Fraction(sum(o for _, o in b), n_b)
        )
    check(recomputed == total_error, "calibration error recomputation mismatch")

    unbinned_rounded = _round_half_up_frac(unbinned_error, dp)
    wrong_direction = "under-confident" if direction == "over-confident" else "over-confident"
    alternatives = [
        f"ECE = {stated_precision_string(ece_rounded, dp)}, {wrong_direction}",
        f"ECE = {stated_precision_string(unbinned_rounded, dp)}, {direction}",
        f"ECE = {stated_precision_string(unbinned_rounded, dp)}, {wrong_direction}",
    ]
    check(answer_text not in alternatives, "a calibration distractor equals the answer")
    check(len(set(alternatives)) == len(alternatives), "calibration distractors collide")

    points_text = "; ".join(f"p{i + 1}={preds[i]}, o{i + 1}={outcomes[i]}" for i in range(num_points))
    edges_text = ", ".join(f"bin {i + 1}: [{edges[i]}, {edges[i + 1]})" for i in range(num_bins))
    if equal_width:
        bin_clause = f"Sort the predictions into {num_bins} equal-width bins by predicted probability ({edges_text}, the last bin closed at 1)."
    else:
        # EXAM ONLY: the bins are NOT equal-width --
        # the learner must read the stated edges rather than assume an
        # equal division, a step the hard tier never asks for.
        bin_clause = f"Sort the predictions into the {num_bins} STATED bins, which are not equal width: {edges_text}, the last bin closed at 1."
    text = (
        f"A model reports {num_points} validation predictions and observed outcomes: "
        f"{points_text}. {bin_clause} Using the "
        "expected calibration error convention (for each bin, take the absolute "
        "difference between the bin's mean predicted probability and its mean "
        "observed outcome, then combine the bins weighted by their share of the "
        f"{num_points} points), compute the calibration error to {dp} decimal places "
        "and state whether the model is over-confident or under-confident overall "
        "(comparing the overall mean prediction to the overall mean outcome)."
    )

    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Per-bin |mean predicted - mean observed|, weighted by bin share, "
            f"sums to {total_error} (~{stated_precision_string(ece_rounded, dp)}).",
            f"Overall mean prediction {avg_pred_overall} vs mean outcome {avg_obs_overall}: {direction}.",
        ],
        distractors=pick_distractors(answer_text, alternatives, n=3),
    )


def _round_half_up_frac(value: Fraction, places: int) -> Fraction:
    import math

    scale = 10**places
    scaled = value * scale
    magnitude = math.floor(scaled + Fraction(1, 2))
    return Fraction(magnitude, scale)


# ---------------------------------------------------------------------------
# 6. f10-shape-sorting-algorithm -- p15d_shape_sorting
# ---------------------------------------------------------------------------
#
# Narrowed kernel: given a STATED shape-sorting procedure (numbered tests,
# never the shape's own geometric name) and a shape's stated facts (side
# lengths, and where relevant a right-angle fact and a parallel-sides fact,
# in f10-polygon-classification's own declarative style), trace the
# procedure and state the resulting bin. Leaves are procedure-defined
# labels ("Bin A".."Bin J"), never ordinary shape names, so the answer is
# unreachable from geometry knowledge alone.
#
# The previous design assigned one test NAME per tree
# LEVEL regardless of branch history, which is fine for statistical balance
# but not for logical coherence -- it could ask "exactly 3 sides?" then
# "exactly 4 sides?" on the SAME path (the second test's answer is already
# forced, and its "yes" branch is mathematically unreachable), and it
# described the shape as a bare boolean-attribute list ("all of its sides
# are not all equal in length") rather than a shape a learner reads facts
# from. Both are fixed by building the tree from real, named triangle/
# quadrilateral TYPES (each with a fixed, geometrically consistent
# attribute vector -- see `_TRIANGLE_ATTRS`/`_QUAD_ATTRS`) rather than from
# independent per-level booleans: every test is asked only where the
# remaining candidate types actually differ on it (so both branches are
# always reachable by construction), and the rendered shape states side
# lengths -- from which `equal_sides`/`sides3` are read, not told -- plus a
# right-angle sentence and a parallel-sides sentence only when this tier's
# tree asks about them, mirroring f10-polygon-classification's own
# declarative-fact convention.


def _leaf(name):
    return ("leaf", name)


def _node(test, yes_branch, no_branch):
    return ("node", test, yes_branch, no_branch)


def _trace(tree, attrs):
    node = tree
    while node[0] != "leaf":
        _, test, yes_branch, no_branch = node
        node = yes_branch if attrs[test] else no_branch
    return node[1]


def _all_leaves(tree, out=None):
    if out is None:
        out = []
    if tree[0] == "leaf":
        out.append(tree[1])
    else:
        _all_leaves(tree[2], out)
        _all_leaves(tree[3], out)
    return out


def _all_tests(tree, out=None):
    if out is None:
        out = set()
    if tree[0] != "leaf":
        out.add(tree[1])
        _all_tests(tree[2], out)
        _all_tests(tree[3], out)
    return out


# Named triangle/quadrilateral types, each a fixed, geometrically genuine
# attribute vector (never an invented boolean combination -- every entry
# below is a real, standard shape). `_attrs_for` adds the `sides3`/
# `has_parallel_pair`/`has_second_parallel_pair` derived keys.
_TRIANGLE_ATTRS = {
    "equilateral": dict(equal_sides=True, right_angle=False),
    "right_scalene": dict(equal_sides=False, right_angle=True),
    "scalene": dict(equal_sides=False, right_angle=False),
}
_QUAD_ATTRS = {
    "square": dict(equal_sides=True, right_angle=True, parallel="both"),
    "rhombus": dict(equal_sides=True, right_angle=False, parallel="both"),
    "rectangle": dict(equal_sides=False, right_angle=True, parallel="both"),
    "parallelogram": dict(equal_sides=False, right_angle=False, parallel="both"),
    "right_trapezium": dict(equal_sides=False, right_angle=True, parallel="one"),
    "isosceles_trapezium": dict(equal_sides=False, right_angle=False, parallel="one"),
    "kite": dict(equal_sides=False, right_angle=False, parallel="none"),
}
_ALL_TYPES = [("triangle", t) for t in _TRIANGLE_ATTRS] + [("quad", t) for t in _QUAD_ATTRS]


def _attrs_for(shape, type_name):
    if shape == "triangle":
        a = dict(_TRIANGLE_ATTRS[type_name])
        a["sides3"] = True
        return a
    a = dict(_QUAD_ATTRS[type_name])
    a["sides3"] = False
    a["has_parallel_pair"] = a["parallel"] in ("one", "both")
    a["has_second_parallel_pair"] = a["parallel"] == "both"
    return a


# Hand-built per-tier trees over real types (never a generic per-level
# builder): every internal node's test genuinely splits the types that can
# still reach it, so both branches are reachable and no test's answer is
# ever already forced by an earlier test on the same path.
#   easy:   sides3? -> equal_sides? (triangle only) / single quad bin.
#   medium: + right_angle, on BOTH branches (square/rhombus/rectangle/
#           parallelogram all differ on it; equilateral+right_angle is
#           impossible, so that leaf is terminal one level earlier).
#   hard:   + has_second_parallel_pair, splitting rectangle/right_trapezium
#           and parallelogram/isosceles_trapezium (both pairs already
#           guaranteed to have >= 1 parallel pair, so "second pair too?"
#           is genuinely informative there).
#   exam:   + has_parallel_pair, asked ONE level higher in the "not equal,
#           not right" quad branch to split off kite (no parallel pair at
#           all) before the hard tier's own second-pair question -- a step
#           the hard tier never asks.
_TREE_EASY = _node("sides3", _node("equal_sides", _leaf("Bin A"), _leaf("Bin B")), _leaf("Bin C"))

_TREE_MEDIUM = _node(
    "sides3",
    _node("equal_sides", _leaf("Bin A"), _node("right_angle", _leaf("Bin B"), _leaf("Bin C"))),
    _node(
        "equal_sides",
        _node("right_angle", _leaf("Bin D"), _leaf("Bin E")),
        _node("right_angle", _leaf("Bin F"), _leaf("Bin G")),
    ),
)

_TREE_HARD = _node(
    "sides3",
    _node("equal_sides", _leaf("Bin A"), _node("right_angle", _leaf("Bin B"), _leaf("Bin C"))),
    _node(
        "equal_sides",
        _node("right_angle", _leaf("Bin D"), _leaf("Bin E")),
        _node(
            "right_angle",
            _node("has_second_parallel_pair", _leaf("Bin F"), _leaf("Bin G")),
            _node("has_second_parallel_pair", _leaf("Bin H"), _leaf("Bin I")),
        ),
    ),
)

_TREE_EXAM = _node(
    "sides3",
    _node("equal_sides", _leaf("Bin A"), _node("right_angle", _leaf("Bin B"), _leaf("Bin C"))),
    _node(
        "equal_sides",
        _node("right_angle", _leaf("Bin D"), _leaf("Bin E")),
        _node(
            "right_angle",
            _node("has_second_parallel_pair", _leaf("Bin F"), _leaf("Bin G")),
            _node(
                "has_parallel_pair", _node("has_second_parallel_pair", _leaf("Bin H"), _leaf("Bin I")), _leaf("Bin J")
            ),
        ),
    ),
)

_TREES = {"easy": _TREE_EASY, "medium": _TREE_MEDIUM, "hard": _TREE_HARD, "exam": _TREE_EXAM}

#: Two independently-drawn wordings per test -- prompt-text variety only,
#: never a math change.
_TEST_PROMPT_VARIANTS = {
    "sides3": ["Does the shape have exactly 3 sides?", "Is the number of sides of the shape exactly 3?"],
    "equal_sides": ["Are all of the shape's sides equal in length?", "Is every side of the shape the same length?"],
    "right_angle": ["Does the shape have at least one right angle?", "Does the shape contain a right angle?"],
    "has_parallel_pair": [
        "Does the shape have at least one pair of parallel sides?",
        "Is at least one pair of the shape's opposite sides parallel?",
    ],
    "has_second_parallel_pair": [
        "Does the shape have a SECOND pair of parallel sides too?",
        "Are both pairs of the shape's opposite sides parallel?",
    ],
}


def _describe_tree(tree, prompts):
    order = []

    def collect(node):
        if node[0] == "leaf":
            return
        order.append(node)
        collect(node[2])
        collect(node[3])

    collect(tree)
    number_of = {id(n): i + 1 for i, n in enumerate(order)}

    def branch_text(node):
        if node[0] == "leaf":
            return f"assign {node[1]}"
        return f"go to Test {number_of[id(node)]}"

    lines = []
    for node in order:
        _, test, yes_b, no_b = node
        num = number_of[id(node)]
        lines.append(f"Test {num}: {prompts[test]} If yes, {branch_text(yes_b)}. If no, {branch_text(no_b)}.")
    return " ".join(lines)


def _triangle_sides(rng, type_name):
    """Concrete, genuinely-consistent side lengths for a named triangle
    type, varied per draw -- the numbers a learner reads `equal_sides` off."""
    if type_name == "equilateral":
        s = rng.randint(3, 12)
        return [s, s, s]
    if type_name == "right_scalene":
        triple = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (7, 24, 25)])
        k = rng.randint(1, 2)
        return [v * k for v in triple]
    # scalene, no right angle: any triangle-inequality-respecting, all-
    # different, non-Pythagorean triple.
    for _ in range(50):
        a, b, c = sorted(rng.sample(range(3, 15), 3))
        if a + b > c and a * a + b * b != c * c:
            return [a, b, c]
    return [4, 5, 6]  # 4-5-6 is scalene and not right-angled; safe fallback


def _quad_sides(rng, type_name):
    """Concrete side lengths for a named quadrilateral type -- always
    matching the type's own `equal_sides` value exactly."""
    if type_name in ("square", "rhombus"):
        s = rng.randint(3, 12)
        return [s, s, s, s]
    if type_name in ("rectangle", "parallelogram"):
        a, b = rng.sample(range(3, 13), 2)
        return [a, b, a, b]
    if type_name == "isosceles_trapezium":
        leg = rng.randint(3, 10)
        base1, base2 = rng.sample([v for v in range(3, 15) if v != leg], 2)
        return [leg, base1, leg, base2]
    if type_name == "kite":
        a, b = rng.sample(range(3, 13), 2)
        return [a, a, b, b]
    # right_trapezium: four genuinely distinct lengths.
    return sorted(rng.sample(range(3, 16), 4))


def _types_reaching(tree):
    """Every real type that traces to each leaf of `tree` -- used to
    construct-to-target the leaf FIRST,
    since drawing a type uniformly first over-represents whichever leaf
    the most real types happen to reach (7 of the 10 named types are
    quadrilaterals, all landing in the same easy-tier leaf, which measured
    74%/200 -- a real concentration defect, not merely uneven wording)."""
    result = {leaf: [] for leaf in _all_leaves(tree)}
    for shape, type_name in _ALL_TYPES:
        leaf = _trace(tree, _attrs_for(shape, type_name))
        result[leaf].append((shape, type_name))
    return result


@generator(
    "p15d_shape_sorting",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p15d_shape_sorting(rng, tier="easy"):
    """Trace a stated shape-sorting procedure on a concrete shape's stated
    facts (side lengths, and a right-angle/parallel-sides fact where this
    tier's tree needs one)."""
    tree = _TREES[tier]
    # Sorted: `_all_tests` returns a set of strings, whose iteration order changes with PYTHONHASHSEED,
    # and the prompt wording below is drawn from `rng` in that order. Sorting makes the same seed give
    # the same question in every process.
    tests_used = sorted(_all_tests(tree))
    leaves = _all_leaves(tree)
    check(len(leaves) >= 3, "procedure has too few leaves")

    # Construct-to-target: pick the LEAF uniformly first,
    # then draw a real type that reaches it -- never draw the type first
    # and let the tree's own leaf sizes dictate the answer distribution.
    types_by_leaf = _types_reaching(tree)
    target_leaf = rng.choice(leaves)
    candidates = types_by_leaf[target_leaf]
    check(candidates, f"no real type reaches leaf {target_leaf!r}")
    shape, type_name = rng.choice(candidates)
    attrs = _attrs_for(shape, type_name)
    prompts = {t: rng.choice(_TEST_PROMPT_VARIANTS[t]) for t in tests_used}
    result = _trace(tree, attrs)
    check(result == target_leaf, "shape-sorting construct-to-target mismatch")

    # Independent route: re-derive the leaf by walking the tree a second
    # time via an explicit path list rather than the recursive walk above.
    node = tree
    path = []
    while node[0] != "leaf":
        _, test, yes_b, no_b = node
        taken = attrs[test]
        path.append((test, taken))
        node = yes_b if taken else no_b
    result_check = node[1]
    check(result == result_check, "shape-sorting trace mismatch")
    for test, taken in path:
        check(attrs[test] == taken, "shape-sorting path/attribute mismatch")

    if shape == "triangle":
        sides = _triangle_sides(rng, type_name)
        shape_sentence = "A triangle has sides measuring " + ", ".join(str(s) for s in sides) + " units."
    else:
        sides = _quad_sides(rng, type_name)
        shape_sentence = "A quadrilateral has sides measuring " + ", ".join(str(s) for s in sides) + " units."
    check(attrs["equal_sides"] == (len(set(sides)) == 1), "stated side lengths contradict equal_sides")
    check(attrs["sides3"] == (len(sides) == 3), "stated side count contradicts sides3")

    facts = [shape_sentence]
    if "right_angle" in tests_used:
        facts.append(
            "One of its interior angles is a right angle (90 degrees)."
            if attrs["right_angle"]
            else "None of its interior angles is a right angle."
        )
    if "has_parallel_pair" in tests_used or "has_second_parallel_pair" in tests_used:
        parallel = attrs.get("parallel", "none")
        facts.append(
            {
                "both": "Both pairs of its opposite sides are parallel.",
                "one": "Exactly one pair of its opposite sides is parallel.",
                "none": "No pair of its opposite sides is parallel.",
            }[parallel]
        )
    facts_text = " ".join(facts)

    procedure_text = _describe_tree(tree, prompts)
    text = (
        f"A shape-sorting procedure is defined by these numbered tests: {procedure_text} "
        f"{facts_text} Trace the procedure and state which bin it assigns the shape to."
    )

    answer_text = result
    alternatives = [leaf for leaf in leaves if leaf != answer_text]
    check(len(alternatives) >= 2, "not enough shape-sorting bins")
    check(len(set(alternatives)) == len(alternatives), "shape-sorting bin labels collide")

    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[f"Trace: " + " -> ".join(f"{t}={'yes' if v else 'no'}" for t, v in path), f"Result: {answer_text}."],
        distractors=pick_distractors(answer_text, alternatives, n=3),
    )
