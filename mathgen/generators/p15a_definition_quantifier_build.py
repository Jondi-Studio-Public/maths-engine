"""Generates `p15a_definition_quantifier_check` questions: identify the domain, codomain, quantifier scope or
function-ness of a stated finite relation.

`ucv1-definition-quantifier-check`. `answer: mcq`.

The contract lists four things to identify for a supplied finite relation:
domain, codomain, quantifier scope, and whether the relation is a function.
Each draw asks about ONE of these four (drawn uniformly), never all four at
once, with the OTHER named things (A, B, the relation) always fully stated
in the prompt so the aspect asked about is always well-posed.

A = a small set of distinct integers, B = a small set of distinct lowercase
letters -- deliberately different "alphabets" so a domain-side set can never
accidentally collide, as a string, with a codomain-side set (no
class is invented artificially -- this is a wording choice that keeps four
genuinely different sets visibly different, not a narrowing of the maths).

Three of the four aspects need the SAME "impossible option, never the key" device already used elsewhere
(module p15a_subgroup_isomorphism_build's finite subgroup test):

- is-it-a-function: "not a function: two different elements of A map to the
  same element of B" is ALWAYS a wrong reason (a shared output never makes a
  relation fail to be a function -- that is a statement about injectivity,
  a different property entirely), so it is offered every draw and is never
  the key.
- quantifier-scope: for the pair of statements (i) "for all a in A there
  exists b in B with (a,b) in R" and (ii) "there exists b in B such that for
  all a in A, (a,b) in R", (ii) LOGICALLY IMPLIES (i) (a single b that works
  for everyone is certainly a witness for every a), so "only (ii) is true"
  can never be the true state of any relation -- offered every draw, never
  the key.
"""

from __future__ import annotations

from mathgen.core import check, generator, latex_answer

_A_POOL = list(range(1, 10))
# Never "a" or "b" -- both are BOUND VARIABLE names in every statement this skill renders ("for
# every a in A there exists b in B...", "row a, column b"). A learner cannot tell the bound variable
# b from an element of B literally named b. p..w never appears as a bound variable anywhere in this
# module, so an element of B is always visually distinct from the quantifier prose around it.
_B_POOL = list("pqrstuvw")


def _draw_sets(rng, m: int, n: int):
    A = sorted(rng.sample(_A_POOL, m))
    B = sorted(rng.sample(_B_POOL, n))
    return A, B


def _set_text(items) -> str:
    return "{" + ", ".join(str(v) for v in items) + "}"


def _rel_text(R) -> str:
    return "{" + ", ".join(f"({a}, {b})" for a, b in R) + "}"


# ===========================================================================
# domain / codomain
# ===========================================================================


def _build_domain_codomain(rng, m: int, n: int):
    A, B = _draw_sets(rng, m, n)
    excluded_a = rng.choice(A)
    excluded_b = rng.choice(B)
    pool = [(a, b) for a in A if a != excluded_a for b in B if b != excluded_b]
    k = rng.randint(2, max(2, len(pool) - 1)) if len(pool) > 2 else len(pool)
    R = sorted(rng.sample(pool, min(k, len(pool))))
    check(len(R) >= 1, "domain/codomain construction produced an empty relation")
    active_domain = sorted({a for a, _ in R})
    active_codomain = sorted({b for _, b in R})
    check(active_domain != A, "domain construction failed to leave an element of A uncovered")
    check(active_codomain != B, "codomain construction failed to leave an element of B uncovered")
    return A, B, R, active_domain, active_codomain


def _domain_or_codomain_question(rng, m, n, which: str):
    A, B, R, active_domain, active_codomain = _build_domain_codomain(rng, m, n)
    if which == "domain":
        correct = A
        decoys = [B, active_domain, active_codomain]
    else:
        correct = B
        decoys = [A, active_domain, active_codomain]
    options_sets = [correct] + decoys
    check(len({tuple(s) for s in options_sets}) == 4, f"{which} options must all be distinct sets")
    labels = ["A", "B", "C", "D"]
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options_sets[i] for i in order]
    answer_idx = shuffled.index(correct)
    opts_text = "   ".join(f"{labels[i]}: {_set_text(shuffled[i])}" for i in range(4))
    # No convention sentence here. "R is a relation from A to B" already fixes domain=A, codomain=B
    # by the standard convention; the decoys (active_domain/active_codomain, the *projections* of R)
    # are what actually test whether the learner knows that convention.
    text = (
        f"Let A = {_set_text(A)} and B = {_set_text(B)}, and let R = {_rel_text(R)} "
        f"be a relation from A to B. What is the {which} of R? {opts_text}"
    )
    return text, labels[answer_idx]


# ===========================================================================
# quantifier scope
# ===========================================================================

_QS_CLASSES = ["both", "only_i", "neither"]
_QS_LABEL = {
    "both": "Both statements are true.",
    "only_i": "Only statement (i) is true.",
    "only_ii": "Only statement (ii) is true.",
    "neither": "Neither statement is true.",
}
_QS_STATEMENT_PREFIX = (
    "Consider these two statements about R: "
    "(i) for every a in A there exists b in B with (a, b) in R; "
    "(ii) there exists a single b in B such that for every a in A, "
    "(a, b) in R. "
)


def _build_quantifier_scope(rng, m: int, n: int, target: str):
    check(m >= 2 and n >= 2, "quantifier-scope needs at least 2 elements on each side")
    A, B = _draw_sets(rng, m, n)
    if target == "both":
        b_star = rng.choice(B)
        R = [(a, b_star) for a in A]
    elif target == "only_i":
        for _ in range(200):
            f = {a: rng.choice(B[:-1]) for a in A}  # never uses the last element of B
            if len(set(f.values())) >= 2:
                R = [(a, f[a]) for a in A]
                break
        else:
            raise AssertionError("could not build an only-i quantifier-scope draw")
    else:  # neither
        dropped = rng.choice(A)
        remaining = [a for a in A if a != dropped]
        f = {a: rng.choice(B) for a in remaining}
        R = [(a, f[a]) for a in remaining]
    return A, B, R


def _classify_quantifier_scope(A, B, R) -> str:
    R_set = set(R)
    i_true = all(any((a, b) in R_set for b in B) for a in A)
    ii_true = any(all((a, b) in R_set for a in A) for b in B)
    check(not (ii_true and not i_true), "quantifier-scope theorem violated: (ii) without (i)")
    if i_true and ii_true:
        return "both"
    if i_true:
        return "only_i"
    return "neither"


def _quantifier_scope_question(rng, m, n):
    target = rng.choice(_QS_CLASSES)
    A, B, R = _build_quantifier_scope(rng, m, n, target)
    recomputed = _classify_quantifier_scope(A, B, R)
    check(recomputed == target, "quantifier-scope construction mismatch")

    labels = ["A", "B", "C", "D"]
    keys = ["both", "only_i", "only_ii", "neither"]
    distractors = [k for k in keys if k != target]
    check(len(distractors) == 3, "quantifier-scope must offer exactly 3 distractors")
    order = keys[:]
    rng.shuffle(order)
    answer_idx = order.index(target)
    opts_text = "   ".join(f"{labels[i]}: {_QS_LABEL[order[i]]}" for i in range(4))
    text = (
        f"Let A = {_set_text(A)} and B = {_set_text(B)}, and let R = {_rel_text(R)} "
        f"be a relation from A to B. {_QS_STATEMENT_PREFIX}"
        f"Which is correct? {opts_text}"
    )
    return text, labels[answer_idx]


# ===========================================================================
# is it a function
# ===========================================================================

_FN_CLASSES = ["is_function", "fails_multi", "fails_missing"]
_FN_ALWAYS_WRONG = "output-collision"


def _fn_label(cls: str, culprit=None) -> str:
    if cls == "is_function":
        return "It IS a function: every element of A has exactly one image in R."
    if cls == "fails_multi":
        return f"NOT a function: {culprit} has more than one image in R."
    if cls == "fails_missing":
        return f"NOT a function: {culprit} has no image in R."
    return "NOT a function: two different elements of A map to the same element of B."


def _build_function_question(rng, m, n):
    check(n >= 2, "is-it-a-function needs at least 2 elements of B")
    target = rng.choice(_FN_CLASSES)
    A, B = _draw_sets(rng, m, n)
    f = {a: rng.choice(B) for a in A}
    culprit = None
    if target == "is_function":
        R = [(a, f[a]) for a in A]
    elif target == "fails_multi":
        culprit = rng.choice(A)
        other_b = rng.choice([b for b in B if b != f[culprit]])
        R = [(a, f[a]) for a in A] + [(culprit, other_b)]
    else:  # fails_missing
        culprit = rng.choice(A)
        R = [(a, f[a]) for a in A if a != culprit]

    degree = {a: sum(1 for x, _ in R if x == a) for a in A}
    if all(d == 1 for d in degree.values()):
        recomputed = "is_function"
    elif any(d >= 2 for d in degree.values()):
        recomputed = "fails_multi"
    else:
        recomputed = "fails_missing"
    check(recomputed == target, "is-it-a-function construction mismatch")
    if target != "is_function":
        check(degree[culprit] != 1, "is-it-a-function: named culprit is not actually the offending element")

    # The decoys name an element too. When the relation is a function every element has exactly one image, so any
    # element makes both decoys false.
    named = culprit if culprit is not None else A[0]
    labels = ["A", "B", "C", "D"]
    real_labels = {
        "is_function": _fn_label("is_function"),
        "fails_multi": _fn_label("fails_multi", named),
        "fails_missing": _fn_label("fails_missing", named),
    }
    options = [real_labels[c] for c in _FN_CLASSES] + [_fn_label("always_wrong")]
    check(len(set(options)) == 4, "is-it-a-function options must all display differently")
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    answer_text_full = real_labels[target]
    answer_idx = shuffled.index(answer_text_full)
    opts_text = "   ".join(f"{labels[i]}: {shuffled[i]}" for i in range(4))
    text = (
        f"Let A = {_set_text(A)} and B = {_set_text(B)}, and let R = {_rel_text(R)} "
        f"be a relation from A to B. Is R a function from A to B? {opts_text}"
    )
    return text, labels[answer_idx]


# ===========================================================================
# exam: relation given as a grid instead of a pair list
# ===========================================================================


def _grid_text(A, B, R) -> str:
    R_set = set(R)
    header = "     " + "  ".join(str(b) for b in B)
    rows = [header]
    for a in A:
        row = f"{a:>3}: " + "  ".join("X" if (a, b) in R_set else "-" for b in B)
        rows.append(row)
    return " | ".join(rows)


def _exam_question(rng, m, n):
    aspect = rng.choice(["domain", "codomain", "is_function", "quantifier_scope"])
    if aspect in ("domain", "codomain"):
        A, B, R, active_domain, active_codomain = _build_domain_codomain(rng, m, n)
        if aspect == "domain":
            correct = A
            decoys = [B, active_domain, active_codomain]
        else:
            correct = B
            decoys = [A, active_domain, active_codomain]
        options_sets = [correct] + decoys
        check(len({tuple(s) for s in options_sets}) == 4, "exam domain/codomain options must be distinct")
        labels = ["A", "B", "C", "D"]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [options_sets[i] for i in order]
        answer_idx = shuffled.index(correct)
        opts_text = "   ".join(f"{labels[i]}: {_set_text(shuffled[i])}" for i in range(4))
        # No convention sentence (see _domain_or_codomain_question above).
        text = (
            f"Let A = {_set_text(A)} and B = {_set_text(B)}. A relation R from A to B is "
            f"given by the grid below (X marks a pair (a, b) in R, row a, column b): "
            f"{_grid_text(A, B, R)}. What is the {aspect} of R? {opts_text}"
        )
        return text, labels[answer_idx]
    if aspect == "is_function":
        return _function_question_grid(rng, m, n)
    return _quantifier_scope_question_grid(rng, m, n)


def _function_question_grid(rng, m, n):
    check(n >= 2, "is-it-a-function needs at least 2 elements of B")
    target = rng.choice(_FN_CLASSES)
    A, B = _draw_sets(rng, m, n)
    f = {a: rng.choice(B) for a in A}
    culprit = None
    if target == "is_function":
        R = [(a, f[a]) for a in A]
    elif target == "fails_multi":
        culprit = rng.choice(A)
        other_b = rng.choice([b for b in B if b != f[culprit]])
        R = [(a, f[a]) for a in A] + [(culprit, other_b)]
    else:
        culprit = rng.choice(A)
        R = [(a, f[a]) for a in A if a != culprit]

    degree = {a: sum(1 for x, _ in R if x == a) for a in A}
    if all(d == 1 for d in degree.values()):
        recomputed = "is_function"
    elif any(d >= 2 for d in degree.values()):
        recomputed = "fails_multi"
    else:
        recomputed = "fails_missing"
    check(recomputed == target, "exam is-it-a-function construction mismatch")

    # The decoys name an element too. When the relation is a function every element has exactly one image, so any
    # element makes both decoys false.
    named = culprit if culprit is not None else A[0]
    labels = ["A", "B", "C", "D"]
    real_labels = {
        "is_function": _fn_label("is_function"),
        "fails_multi": _fn_label("fails_multi", named),
        "fails_missing": _fn_label("fails_missing", named),
    }
    options = [real_labels[c] for c in _FN_CLASSES] + [_fn_label("always_wrong")]
    check(len(set(options)) == 4, "exam is-it-a-function options must all display differently")
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    answer_text_full = real_labels[target]
    answer_idx = shuffled.index(answer_text_full)
    opts_text = "   ".join(f"{labels[i]}: {shuffled[i]}" for i in range(4))
    text = (
        f"Let A = {_set_text(A)} and B = {_set_text(B)}. A relation R from A to B is given "
        f"by the grid below (X marks a pair (a, b) in R, row a, column b): "
        f"{_grid_text(A, B, R)}. Is R a function from A to B? {opts_text}"
    )
    return text, labels[answer_idx]


def _quantifier_scope_question_grid(rng, m, n):
    target = rng.choice(_QS_CLASSES)
    A, B, R = _build_quantifier_scope(rng, m, n, target)
    recomputed = _classify_quantifier_scope(A, B, R)
    check(recomputed == target, "exam quantifier-scope construction mismatch")

    labels = ["A", "B", "C", "D"]
    keys = ["both", "only_i", "only_ii", "neither"]
    order = keys[:]
    rng.shuffle(order)
    answer_idx = order.index(target)
    opts_text = "   ".join(f"{labels[i]}: {_QS_LABEL[order[i]]}" for i in range(4))
    text = (
        f"Let A = {_set_text(A)} and B = {_set_text(B)}. A relation R from A to B is given "
        f"by the grid below (X marks a pair (a, b) in R, row a, column b): "
        f"{_grid_text(A, B, R)}. {_QS_STATEMENT_PREFIX}Which is correct? {opts_text}"
    )
    return text, labels[answer_idx]


@generator(
    "p15a_definition_quantifier_check",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p15a_definition_quantifier_check(rng, tier="easy"):
    if tier == "easy":
        m, n = 2, 3
    elif tier == "medium":
        m, n = 3, 3
    elif tier == "hard":
        m, n = 4, 4
    else:
        m, n = 3, 3

    if tier == "exam":
        text, answer_text = _exam_question(rng, m, n)
    else:
        aspect = rng.choice(["domain", "codomain", "is_function", "quantifier_scope"])
        if aspect in ("domain", "codomain"):
            text, answer_text = _domain_or_codomain_question(rng, m, n, aspect)
        elif aspect == "is_function":
            text, answer_text = _build_function_question(rng, m, n)
        else:
            text, answer_text = _quantifier_scope_question(rng, m, n)

    labels = ["A", "B", "C", "D"]
    distractors = [lbl for lbl in labels if lbl != answer_text]
    check(len(distractors) == 3, "definition-quantifier-check must offer exactly 3 distractors")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[f"Answer: {answer_text}."],
        distractors=distractors,
    )
