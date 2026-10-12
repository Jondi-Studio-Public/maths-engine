"""Exam tiers for evaluating a propositional formula and for testing two formulas for equivalence.

Both tiers work on four atoms (p, q, r, s, so sixteen truth assignments) and use fully bracketed formulas of two to
four connectives, drawn at random so the counts vary.

``exd_x_sc4_prop_eval``: one formula F.
  (a) how many of the sixteen assignments make it true (a full truth-table count);
  (b) the probability that F is true when the assignment is chosen at random, which uses (a) (``follow_through``);
  (c) how many assignments make F true while one named atom has a stated value;
  (d) the same for a second atom;
  (e) the same for two atoms at once.

``exd_x_sc4_prop_equiv``: a pair F and G where G is F with one connective changed, so they look alike but are not
equivalent.
  (a) how many of the sixteen assignments give F and G different truth values;
  (b) the probability they agree, which uses (a) (``follow_through``);
  (c) how many assignments make both true;
  (d) how many make F true and G false;
  (e) how many make both false;
  (f) how many of the differing rows have a named atom at a stated value.

Every count is taken by exhaustive evaluation of the formula tree. A distractor is the count a learner reaches by
a named slip: counting the false rows, reading an implication as its converse or as a conjunction, or answering
the complementary event.
"""

from __future__ import annotations

import random
from fractions import Fraction
from functools import lru_cache
from itertools import product

from ..answers import Distractor, exact_value
from ..core import check, draw_until, generator, latex_answer, pick_distractors, rational_text
from ._exd_x_sc4_kit import count_part

_ATOMS = ("p", "q", "r", "s")
_N = 2 ** len(_ATOMS)
_ONE_SIDED = 0.25  # weight of a pair whose differing rows are all one way round, which makes (d) equal (a)
_POOL = 300  # candidates drawn per question, then grouped by the answer they give
_ROWS = list(product([True, False], repeat=len(_ATOMS)))
_SYMBOL = {"and": "∧", "or": "∨", "implies": "→"}


def _render(tree, top=True) -> str:
    op = tree[0]
    if op == "atom":
        return tree[1]
    if op == "not":
        return f"¬{_render(tree[1], top=False)}"
    text = f"{_render(tree[1], top=False)} {_SYMBOL[op]} {_render(tree[2], top=False)}"
    return text if top else f"({text})"


def _eval(tree, env) -> bool:
    op = tree[0]
    if op == "atom":
        return env[tree[1]]
    if op == "not":
        return not _eval(tree[1], env)
    a, b = _eval(tree[1], env), _eval(tree[2], env)
    return {"and": a and b, "or": a or b, "implies": (not a) or b}[op]


@lru_cache(maxsize=100_000)
def _table(tree) -> tuple[bool, ...]:
    """Truth value of `tree` on every row of `_ROWS`, in order. Cached by subtree, so a pool of similar formulas shares work."""
    op = tree[0]
    if op == "atom":
        i = _ATOMS.index(tree[1])
        return tuple(row[i] for row in _ROWS)
    if op == "not":
        return tuple(not v for v in _table(tree[1]))
    left, right = _table(tree[1]), _table(tree[2])
    if op == "and":
        return tuple(a and b for a, b in zip(left, right))
    if op == "or":
        return tuple(a or b for a, b in zip(left, right))
    return tuple((not a) or b for a, b in zip(left, right))


def _count(tree, where=lambda env: True) -> int:
    n = 0
    for row, value in zip(_ROWS, _table(tree)):
        if value and where(dict(zip(_ATOMS, row))):
            n += 1
    return n


def _differ(f, g) -> int:
    return sum(a != b for a, b in zip(_table(f), _table(g)))


def _map_ops(tree, change):
    """`tree` with every connective rewritten by `change(op, left, right)` -> new op."""
    if tree[0] == "atom":
        return tree
    if tree[0] == "not":
        return ("not", _map_ops(tree[1], change))
    left, right = _map_ops(tree[1], change), _map_ops(tree[2], change)
    return (change(tree[0], left, right), left, right)


def _converse(tree):
    if tree[0] == "atom":
        return tree
    if tree[0] == "not":
        return ("not", _converse(tree[1]))
    left, right = _converse(tree[1]), _converse(tree[2])
    return ("implies", right, left) if tree[0] == "implies" else (tree[0], left, right)


def _implication_as_and(tree):
    return _map_ops(tree, lambda op, left, right: "and" if op == "implies" else op)


def _binary_paths(tree, path=()):
    """Paths to every binary connective in `tree`."""
    if tree[0] == "atom":
        return []
    if tree[0] == "not":
        return _binary_paths(tree[1], (*path, 1))
    found = [path]
    for i in (1, 2):
        found += _binary_paths(tree[i], (*path, i))
    return found


def _replace_op(tree, path, new_op):
    if not path:
        return (new_op, tree[1], tree[2])
    parts = list(tree)
    parts[path[0]] = _replace_op(tree[path[0]], path[1:], new_op)
    return tuple(parts)


def _node(tree, path):
    for i in path:
        tree = tree[i]
    return tree


def _literal(rng: random.Random, atom: str):
    node = ("atom", atom)
    return ("not", node) if rng.random() < 0.3 else node


def _join(rng: random.Random, left, right, negate: float = 0.2):
    node = (rng.choice(("and", "or", "implies")), left, right)
    return ("not", node) if rng.random() < negate else node


def _formula(rng: random.Random, negate: float = 0.2):
    """Four literals, one per atom in random order, joined by three connectives in a random bracketing. `negate` is
    the chance that a joined pair is negated as a whole."""
    a, b, c, d = (_literal(rng, atom) for atom in rng.sample(_ATOMS, len(_ATOMS)))

    def join(left, right):
        return _join(rng, left, right, negate)

    shape = rng.randint(0, 3)
    if shape == 0:
        return join(join(a, b), join(c, d))
    if shape == 1:
        return join(join(join(a, b), c), d)
    if shape == 2:
        return join(a, join(join(b, c), d))
    return join(join(a, join(b, c)), d)


def _fraction_part(prompt, value, marks, working, cands, follow_through=None):
    value = Fraction(value)
    text = rational_text(value)
    options = []
    for wrong, mid in cands:
        wrong = Fraction(wrong)
        if wrong < 0 or wrong > 1 or wrong == value:
            continue
        options.append(Distractor(rational_text(wrong), mid, wrong))
    picked = pick_distractors(text, options, 3)
    check(len(picked) >= 2, "fewer than two usable distractors")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=picked,
        sensible_range=(0, 1),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


@generator("exd_x_sc4_prop_eval", level=16, presets={"exam": dict(tier="exam")})
def exd_x_sc4_prop_eval(rng: random.Random, tier: str = "exam") -> dict:
    """Count F's true rows, give the probability, then count the rows with a given atom's value, twice."""
    given_atom, other_atom = rng.sample(_ATOMS, 2)
    given_value, other_value = rng.choice([True, False]), rng.choice([True, False])

    def conditional(tree, atom=given_atom, value=given_value):
        return _count(tree, lambda env: env[atom] == value)

    def fine(t):
        half = _N // 2
        return (
            2 <= _count(t) <= _N - 2
            and 1 <= conditional(t) <= half - 1
            and 1 <= conditional(t, other_atom, other_value) <= half - 1
        )

    pool = [t for t in (_formula(rng) for _ in range(_POOL)) if fine(t)]
    check(pool, "no usable formula in the pool")
    # Pick the true-row count first, so a common count does not dominate the answers.
    by_count: dict[int, list] = {}
    for t in pool:
        by_count.setdefault(_count(t), []).append(t)
    tree = rng.choice(by_count[rng.choice(sorted(by_count))])
    k = _count(tree)
    cond = conditional(tree)
    text = _render(tree)
    word = "true" if given_value else "false"
    half = _N // 2

    conv, conj = _converse(tree), _implication_as_and(tree)
    a_part = count_part(
        f"Complete the truth table for F. For how many of the {_N} possible truth assignments to p, q, r and s "
        "is F true?",
        k,
        3,
        [f"Evaluate F on all {_N} rows: {k} of them give true."],
        [
            (_N - k, "found-complement-not-event"),
            (_count(conv), "swaps-hypothesis-and-conclusion"),
            (_count(conj), "treats-implication-as-and"),
            (k + 1, None),
            (k - 1, None),
        ],
        hi=_N,
    )
    b_part = _fraction_part(
        f"One truth assignment to p, q, r and s is chosen at random, all {_N} being equally likely. "
        "Hence find the probability that F is true.",
        Fraction(k, _N),
        1,
        [f"P(F true) = {k}/{_N}."],
        [
            (Fraction(_N - k, _N), "found-complement-not-event"),
            (Fraction(k, _N - k), "odds-for-probability"),
            (Fraction(k, _N - 1), "counted-from-wrong-total"),
        ],
        follow_through=f"part_a/{_N}",
    )
    c_part = count_part(
        f"For how many of the {_N} assignments is F true and {given_atom} {word}?",
        cond,
        2,
        [f"Of the {half} rows with {given_atom} {word}, F is true on {cond}."],
        [
            (k, "ignored-the-given-condition"),
            (k - cond, "used-opposite-condition-value"),
            (half - cond, "found-complement-not-event"),
            (cond + 1, None),
            (cond - 1, None),
        ],
        hi=half,
    )
    other = conditional(tree, other_atom, other_value)
    d_part = count_part(
        f"For how many of the {_N} assignments is F true and {other_atom} {'true' if other_value else 'false'}?",
        other,
        2,
        [f"Of the {half} rows with {other_atom} {'true' if other_value else 'false'}, F is true on {other}."],
        [
            (k, "ignored-the-given-condition"),
            (k - other, "used-opposite-condition-value"),
            (half - other, "found-complement-not-event"),
            (other + 1, None),
            (other - 1, None),
        ],
        hi=half,
    )
    two_a, two_b = rng.sample(_ATOMS, 2)
    two_va, two_vb = rng.choice([True, False]), rng.choice([True, False])
    both_cond = _count(tree, lambda env: env[two_a] == two_va and env[two_b] == two_vb)
    wa, wb = ("true" if two_va else "false"), ("true" if two_vb else "false")
    e_part = count_part(
        f"For how many of the {_N} assignments is F true, {two_a} {wa} and {two_b} {wb}?",
        both_cond,
        2,
        [f"Only the {_N // 4} rows with {two_a} {wa} and {two_b} {wb} count; F is true on {both_cond} of them."],
        [
            (k, "ignored-the-given-condition"),
            (conditional(tree, two_a, two_va), "ignored-the-given-condition"),
            (conditional(tree, two_b, two_vb), "ignored-the-given-condition"),
            (_N // 4 - both_cond, "found-complement-not-event"),
            (both_cond + 1, None),
            (both_cond - 1, None),
        ],
        hi=_N // 4,
    )
    return dict(text=f"Let F be the formula {text}.", parts=[a_part, b_part, c_part, d_part, e_part])


@generator("exd_x_sc4_prop_equiv", level=16, presets={"exam": dict(tier="exam")})
def exd_x_sc4_prop_equiv(rng: random.Random, tier: str = "exam") -> dict:
    """Compare F with a near copy G over all sixteen assignments. See the module docstring."""

    def draw(r):
        tree = _formula(r, 0.4)
        path = r.choice(_binary_paths(tree))
        current = _node(tree, path)[0]
        new_op = r.choice([o for o in ("and", "or", "implies") if o != current])
        return tree, _replace_op(tree, path, new_op)

    def fine(pair):
        f, g = pair
        return 1 <= _differ(f, g) <= _N - 4

    pool = [pair for pair in (draw(rng) for _ in range(_POOL)) if fine(pair)]
    check(pool, "no usable pair in the pool")
    # Part (d) asks for the rows where F is true and G false. Which formula is called F is free, so a pair is filed
    # in the direction(s) that make (d) non-zero: both when each formula is true somewhere the other is false, the
    # one otherwise (swapping one connective often leaves G weaker or stronger than F, which would leave (d) at 0
    # on every seed). Groups are keyed by (rows that differ, (d)) and weighted by 1 / sqrt(groups sharing the
    # differ count x groups sharing the (d) count), which evens out both counts, and by sqrt(group size), so a
    # group the pool holds only a pair or two of is not drawn over and over. A pair that differs in one direction
    # only gives (d) equal to (a), so those groups are down-weighted.
    by_key: dict[tuple[int, int], list] = {}
    for f_, g_ in pool:
        differ_ = _differ(f_, g_)
        f_only_ = sum(a and not b for a, b in zip(_table(f_), _table(g_)))
        if f_only_:
            by_key.setdefault((differ_, f_only_), []).append((f_, g_))
        if f_only_ < differ_:
            by_key.setdefault((differ_, differ_ - f_only_), []).append((g_, f_))
    keys = sorted(by_key)
    weights = [
        len(by_key[key]) ** 0.5
        / (sum(k[0] == key[0] for k in keys) * sum(k[1] == key[1] for k in keys)) ** 0.5
        * (_ONE_SIDED if key[0] == key[1] else 1)
        for key in keys
    ]
    f, g = rng.choice(by_key[rng.choices(keys, weights)[0]])
    rows = [dict(zip(_ATOMS, row)) for row in _ROWS]
    differ = sum(1 for env in rows if _eval(f, env) != _eval(g, env))
    both = sum(1 for env in rows if _eval(f, env) and _eval(g, env))
    f_true, g_true = _count(f), _count(g)
    agree = _N - differ
    check(both <= min(f_true, g_true), "rows true for both cannot exceed either count")

    a_part = count_part(
        f"Find the number of the {_N} possible truth assignments to p, q, r and s for which F and G have "
        "different truth values.",
        differ,
        3,
        [f"Evaluate F and G on all {_N} rows: they differ on {differ}."],
        [
            (agree, "found-complement-not-event"),
            (f_true, "counted-one-formula-only"),
            (g_true, "counted-one-formula-only"),
            (differ + 1, None),
            (differ - 1, None),
            (differ + 2, None),
        ],
        hi=_N,
    )
    b_part = _fraction_part(
        f"One truth assignment is chosen at random, all {_N} being equally likely. "
        "Hence find the probability that F and G have the same truth value.",
        Fraction(agree, _N),
        1,
        [f"P(same) = {agree}/{_N}."],
        [
            (Fraction(differ, _N), "found-complement-not-event"),
            (Fraction(agree, _N - 1), "counted-from-wrong-total"),
            (Fraction(agree, differ), "odds-for-probability"),
        ],
        follow_through=f"1 - part_a/{_N}",
    )
    c_part = count_part(
        f"For how many of the {_N} assignments are F and G both true?",
        both,
        2,
        [f"Both are true on {both} of the {_N} rows."],
        [
            (f_true, "counted-one-formula-only"),
            (g_true, "counted-one-formula-only"),
            (agree, "found-complement-not-event"),
            (both + 1, None),
            (both - 1, None),
            (both + 2, None),
        ],
        hi=_N,
    )
    f_only = sum(1 for env in rows if _eval(f, env) and not _eval(g, env))
    d_part = count_part(
        f"For how many of the {_N} assignments is F true and G false?",
        f_only,
        2,
        [f"F is true and G false on {f_only} of the {_N} rows."],
        [
            (differ, "counted-both-directions"),
            (differ - f_only, "counted-wrong-direction"),
            (f_true, "counted-one-formula-only"),
            (f_only + 1, None),
            (f_only - 1, None),
        ],
        hi=_N,
    )
    text = f"Let F be the formula {_render(f)} and G the formula {_render(g)}."
    neither = sum(1 for env in rows if not _eval(f, env) and not _eval(g, env))
    e_part = count_part(
        f"For how many of the {_N} assignments are F and G both false?",
        neither,
        2,
        [f"Both are false on {neither} of the {_N} rows."],
        [
            (differ, "counted-both-directions"),
            (_N - f_true, "counted-one-formula-only"),
            (_N - g_true, "counted-one-formula-only"),
            (neither + 1, None),
            (neither - 1, None),
        ],
        hi=_N,
    )
    atom = rng.choice(_ATOMS)
    value = rng.choice([True, False])
    word = "true" if value else "false"
    at_value = sum(1 for env in rows if env[atom] == value and _eval(f, env) != _eval(g, env))
    f_part = count_part(
        f"For how many of the {_N} assignments do F and G differ and {atom} is {word}?",
        at_value,
        2,
        [f"Of the {_N // 2} rows with {atom} {word}, F and G differ on {at_value}."],
        [
            (differ, "ignored-the-given-condition"),
            (differ - at_value, "used-opposite-condition-value"),
            (_N // 2 - at_value, "found-complement-not-event"),
            (at_value + 1, None),
            (at_value - 1, None),
        ],
        hi=_N // 2,
    )
    return dict(text=text, parts=[a_part, b_part, c_part, d_part, e_part, f_part])
