"""Given a described relation -- a finite set of ordered pairs, an equation, or
a text description of a graph -- classify it as a function or not a function
by applying the vertical line test.

Every draw offers exactly FOUR options: one fixed "Function: every x-value
has exactly one y-value" option, plus three "Not a function: x = {x} gives
two different y-values, {a} and {b}" claims, each a concrete, checkable
statement about the relation. Exactly one option is true on every draw:

- When the relation IS a function, the "Function" option is the key and all
  three "Not a function" claims are constructed to be FALSE for this
  relation (wrong x, or a right x with a wrong y-pair).
- When it is NOT a function, exactly one "Not a function" claim is
  constructed to be TRUE (a real witness x with two real y-values), and the
  other two are FALSE, and the "Function" option is false.

The hazard here (a circle or `x = y^2 + c` fails at MANY
x-values, so two naively-built "fails at x = k" claims could both be true)
is closed structurally: every relation exposes one `contains(x, y) -> bool`
predicate, and a claim `(x, a, b)`'s truth is *always* recomputed directly
from `contains` -- `contains(x, a) and contains(x, b) and a != b` -- never
assumed from how it was constructed. `_verify_claims` re-checks every claim,
including the ones built to be false, on every draw, and `check()`s that
exactly the right number are true. `GENUINE_TWO_OPTION_TOPICS` is never
touched -- this is a genuine 4-way mcq with 3 distractors every draw.
"""

from __future__ import annotations

from mathgen.core import check, generator, latex_answer, latex_prompt, phrase

_LETTERS = ["A", "B", "C", "D"]
_FUNC_TEXT = "Function: every x-value has exactly one y-value"
_Y_RANGE = tuple(range(-14, 15))


# ===========================================================================
# Generic option assembly -- shared by every representation
# ===========================================================================


def _scan_ys(contains, x):
    return [y for y in _Y_RANGE if contains(x, y)]


def _find_false_pair(rng, contains, x):
    """Two distinct y in `_Y_RANGE` such that the claim `x gives y=a,b` is
    FALSE for this relation (not both a and b actually satisfy `contains`).

    The caller (`_build_claims`) guarantees `x` has at most one real y under
    `contains` before calling this -- never a genuine two-value point (that's
    reserved for the true witness) -- so a single loop over `_Y_RANGE`
    excluding the one real y (if any) always finds a false pair immediately.
    `check()`s the same restriction at its own top too (defence in depth, not
    just relying on the caller getting it right)."""
    ys = _scan_ys(contains, x)
    check(len(ys) <= 1, "function-or-relation: a false-claim x must not be a genuine two-value point")
    pool = list(_Y_RANGE)
    rng.shuffle(pool)
    if ys:
        a = ys[0]
        for b in pool:
            if b != a:
                return (a, b) if a < b else (b, a)
    else:
        a, b = pool[0], pool[1]
        return (a, b) if a < b else (b, a)
    raise AssertionError("could not build a false claim")


def _build_claims(rng, contains, is_function, x_candidates, n=3):
    """Build exactly `n` claims: if `is_function` is False, one is a genuine
    witness (a `two_value` claim, both a, b really satisfy `contains(x, .)`);
    every other claim is constructed false. Never assumed -- `_verify_claims`
    re-checks every one directly against `contains`.

    Every claim is a dict `{"kind": ..., "x": x, "a": a, "b": b}` (`b` only
    for `two_value`):
    - `"two_value"`: "Not a function: x = {x} gives two different y-values,
      {a} and {b}" -- true iff `contains(x, a) and contains(x, b) and a != b`.
    - `"one_value"`: "Function: x = {x} gives only y = {a}" -- true iff `a`
      is the ONLY real y at `x` (`_scan_ys(contains, x) == [a]`). Used for a
      false claim's x that is itself a genuine two-real-y point (the vertical
      line test also fails there): the claim UNDERSTATES it as single-valued,
      which is checkably false without ever citing an x absent from the
      relation.

    Two passes: first locate a true `two_value` witness ANYWHERE in
    `x_candidates`, then spend the remaining slots on false claims at the
    other candidate x's, in THREE preference bands (an earlier version had only two bands and left genuine extra witness
    points unused, forcing an avoidable fallback to x's absent from the
    relation on shapes with only one single-valued point, e.g. a parabola's
    vertex):
    1. `single_valued` (exactly one real y -- e.g. a used pair's x, or a
       parabola's vertex): a `two_value` claim with one real number and one
       fudged number, per `_find_false_pair`.
    2. `extra_witness` (>= 2 real y's, genuinely in the relation, just not
       the one chosen as the true witness -- e.g. a parabola or circle's
       OTHER solution points): a `one_value` claim, always false by
       construction (see above) and always genuinely on the relation.
    3. `outside_domain` (zero real y, wholly outside the relation) -- last
       resort only; with band 2 available, every shape in this module (set,
       circle, and both parabola families) now has enough in-domain x's that
       this band is never actually reached, but it stays as a safety net."""
    xs = list(dict.fromkeys(x_candidates))
    rng.shuffle(xs)
    claims: list[dict] = []
    remaining_xs = list(xs)

    if not is_function:
        for x in xs:
            ys = _scan_ys(contains, x)
            if len(ys) >= 2:
                a, b = sorted(rng.sample(ys, 2))
                claims.append({"kind": "two_value", "x": x, "a": a, "b": b})
                remaining_xs.remove(x)
                break
        check(len(claims) == 1, "function-or-relation: no true witness found for a non-function relation")

    single_valued = []
    extra_witness = []
    outside_domain = []
    for x in remaining_xs:
        ys = _scan_ys(contains, x)
        if len(ys) >= 2:
            extra_witness.append(x)
        elif len(ys) == 1:
            single_valued.append(x)
        else:
            outside_domain.append(x)
    rng.shuffle(single_valued)
    rng.shuffle(extra_witness)
    rng.shuffle(outside_domain)

    for x in single_valued + extra_witness + outside_domain:
        if len(claims) >= n:
            break
        ys = _scan_ys(contains, x)
        if len(ys) >= 2:
            a = rng.choice(ys)
            claim = {"kind": "one_value", "x": x, "a": a}
        else:
            check(len(ys) <= 1, "function-or-relation: a two_value false-claim x must not be a genuine two-value point")
            a, b = _find_false_pair(rng, contains, x)
            claim = {"kind": "two_value", "x": x, "a": a, "b": b}
        if claim in claims:
            continue
        claims.append(claim)
    check(len(claims) == n, "function-or-relation: could not build enough distinct claims")
    return claims


def _verify_claims(contains, claims):
    truths = []
    for c in claims:
        if c["kind"] == "two_value":
            truths.append(contains(c["x"], c["a"]) and contains(c["x"], c["b"]) and c["a"] != c["b"])
        else:
            truths.append(_scan_ys(contains, c["x"]) == [c["a"]])
    return truths


def _claim_text(claim):
    if claim["kind"] == "two_value":
        return f"Not a function: x = {claim['x']} gives two different y-values, {claim['a']} and {claim['b']}"
    return f"Function: x = {claim['x']} gives only y = {claim['a']}"


def _assemble_options(rng, contains, is_function, x_candidates):
    claims = _build_claims(rng, contains, is_function, x_candidates, n=3)
    truths = _verify_claims(contains, claims)
    n_true = sum(truths)
    check(
        n_true == (0 if is_function else 1),
        f"function-or-relation: expected {0 if is_function else 1} true not-a-function claim(s), got {n_true}",
    )

    not_texts = [_claim_text(c) for c in claims]
    check(len(set(not_texts)) == 3, "function-or-relation: not-a-function options must all display differently")

    option_texts = [_FUNC_TEXT] + not_texts
    option_truth = [is_function] + truths
    check(sum(option_truth) == 1, "function-or-relation: exactly one option must be true")

    order = list(range(4))
    rng.shuffle(order)
    shuffled_texts = [option_texts[i] for i in order]
    shuffled_truth = [option_truth[i] for i in order]
    answer_idx = shuffled_truth.index(True)
    answer_letter = _LETTERS[answer_idx]
    distractor_letters = [l for l in _LETTERS if l != answer_letter]
    opts_str = "   ".join(f"{_LETTERS[i]}: {shuffled_texts[i]}" for i in range(4))
    return opts_str, answer_letter, distractor_letters, shuffled_texts[answer_idx]


def _prompt_latex(text: str) -> str:
    """`latex_prompt(text)`, with the literal set-notation braces `{`/`}`
    (present verbatim in `text` for the "ordered pairs" prompt shape) lifted
    out of math mode.

    `latex_prompt` escapes a literal `{`/`}` character to `\\{`/`\\}` IN
    PLACE, inside whichever `\\text{...}` span it fell into when the prose
    was split around a negative number -- e.g. `\\text{The relation
    \\{(}-3\\text{, 2), ...}`. That nested `{`/`}` inside the `\\text{}`
    argument breaks `tests/test_prompt_latex_mangled.py`'s one-level-nesting
    `\\text{[^{}]*}` assumption (the same one-level convention every other
    skill in the corpus honours), so the surviving prose after that group
    "leaks" through as an apparent hand-rolled mangled string.

    Fix: close the current `\\text{}` span right before the brace, emit the
    brace as bare math (`\\{`/`\\}`, valid in math mode and renders
    identically to the text-mode escape), then reopen a fresh `\\text{}`
    span. This produces the same rendered braces with clean one-level
    `\\text{}` nesting throughout, so it clears the detector.

    (`\\lbrace`/`\\rbrace` were tried first and rejected: KaTeX only defines
    them in MATH mode -- inside a `\\text{}` argument they are "Undefined
    control sequence", verified with `katex.renderToString`, worse than the
    defect. Math-mode `\\{`/`\\}` between two `\\text{}` groups is what
    `latex_prompt` itself does for every other lifted math token.)
    """
    latex = latex_prompt(text)
    return latex.replace(r"\{", r"}\{\text{").replace(r"\}", r"}\}\text{")


def _sign_part(c: int) -> str:
    if c > 0:
        return f" + {c}"
    if c < 0:
        return f" - {abs(c)}"
    return ""


def _shift_part(var: str, offset: int) -> str:
    """`(x - 3)`, `(x + 3)`, or plain `x` when `offset == 0` -- used for the
    exam-tier parabola descriptions so a zero vertex coordinate never prints
    the odd `(x - 0)`."""
    if offset == 0:
        return var
    if offset > 0:
        return f"({var} - {offset})"
    return f"({var} + {abs(offset)})"


# ===========================================================================
# Easy: a finite set of 4-6 ordered pairs
# ===========================================================================


def _pairs_is_function(pairs) -> bool:
    d: dict[int, set[int]] = {}
    for x, y in pairs:
        d.setdefault(x, set()).add(y)
    return all(len(ys) <= 1 for ys in d.values())


def _build_easy(rng):
    is_function = rng.random() < 0.30
    n_pairs = rng.randint(4, 6)
    x_pool = list(range(-7, 8))
    if is_function:
        xs = rng.sample(x_pool, n_pairs)
        ys = [rng.randint(-9, 9) for _ in xs]
        pairs = list(zip(xs, ys))
    else:
        rng.shuffle(x_pool)
        rep_x = x_pool[0]
        other_xs = x_pool[1 : n_pairs - 1]
        y1, y2 = rng.sample(range(-9, 10), 2)
        pairs = [(rep_x, y1), (rep_x, y2)] + [(x, rng.randint(-9, 9)) for x in other_xs]
        rng.shuffle(pairs)
    check(_pairs_is_function(pairs) == is_function, "easy: pairs structure disagrees with the drawn class")

    def contains(x, y):
        return (x, y) in pairs

    used_xs = list(dict.fromkeys(x for x, _y in pairs))
    extra_xs = [v for v in x_pool if v not in used_xs]
    x_candidates = used_xs + extra_xs

    pairs_text = "{" + ", ".join(f"({x}, {y})" for x, y in pairs) + "}"
    opts, answer_letter, distractor_letters, _ = _assemble_options(rng, contains, is_function, x_candidates)
    text = phrase(
        rng,
        [
            "A relation consists of the ordered pairs {pairs}. Is this relation a function? {opts}",
            "Consider the relation given by the set of ordered pairs {pairs}. Apply the vertical "
            "line test: is it a function? {opts}",
            "The relation {pairs} is given as a set of ordered pairs. Decide whether it is a function. {opts}",
        ],
        pairs=pairs_text,
        opts=opts,
    )
    working = [f"Ordered pairs: {pairs_text}.", f"Option {answer_letter} is the only true statement."]
    return text, answer_letter, distractor_letters, working


# ===========================================================================
# Medium: equations -- y = polynomial (function) vs x = y^2 + c, x = k (not)
# ===========================================================================


def _build_medium(rng):
    # p=0.35 here (not the 0.30 used by the other tiers): measured at 200
    # seeds, 0.30 landed this tier's function-key share at 24.5%, just under
    # the contract's 25-35% band; 0.35 clears it with margin (see the build
    # test `test_answer_class_distribution_within_bounds`, medium).
    is_function = rng.random() < 0.35
    x_pool = list(range(-6, 7))
    if is_function:
        if rng.random() < 0.5:
            a = rng.choice([1, -1, 2, -2])
            b = rng.randint(-3, 3)
            c = rng.randint(-5, 5)
            eq_text = f"y = {a}x^2{_sign_part(b)}x{_sign_part(c)}" if b != 0 else f"y = {a}x^2{_sign_part(c)}"

            def contains(x, y):
                return y == a * x * x + b * x + c
        else:
            a = rng.choice([1, -1, 2, -2, 3, -3])
            b = rng.randint(-6, 6)
            eq_text = f"y = {a}x{_sign_part(b)}"

            def contains(x, y):
                return y == a * x + b

        x_candidates = x_pool
    else:
        if rng.random() < 0.5:
            c = rng.randint(-4, 4)
            eq_text = f"x = y^2{_sign_part(c)}"

            def contains(x, y):
                return x == y * y + c

            # x = y^2 + c: domain x >= c; the vertex x = c (k=0) is the ONE
            # genuine single-real-y point (y = 0), a false-claim x the fix
            # can spend before falling back to outside-domain x's.
            x_candidates = [c] + [c + k * k for k in range(1, 5)] + [c - 1, c - 2, c - 3]
        else:
            # The old `x = k` vertical line has ZERO x's with exactly one real y (every y at x = k
            # is a TRUE claim; every other x has zero real y) -- no honest false claim exists
            # anywhere in its domain, no matter how the pool is ordered. Replaced with the
            # mirror-image parabola `x = -y^2 + c` (opens the other way from the branch above): its
            # vertex x = c is a genuine single-real-y point (y = 0), giving the false-claim pool
            # exactly one in-domain x to spend before falling back -- the same shape of coverage the
            # `x = y^2 + c` branch gets.
            c = rng.randint(-4, 4)
            eq_text = f"x = -y^2{_sign_part(c)}"

            def contains(x, y):
                return x == -(y * y) + c

            # x = -y^2 + c <=> y^2 = c - x: domain x <= c; witnesses at
            # x = c - k^2 (y = +-k); the vertex x = c (k=0) is the ONE
            # single-real-y point; x > c has no real solution.
            x_candidates = [c] + [c - k * k for k in range(1, 5)] + [c + 1, c + 2, c + 3]

    opts, answer_letter, distractor_letters, _ = _assemble_options(rng, contains, is_function, x_candidates)
    text = phrase(
        rng,
        [
            "A relation is given by the equation {eq}. Is this relation a function? {opts}",
            "Consider the relation {eq}. Apply the vertical line test: is it a function? {opts}",
            "A relation between x and y is defined by the equation {eq}. Decide whether it is a function. {opts}",
        ],
        eq=eq_text,
        opts=opts,
    )
    working = [f"Relation: {eq_text}.", f"Option {answer_letter} is the only true statement."]
    return text, answer_letter, distractor_letters, working


# ===========================================================================
# Hard: near-miss traps
# ===========================================================================


def _build_hard(rng):
    is_function = rng.random() < 0.30
    x_pool = list(range(-7, 8))
    if is_function:
        if rng.random() < 0.5:
            xs = rng.sample(x_pool, 5)
            ys = [rng.randint(-4, 4) for _ in xs]
            ys[1] = ys[0]  # forced repeated y at a DISTINCT x -- still a function
            pairs = list(zip(xs, ys))
            note = "repeated y-value at distinct x-values"
        else:
            xs = rng.sample(x_pool, 4)
            ys = [rng.randint(-9, 9) for _ in xs]
            pairs = list(zip(xs, ys))
            pairs = pairs + [pairs[0]]  # the SAME pair listed twice
            rng.shuffle(pairs)
            note = "a duplicate identical pair"
        check(_pairs_is_function(pairs) == is_function, f"hard ({note}): structure disagrees with class")

        def contains(x, y):
            return (x, y) in pairs

        used_xs = list(dict.fromkeys(x for x, _y in pairs))
        x_candidates = used_xs + [v for v in x_pool if v not in used_xs]
        pairs_text = "{" + ", ".join(f"({x}, {y})" for x, y in pairs) + "}"
        opts, answer_letter, distractor_letters, _ = _assemble_options(rng, contains, is_function, x_candidates)
        text = phrase(
            rng,
            [
                "A relation consists of the ordered pairs {pairs}. Is this relation a function? {opts}",
                "Consider the relation given by the set of ordered pairs {pairs}. Apply the "
                "vertical line test: is it a function? {opts}",
                "The relation {pairs} is given as a set of ordered pairs. Decide whether it is a function. {opts}",
            ],
            pairs=pairs_text,
            opts=opts,
        )
        working = [f"Ordered pairs: {pairs_text} ({note}).", f"Option {answer_letter} is the only true statement."]
    else:
        if rng.random() < 0.5:
            c = rng.randint(-4, 4)
            eq_text = f"y^2 = x{_sign_part(c)}"

            def contains(x, y):
                return y * y == x + c

            # y^2 = x + c <=> x = y^2 - c: witnesses at x = k^2 - c (y = +-k); x < -c has no real
            # solution at all (y^2 would be negative).
            x_candidates = [-c] + [k * k - c for k in range(1, 5)] + [-c - 1, -c - 2, -c - 3]
        else:
            r = rng.choice([5, 10, 13])

            def contains(x, y):
                return x * x + y * y == r * r

            eq_text = f"x^2 + y^2 = {r * r}"
            interior = [v for v in range(-r + 1, r) if v != 0]
            x_candidates = [0, r, -r] + interior
        opts, answer_letter, distractor_letters, _ = _assemble_options(rng, contains, is_function, x_candidates)
        text = phrase(
            rng,
            [
                "A relation is given by the equation {eq}. Is this relation a function? {opts}",
                "Consider the relation {eq}. Apply the vertical line test: is it a function? {opts}",
                "A relation is defined by the equation {eq}. Decide whether it is a function. {opts}",
            ],
            eq=eq_text,
            opts=opts,
        )
        working = [f"Relation: {eq_text}.", f"Option {answer_letter} is the only true statement."]
    return text, answer_letter, distractor_letters, working


# ===========================================================================
# Exam: described graphs / domain-restricted relations
# ===========================================================================


def _build_exam(rng):
    is_function = rng.random() < 0.30
    r = rng.choice([5, 10, 13])
    interior = [v for v in range(-r + 1, r) if v != 0]
    if is_function:
        variant = rng.choice(["upper_semicircle", "restricted_ge0", "parabola_up"])
        if variant == "upper_semicircle":

            def contains(x, y):
                return x * x + y * y == r * r and y >= 0

            desc = f"the upper semicircle y = sqrt({r * r} - x^2)"
            x_candidates = [0, r, -r] + interior
        elif variant == "restricted_ge0":

            def contains(x, y):
                return x * x + y * y == r * r and y >= 0

            desc = f"the part of the circle x^2 + y^2 = {r * r} with y >= 0"
            x_candidates = [0, r, -r] + interior
        else:
            h, k = rng.randint(-4, 4), rng.randint(-4, 4)

            def contains(x, y):
                return y == (x - h) ** 2 + k

            desc = f"the parabola y = {_shift_part('x', h)}^2{_sign_part(k)}, which opens upward with vertex ({h}, {k})"
            x_candidates = list(range(h - 5, h + 6))
    else:
        variant = rng.choice(["whole_circle", "parabola_right", "parabola_left"])
        if variant == "whole_circle":

            def contains(x, y):
                return x * x + y * y == r * r

            desc = f"the whole circle x^2 + y^2 = {r * r}, centred at the origin with radius {r}"
            x_candidates = [0, r, -r] + interior
        elif variant == "parabola_right":
            h, k = rng.randint(-4, 4), rng.randint(-4, 4)

            def contains(x, y):
                return x == (y - k) ** 2 + h

            desc = f"the parabola x = {_shift_part('y', k)}^2{_sign_part(h)}, which opens to the right with vertex ({h}, {k})"
            x_candidates = list(range(h - 5, h + 12))
        else:
            # The old `vertical_line` variant has the SAME defect as medium's old `x = k` branch --
            # zero x's with exactly one real y anywhere in its domain. Replaced with a second
            # single-vertex shape, distinct in wording and vertex-sign from `parabola_right`: a
            # leftward-opening parabola `x = -(y - k)^2 + h`. Its vertex x = h is a genuine
            # single-real-y point (y = k); every other x in its domain (x <= h) is a genuine
            # two-value witness, and every x > h is outside the domain (zero real y) -- the same
            # coverage shape as `parabola_right`, mirrored.
            h, k = rng.randint(-4, 4), rng.randint(-4, 4)

            def contains(x, y):
                return x == -((y - k) ** 2) + h

            desc = f"the parabola x = -{_shift_part('y', k)}^2{_sign_part(h)}, which opens to the left with vertex ({h}, {k})"
            x_candidates = list(range(h - 11, h + 6))

    opts, answer_letter, distractor_letters, _ = _assemble_options(rng, contains, is_function, x_candidates)
    text = phrase(
        rng,
        [
            "A relation is described as {desc}. Is this relation a function? {opts}",
            "Consider {desc}. Apply the vertical line test to this described graph: is it a function? {opts}",
            "A graph is described as {desc}. Decide whether the relation it represents is a function. {opts}",
        ],
        desc=desc,
        opts=opts,
    )
    working = [f"Description: {desc}.", f"Option {answer_letter} is the only true statement."]
    return text, answer_letter, distractor_letters, working


@generator(
    "p19e_function_or_relation",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19e_function_or_relation(rng, tier="easy"):
    """Given a described relation -- a set of ordered pairs, an equation, or
    a description of its graph -- classify it as a function or not a
    function by applying the vertical line test."""
    if tier == "easy":
        text, answer_letter, distractor_letters, working = _build_easy(rng)
    elif tier == "medium":
        text, answer_letter, distractor_letters, working = _build_medium(rng)
    elif tier == "hard":
        text, answer_letter, distractor_letters, working = _build_hard(rng)
    elif tier == "exam":
        text, answer_letter, distractor_letters, working = _build_exam(rng)
    else:
        raise ValueError(f"unknown tier: {tier!r}")

    return dict(
        text=text,
        latex=_prompt_latex(text),
        answer=answer_letter,
        answer_text=answer_letter,
        answer_latex=latex_answer(answer_letter, plain=True),
        working=working,
        distractors=distractor_letters,
    )
