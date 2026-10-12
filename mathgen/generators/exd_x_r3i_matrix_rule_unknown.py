"""Exam-tier source for `senior-understand-the-matrix-definition-and-notation`: a matrix given by a rule with an
unknown constant, fixed by one known entry and then used to locate and read entries.

The stem gives a rule for a_{i,j} with one unknown constant k and the size of a square matrix, and states one entry.
The parts use the notation in turn:

(a) find k from the stated entry                              number, 2 marks
(b) hence find another named entry                            number, 1 mark,  rule(r, s, part_a)
(c) the column in a named row where an entry takes a stated value   number, 2 marks, the solution of rule(t, j, part_a) = target for j
(d) the entry whose row number is the column found in (c) and whose column number is the named row's number
                                                              number, 2 marks, rule(part_c, t, part_a)

## Construction

Four rule shapes put k in front of i, in front of i*j, in front of j, or in front of -i, with a small whole-number
constant elsewhere. k and the stated entry's position are drawn so k is a whole number (the coefficient of k at that
position is never zero), and the rule is linear in j in every named row, so the stated value occurs in exactly one
column. Every answer is a whole number.
"""

from __future__ import annotations

import re
from fractions import Fraction

from ..core import attempts, check, generator, join_terms, phrase
from ._exd_x_r3_distinct import distinct_answers
from .exd_x_r3i_support import number_part

#: (text with {c}, coefficient of k, rest) as functions of (i, j, c)
_RULES = (
    ("k i + {c} j", lambda i, j, c: i, lambda i, j, c: c * j),
    ("k i j + {c}", lambda i, j, c: i * j, lambda i, j, c: c),
    ("{c} i^2 + k j", lambda i, j, c: j, lambda i, j, c: c * i * i),
    ("{c} i j - k i", lambda i, j, c: -i, lambda i, j, c: c * i * j),
)


def _rule_text(template: str, c: int) -> str:
    """The rule with c in its place, written the way a student would write it (no '+ -', no '1 i')."""
    text = template.format(c=c)
    text = text.replace("+ -", "- ").replace("- -", "+ ").replace("1 i", "i").replace("1 j", "j")
    if text.startswith("-1 "):
        text = "-" + text[3:]
    return re.sub(r"(?<=[\dkij]) (?=[kij])", "", text)


_ENTRY_RE = re.compile(r"a_\{[^}]*\}(?: = [^,.]+(?=[,.]))?")


def _latex_of(text: str) -> str:
    """Real LaTeX for a sentence that mentions entries a_{i,j}: prose in \\text, each entry (and any rule after it) as maths."""

    def math(m):
        body = re.sub(r"(?<=[kij])(?=[kij])", r"\\,", m.group(0))
        body = re.sub(r"\^(\d)", r"^{\1}", body).replace(" = ", "=")
        return "}" + body + r"\text{"

    return r"\text{" + _ENTRY_RE.sub(math, text) + "}"


@generator("exd_x_r3i_matrix_rule_unknown", level=17, presets={"exam": dict(tier="exam")})
@distinct_answers
def exd_x_r3i_matrix_rule_unknown(rng, tier: str = "exam") -> dict:
    """Find k from one entry of a rule-defined matrix, read another entry, locate a value in a row, read again."""
    check(tier == "exam", f"unknown tier {tier!r}")
    template, coef, rest = rng.choice(_RULES)
    n = rng.choice([3, 4, 4])

    for _ in attempts(2000, "a rule with a whole-number k and a unique column"):
        c = rng.choice([-4, -3, -2, 2, 3, 4])
        k = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        p, q = rng.randint(1, n), rng.randint(1, n)
        r, s = rng.randint(1, n), rng.randint(1, n)
        t, j0 = rng.randint(1, n), rng.randint(1, n)

        def entry(i, j):
            return coef(i, j, c) * k + rest(i, j, c)

        if p == q or (r, s) in ((p, q), (q, p)) or coef(p, q, c) == 0 or coef(q, p, c) == 0 or j0 == t or t == p:
            continue
        # The wanted column is unique: the row's entries are all different.
        if len({entry(t, j) for j in range(1, n + 1)}) < n or entry(r, s) == 0 or entry(j0, t) == 0:
            continue
        given = entry(p, q)
        wrong_k = [
            Fraction(given - rest(q, p, c), coef(q, p, c)),
            Fraction(given + rest(p, q, c), coef(p, q, c)),
            Fraction(given, coef(p, q, c)),
            Fraction(given - rest(p, q, c)),
        ]
        if len({w for w in wrong_k if w.denominator == 1 and w != k}) >= 2:
            break

    target = entry(t, j0)
    value = entry(r, s)
    final = entry(j0, t)
    # Follow-through expressions, built by hand from the rule's pieces: entry(i, j) = coef(i, j) * k + rest(i, j).
    check(Fraction(given - rest(p, q, c), coef(p, q, c)) == k, "k does not follow from the stated entry")
    ft_entry = f"({coef(r, s, c)})*part_a + ({rest(r, s, c)})"
    # Along row t the rule is linear in j: (step_k k + step_c) j + (base_k k + base_c).
    step_k, step_c = coef(t, 2, c) - coef(t, 1, c), rest(t, 2, c) - rest(t, 1, c)
    base_k, base_c = coef(t, 1, c) - step_k, rest(t, 1, c) - step_c
    check(
        all(
            coef(t, j + 1, c) - coef(t, j, c) == step_k and rest(t, j + 1, c) - rest(t, j, c) == step_c for j in (1, 2)
        ),
        "the rule is not linear in j along the row",
    )
    check(step_k * k + step_c != 0, "the column is not unique")
    ft_column = f"({target} - ({base_c}) - ({base_k})*part_a)/(({step_c}) + ({step_k})*part_a)"
    check(Fraction(target - base_c - base_k * k, step_c + step_k * k) == j0, "the column does not follow")

    def quadratic_in_i(fn):
        """Coefficients (a0, a1, a2) of fn(i, t) = a0 + a1 i + a2 i^2, read off its values at i = 0, 1, 2."""
        v0, v1, v2 = fn(0, t, c), fn(1, t, c), fn(2, t, c)
        a2 = Fraction(v2 - 2 * v1 + v0, 2)
        return v0, v1 - v0 - a2, a2

    ca, rb = quadratic_in_i(coef), quadratic_in_i(rest)
    ft_final = (
        f"(({ca[0]}) + ({ca[1]})*part_c + ({ca[2]})*part_c**2)*part_a"
        f" + (({rb[0]}) + ({rb[1]})*part_c + ({rb[2]})*part_c**2)"
    )
    check(
        (ca[0] + ca[1] * j0 + ca[2] * j0**2) * k + (rb[0] + rb[1] * j0 + rb[2] * j0**2) == final,
        "the last entry does not follow",
    )

    rule = _rule_text(template, c)
    stem = phrase(
        rng,
        (
            "A {n} by {n} matrix A has entries a_{{i,j}} = {rule}, where k is a constant, for i = 1, ..., {n} and "
            "j = 1, ..., {n}. The entry in row {p} and column {q} is {given}.",
            "The entries of a square matrix A with {n} rows and {n} columns follow the rule a_{{i,j}} = {rule}, for a "
            "constant k. It is known that a_{{{p},{q}}} = {given}.",
        ),
        n=n,
        rule=rule,
        p=p,
        q=q,
        given=given,
    )

    def e_text(i, j):
        return f"a_{{{i},{j}}}"

    part_a = number_part(
        "Find the value of k.",
        Fraction(k),
        0,
        2,
        [
            f"{e_text(p, q)} = {given} gives {join_terms([(coef(p, q, c), 'k'), (rest(p, q, c), '')])} = {given}, so k = {k}."
        ],
        [
            ("row-and-column-swapped", wrong_k[0]),
            ("sign-error-moving-term", wrong_k[1]),
            ("ignored-constant-term", wrong_k[2]),
            ("forgot-to-divide-by-coefficient", wrong_k[3]),
        ],
    )
    part_b = number_part(
        f"Hence find the value of {e_text(r, s)}.",
        Fraction(value),
        0,
        1,
        [f"{e_text(r, s)} = {value}."],
        [
            ("row-and-column-swapped", Fraction(entry(s, r))),
            ("read-wrong-row-or-column", Fraction(entry(r % n + 1, s))),
            ("read-wrong-row-or-column", Fraction(entry(r, s % n + 1))),
            ("left-parameter-out", Fraction(rest(r, s, c))),
        ],
        follow_through=ft_entry,
    )
    # Mistakes in solving for the column: the same three slips as for k, applied to a_{t,j} = target.
    cj = entry(t, 2) - entry(t, 1)  # the row's step per column (non-zero, as the entries differ)
    base = entry(t, 1) - cj  # the row's entry extrapolated to column 0
    part_c = number_part(
        f"Exactly one entry in row {t} equals {target}. In which column is it?",
        Fraction(j0),
        0,
        2,
        [
            f"Row {t}: "
            + ", ".join(str(entry(t, j)) for j in range(1, n + 1))
            + f". The value {target} is in column {j0}."
        ],
        [
            ("sign-error-moving-term", Fraction(target + base, cj) if cj else None),
            ("ignored-constant-term", Fraction(target, cj) if cj else None),
            ("forgot-to-divide-by-coefficient", Fraction(target - base)),
            ("row-and-column-swapped", Fraction(t)),
            ("columns-counted-from-the-right", Fraction(n + 1 - j0)),
        ],
        follow_through=ft_column,
        sensible_range=(0, n + 1),
    )
    part_d = number_part(
        f"Using your answer to (c) as a row number, find the entry in that row and column {t} of A.",
        Fraction(final),
        0,
        2,
        [f"Row {j0}, column {t}: {e_text(j0, t)} = {final}."],
        [
            ("row-and-column-swapped", Fraction(entry(t, j0))),
            ("read-wrong-row-or-column", Fraction(entry(j0 % n + 1, t))),
            ("read-wrong-row-or-column", Fraction(entry(j0, t % n + 1))),
            ("left-parameter-out", Fraction(rest(j0, t, c))),
        ],
        follow_through=ft_final,
    )
    parts = [part_a, part_b, part_c, part_d]
    for part in parts:
        if "a_{" in part["prompt"]:
            part["prompt_latex"] = _latex_of(part["prompt"])
    return dict(text=stem, latex=_latex_of(stem), parts=parts)
