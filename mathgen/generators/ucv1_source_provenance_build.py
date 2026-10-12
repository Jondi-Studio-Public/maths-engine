"""Generates `source_provenance` items: separate what a student's notes reproduce from a supplied source, what
they compute, and what they newly deduce, and cite the supplied source for what is reproduced.

Each question supplies one or two short statements of a result (S1, S2: standard formulas, stated in the
question, never real citations) and a student's numbered notes on a bounded problem. A line is one of

- reproduced: it restates a supplied source (cite it by its label),
- computed: it evaluates an expression with numbers (a calculation, nothing new is claimed),
- a new deduction: it follows from the sources but is not stated in them (own reasoning, to be marked as new),
- unsourced: it states a result that no supplied source gives (it cannot be cited to S1 or S2).

Every figure and every formula is recomputed here (exact fractions, or brute-force sums), so the lines the
learner is asked to classify are true. The answer is marked against a rubric, so `answer` is a marking dict
(see `_crit_kit`), not a value.

Ladder:

- easy: three lines of one problem; classify each line and cite the source for the reproduced one.
- medium: four lines; classify, cite, and give the value each calculation produces.
- hard: two supplied sources and notes that attribute each line (S1, S2, a calculation or own reasoning);
  one or two attributions are wrong; find them and correct them.
- exam: as hard, and one line states a true result that neither source supplies yet is attributed to a source;
  say that it cannot be cited, and which later lines depend on it.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator
from ._crit_kit import attempts, Fm, Prompt, SHORT_TAILS, finish, frac_plain, frac_tex, short_marking

_PRESETS = {d: {"variant": d} for d in ("easy", "medium", "hard", "exam")}


def fr(x: Fraction | int) -> Fm:
    x = Fraction(x)
    return Fm(frac_tex(x.numerator, x.denominator), frac_plain(x.numerator, x.denominator))


def _plain(parts: list) -> str:
    return "".join(p.plain if isinstance(p, Fm) else p for p in parts)


def _emit(p: Prompt, parts: list) -> None:
    for part in parts:
        if isinstance(part, Fm):
            p.m(part)
        else:
            p.t(part)


def _line(kind: str, src: str | None, parts: list, note: str = "") -> dict:
    return dict(kind=kind, src=src, parts=parts, note=note)


KIND_WORD = {
    "R": "reproduced from a supplied source",
    "C": "computed (a calculation, nothing new is claimed)",
    "D": "a new deduction (it follows from the source but the source does not say it)",
    "X": "unsourced (no supplied source states it)",
}


# ---------------------------------------------------------------------------
# Problems. Each returns dict(sources [(label, parts)], problem [parts], lines [...]).
# ---------------------------------------------------------------------------
def _sum_series(a: int, r: Fraction, m: int) -> Fraction:
    return a * r**m / (1 - r)


GEOM = (
    "If ",
    Fm(r"|r|<1", "|r| < 1"),
    ", then ",
    Fm(r"a+ar+ar^{2}+\cdots=\frac{a}{1-r}", "a + a r + a r^2 + ... = a/(1 - r)"),
    ".",
)
GEOM_FINITE = (
    "The sum of the first n terms is ",
    Fm(r"a+ar+\cdots+ar^{n-1}=\frac{a(1-r^{n})}{1-r}", "a + a r + ... + a r^(n-1) = a(1 - r^n)/(1 - r)"),
    " for ",
    Fm(r"r\ne 1", "r != 1"),
    ".",
)
ARITH = ("For every positive integer n, ", Fm(r"1+2+\cdots+n=\frac{n(n+1)}{2}", "1 + 2 + ... + n = n(n+1)/2"), ".")
SQUARES = (
    "For every positive integer n, ",
    Fm(r"1^{2}+2^{2}+\cdots+n^{2}=\frac{n(n+1)(2n+1)}{6}", "1^2 + 2^2 + ... + n^2 = n(n+1)(2n+1)/6"),
    ".",
)
COMPOUND = (
    "A balance P that earns interest at rate r per period, added to the balance each period, is worth ",
    Fm(r"P(1+r)^{n}", "P(1 + r)^n"),
    " after n periods.",
)
SIMPLE = (
    "A balance P that earns interest at rate r per period on the original amount only is worth ",
    Fm(r"P(1+rn)", "P(1 + r n)"),
    " after n periods.",
)
QUAD = (
    "The roots of ",
    Fm(r"x^{2}+bx+c=0", "x^2 + b x + c = 0"),
    " are ",
    Fm(r"x=\frac{-b\pm\sqrt{b^{2}-4c}}{2}", "x = (-b ± sqrt(b^2 - 4c))/2"),
    ".",
)


def _geom_p(rng) -> dict:
    a = rng.choice((2, 3, 4, 5, 6, 8, 9))
    r = rng.choice(
        (
            Fraction(1, 2),
            Fraction(1, 3),
            Fraction(2, 3),
            Fraction(1, 4),
            Fraction(3, 4),
            Fraction(1, 5),
            Fraction(-1, 2),
            Fraction(-1, 3),
        )
    )
    m = rng.choice((2, 3))
    v, w = _sum_series(a, r, 0), _sum_series(a, r, m)
    approx = sum(a * r**k for k in range(m, m + 120))
    check(abs(approx - w) < Fraction(1, 10**9), "series tail")
    check(v == Fraction(a) / (1 - r), "series sum")
    lines = [
        _line(
            "R",
            "S1",
            [
                "The series ",
                Fm(r"a+ar+ar^{2}+\cdots", "a + a r + a r^2 + ..."),
                " has sum ",
                Fm(r"\frac{a}{1-r}", "a/(1 - r)"),
                " when ",
                Fm(r"|r|<1", "|r| < 1"),
                ".",
            ],
        ),
        _line(
            "C",
            None,
            [
                f"With a = {a} and r = ",
                fr(r),
                ", the whole series has sum ",
                Fm(
                    rf"\frac{{{a}}}{{1-{frac_tex(r.numerator, r.denominator) if r >= 0 else '(' + frac_tex(r.numerator, r.denominator) + ')'}}}="
                    + frac_tex(v.numerator, v.denominator),
                    f"{a}/(1 - {frac_plain(r.numerator, r.denominator) if r >= 0 else '(' + frac_plain(r.numerator, r.denominator) + ')'}) = "
                    + frac_plain(v.numerator, v.denominator),
                ),
                ".",
            ],
        ),
        _line(
            "D",
            None,
            [
                f"Dropping the first {m} terms leaves a series with first term ",
                Fm(rf"ar^{{{m}}}", f"a r^{m}"),
                " and the same ratio ",
                Fm("r"),
                ".",
            ],
        ),
        _line(
            "C",
            None,
            [
                "Its sum is ",
                Fm(
                    rf"\frac{{ar^{{{m}}}}}{{1-r}}=" + frac_tex(w.numerator, w.denominator),
                    f"a r^{m}/(1 - r) = " + frac_plain(w.numerator, w.denominator),
                ),
                ".",
            ],
        ),
    ]
    prob = [
        f"Find the sum of the series that starts at its term number {m + 1} (that is, without the first {m} terms) when a = {a} and r = ",
        fr(r),
        ".",
    ]
    return dict(sources=[("S1", list(GEOM))], problem=prob, lines=lines, answer=w)


def _arith_p(rng) -> dict:
    a = rng.randint(3, 12)
    b = a + rng.randint(5, 25)

    def s(n: int) -> int:
        return n * (n + 1) // 2

    check(sum(range(a, b + 1)) == s(b) - s(a - 1), "range sum")
    lines = [
        _line("R", "S1", ["The sum of the first n positive integers is ", Fm(r"\frac{n(n+1)}{2}", "n(n+1)/2"), "."]),
        _line(
            "C",
            None,
            [
                f"For n = {b}: ",
                Fm(rf"\frac{{{b}\cdot {b + 1}}}{{2}}={s(b)}", f"{b} x {b + 1}/2 = {s(b)}"),
                f". For n = {a - 1}: ",
                Fm(rf"\frac{{{a - 1}\cdot {a}}}{{2}}={s(a - 1)}", f"{a - 1} x {a}/2 = {s(a - 1)}"),
                ".",
            ],
        ),
        _line("D", None, [f"The sum from {a} to {b} is the sum to {b} minus the sum to {a - 1}."]),
        _line("C", None, [Fm(f"{s(b)}-{s(a - 1)}={s(b) - s(a - 1)}", f"{s(b)} - {s(a - 1)} = {s(b) - s(a - 1)}"), "."]),
    ]
    return dict(
        sources=[("S1", list(ARITH))],
        problem=[f"Find the sum of the integers from {a} to {b}."],
        lines=lines,
        answer=Fraction(s(b) - s(a - 1)),
    )


def _squares_p(rng) -> dict:
    a = rng.randint(2, 8)
    b = a + rng.randint(4, 14)

    def q(n: int) -> int:
        return n * (n + 1) * (2 * n + 1) // 6

    check(sum(k * k for k in range(a, b + 1)) == q(b) - q(a - 1), "squares range")
    lines = [
        _line(
            "R",
            "S1",
            [
                "The sum of the squares of the first n positive integers is ",
                Fm(r"\frac{n(n+1)(2n+1)}{6}", "n(n+1)(2n+1)/6"),
                ".",
            ],
        ),
        _line(
            "C",
            None,
            [f"For n = {b}: ", Fm(f"{q(b)}", f"{q(b)}"), f". For n = {a - 1}: ", Fm(f"{q(a - 1)}", f"{q(a - 1)}"), "."],
        ),
        _line("D", None, [f"The sum of squares from {a} to {b} is the sum to {b} minus the sum to {a - 1}."]),
        _line("C", None, [Fm(f"{q(b)}-{q(a - 1)}={q(b) - q(a - 1)}", f"{q(b)} - {q(a - 1)} = {q(b) - q(a - 1)}"), "."]),
    ]
    return dict(
        sources=[("S1", list(SQUARES))],
        problem=[f"Find ", Fm(f"{a}^{{2}}+\\cdots+{b}^{{2}}", f"{a}^2 + ... + {b}^2"), "."],
        lines=lines,
        answer=Fraction(q(b) - q(a - 1)),
    )


def _compound_p(rng) -> dict:
    P = rng.choice((1000, 2000, 5000, 4000))
    pct = rng.choice((10, 20))
    n = rng.choice((2, 3))
    r = Fraction(pct, 100)
    val = P * (1 + r) ** n
    check(val.denominator == 1, "whole balance")
    interest = val - P
    lines = [
        _line(
            "R",
            "S1",
            [
                "A balance P at rate r per period, compounded, is worth ",
                Fm(r"P(1+r)^{n}", "P(1 + r)^n"),
                " after n periods.",
            ],
        ),
        _line(
            "C",
            None,
            [Fm(rf"{P}(1+{pct / 100:g})^{{{n}}}={int(val)}", f"{P}(1 + {pct / 100:g})^{n} = {int(val)}"), "."],
        ),
        _line("D", None, ["The interest earned is the final balance minus the starting balance."]),
        _line("C", None, [Fm(f"{int(val)}-{P}={int(interest)}", f"{int(val)} - {P} = {int(interest)}"), "."]),
    ]
    return dict(
        sources=[("S1", list(COMPOUND))],
        problem=[
            f"A balance of {P} dollars earns {pct}% per period, compounded. How much interest has it earned after {n} periods?"
        ],
        lines=lines,
        answer=interest,
    )


def _quad_p(rng) -> dict:
    for _ in attempts(100, "distinct roots"):
        r1, r2 = rng.randint(-6, 7), rng.randint(-6, 7)
        if r1 != r2 and r1 + r2 != 0:
            break
    r1, r2 = sorted((r1, r2))
    b, c = -(r1 + r2), r1 * r2
    disc = b * b - 4 * c
    root = int(disc**0.5)
    check(root * root == disc and (-b + root) // 2 == r2 and (-b - root) // 2 == r1, "roots")
    b_tex = "" if b == 0 else ("+x" if b == 1 else "-x" if b == -1 else f"{b:+d}x")
    b_txt = "" if b == 0 else f" {'+' if b > 0 else '-'} {'' if abs(b) == 1 else f'{abs(b)} '}x"
    eq_tex = f"x^{{2}}{b_tex}{c:+d}=0"
    eq_txt = f"x^2{b_txt} {'+' if c >= 0 else '-'} {abs(c)} = 0"
    lines = [
        _line(
            "R",
            "S1",
            [
                "The roots of ",
                Fm(r"x^{2}+bx+c=0", "x^2 + b x + c = 0"),
                " are ",
                Fm(r"\frac{-b\pm\sqrt{b^{2}-4c}}{2}", "(-b ± sqrt(b^2 - 4c))/2"),
                ".",
            ],
        ),
        _line(
            "C",
            None,
            [
                f"Here b = {b} and c = {c}, so ",
                Fm(rf"b^{{2}}-4c={disc}", f"b^2 - 4c = {disc}"),
                f" and the roots are {r1} and {r2}.",
            ],
        ),
        _line("D", None, ["Adding the two roots, the plus-or-minus parts cancel, so the sum of the roots is -b."]),
        _line("C", None, [Fm(f"{r1}+{r2}={r1 + r2}", f"{r1} + {r2} = {r1 + r2}"), f", which equals -b = {-b}."]),
    ]
    return dict(
        sources=[("S1", list(QUAD))],
        problem=["Find the roots of ", Fm(eq_tex, eq_txt), " and the sum of the roots."],
        lines=lines,
        answer=Fraction(r1 + r2),
    )


SINGLE = [_geom_p, _arith_p, _squares_p, _compound_p, _quad_p]


# ---------------------------------------------------------------------------
# Two-source problems (hard) and exam problems with an unsourced line.
# ---------------------------------------------------------------------------
def _h_split(rng) -> dict:
    n = rng.randint(5, 14)
    v = sum(k * (k + 1) for k in range(1, n + 1))
    s1, s2 = n * (n + 1) // 2, n * (n + 1) * (2 * n + 1) // 6
    check(v == s1 + s2, "split")
    lines = [
        _line("R", "S1", ["The sum of the first n integers is ", Fm(r"\frac{n(n+1)}{2}", "n(n+1)/2"), "."]),
        _line("R", "S2", ["The sum of the first n squares is ", Fm(r"\frac{n(n+1)(2n+1)}{6}", "n(n+1)(2n+1)/6"), "."]),
        _line(
            "D",
            None,
            [
                "Since ",
                Fm(r"k(k+1)=k^{2}+k", "k(k + 1) = k^2 + k"),
                ", the sum of ",
                Fm(r"k(k+1)", "k(k + 1)"),
                " is the sum of the squares plus the sum of the integers.",
            ],
        ),
        _line("C", None, [f"For n = {n}: {s2} + {s1} = {v}."]),
    ]
    return dict(
        sources=[("S1", list(ARITH)), ("S2", list(SQUARES))],
        problem=["Find ", Fm(rf"1\cdot 2+2\cdot 3+\cdots+{n}({n + 1})", f"1 x 2 + 2 x 3 + ... + {n} x {n + 1}"), "."],
        lines=lines,
        answer=Fraction(v),
    )


def _h_tail(rng) -> dict:
    a = rng.choice((2, 3, 4, 5, 6))
    r = rng.choice((Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(1, 4), Fraction(3, 4)))
    n = rng.choice((4, 5, 6))
    inf = a / (1 - r)
    fin = a * (1 - r**n) / (1 - r)
    diff = inf - fin
    check(diff == a * r**n / (1 - r), "tail")
    check(abs(sum(a * r**k for k in range(n, n + 150)) - diff) < Fraction(1, 10**9), "tail by terms")
    lines = [
        _line(
            "R",
            "S1",
            ["The infinite series has sum ", Fm(r"\frac{a}{1-r}", "a/(1 - r)"), " when ", Fm(r"|r|<1", "|r| < 1"), "."],
        ),
        _line(
            "R", "S2", ["The sum of the first n terms is ", Fm(r"\frac{a(1-r^{n})}{1-r}", "a(1 - r^n)/(1 - r)"), "."]
        ),
        _line(
            "D",
            None,
            [
                "The amount by which the first n terms fall short of the whole series is the infinite sum minus the finite sum."
            ],
        ),
        _line(
            "C", None, [f"With a = {a}, r = ", fr(r), f" and n = {n}: ", fr(inf), " - ", fr(fin), " = ", fr(diff), "."]
        ),
    ]
    return dict(
        sources=[("S1", list(GEOM)), ("S2", list(GEOM_FINITE))],
        problem=[
            f"A series has first term {a} and ratio ",
            fr(r),
            f". By how much does the sum of its first {n} terms fall short of the sum of the whole series?",
        ],
        lines=lines,
        answer=diff,
    )


def _h_interest(rng) -> dict:
    P = rng.choice((1000, 2000, 5000))
    pct = rng.choice((10, 20))
    n = rng.choice((2, 3))
    r = Fraction(pct, 100)
    comp = P * (1 + r) ** n
    simp = P * (1 + r * n)
    diff = comp - simp
    check(comp.denominator == 1 and simp.denominator == 1 and diff > 0, "interest")
    lines = [
        _line("R", "S1", ["With compound interest the balance is ", Fm(r"P(1+r)^{n}", "P(1 + r)^n"), "."]),
        _line("R", "S2", ["With simple interest the balance is ", Fm(r"P(1+rn)", "P(1 + r n)"), "."]),
        _line("D", None, ["The extra earned by compounding is the compound balance minus the simple balance."]),
        _line("C", None, [f"{int(comp)} - {int(simp)} = {int(diff)}."]),
    ]
    return dict(
        sources=[("S1", list(COMPOUND)), ("S2", list(SIMPLE))],
        problem=[
            f"A balance of {P} dollars earns {pct}% per period for {n} periods. How much more does it have with compound interest than with simple interest?"
        ],
        lines=lines,
        answer=diff,
    )


HARD = [_h_split, _h_tail, _h_interest]


def _e_cubes(rng) -> dict:
    n = rng.randint(4, 12)
    s1, s2, s3 = n * (n + 1) // 2, n * (n + 1) * (2 * n + 1) // 6, (n * (n + 1) // 2) ** 2
    check(s3 == sum(k**3 for k in range(1, n + 1)), "cubes")
    v = sum(k * (k + 1) * (k + 2) for k in range(1, n + 1))
    check(v == s3 + 3 * s2 + 2 * s1, "cube split")
    lines = [
        _line("R", "S1", ["The sum of the first n integers is ", Fm(r"\frac{n(n+1)}{2}", "n(n+1)/2"), "."]),
        _line("R", "S2", ["The sum of the first n squares is ", Fm(r"\frac{n(n+1)(2n+1)}{6}", "n(n+1)(2n+1)/6"), "."]),
        _line(
            "X",
            "S2",
            ["The sum of the first n cubes is ", Fm(r"\left(\frac{n(n+1)}{2}\right)^{2}", "(n(n+1)/2)^2"), "."],
        ),
        _line(
            "D",
            None,
            [
                "Since ",
                Fm(r"k(k+1)(k+2)=k^{3}+3k^{2}+2k", "k(k + 1)(k + 2) = k^3 + 3k^2 + 2k"),
                ", the required sum is the sum of cubes plus 3 times the sum of squares plus 2 times the sum of integers.",
            ],
        ),
        _line("C", None, [f"For n = {n}: {s3} + 3 x {s2} + 2 x {s1} = {v}."]),
    ]
    return dict(
        sources=[("S1", list(ARITH)), ("S2", list(SQUARES))],
        problem=[
            "Find ",
            Fm(
                rf"1\cdot 2\cdot 3+2\cdot 3\cdot 4+\cdots+{n}({n + 1})({n + 2})",
                f"1 x 2 x 3 + 2 x 3 x 4 + ... + {n} x {n + 1} x {n + 2}",
            ),
            ".",
        ],
        lines=lines,
        answer=Fraction(v),
        depends=[4, 5],
    )


def _e_weighted(rng) -> dict:
    c = rng.choice((2, 3, 4, 5))
    r = rng.choice((Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(2, 3)))
    v = r / (1 - r) ** 2
    check(abs(sum(k * r**k for k in range(1, 400)) - v) < Fraction(1, 10**9), "weighted series")
    lines = [
        _line(
            "R",
            "S1",
            [
                "The series ",
                Fm(r"a+ar+ar^{2}+\cdots", "a + a r + a r^2 + ..."),
                " has sum ",
                Fm(r"\frac{a}{1-r}", "a/(1 - r)"),
                " when ",
                Fm(r"|r|<1", "|r| < 1"),
                ".",
            ],
        ),
        _line(
            "X",
            "S1",
            [
                "The series ",
                Fm(r"r+2r^{2}+3r^{3}+\cdots", "r + 2r^2 + 3r^3 + ..."),
                " has sum ",
                Fm(r"\frac{r}{(1-r)^{2}}", "r/(1 - r)^2"),
                ".",
            ],
        ),
        _line("C", None, ["With r = ", fr(r), ", ", Fm(r"\frac{r}{(1-r)^{2}}", "r/(1 - r)^2"), " = ", fr(v), "."]),
        _line("D", None, [f"Multiplying every term by {c} multiplies the sum by {c}."]),
        _line("C", None, [f"{c} x ", fr(v), " = ", fr(c * v), "."]),
    ]
    return dict(
        sources=[("S1", list(GEOM)), ("S2", list(ARITH))],
        problem=[f"Find the sum of the series {c}r + {2 * c}r^2 + {3 * c}r^3 + ... when r = ", fr(r), "."],
        lines=lines,
        answer=c * v,
        depends=[3, 4, 5],
    )


EXAM = [_e_cubes, _e_weighted]


# ---------------------------------------------------------------------------
# Rendering and marking
# ---------------------------------------------------------------------------
def _true_tag(ln: dict) -> str:
    return {"R": ln["src"], "C": "calculation", "D": "own reasoning", "X": "no source"}[ln["kind"]]


def _shown_tag(ln: dict) -> str:
    return ln["src"] if ln["kind"] == "X" else _true_tag(ln)


def _corrupt(rng, lines: list[dict], k: int, labels: list[str], avoid: set[int]) -> dict[int, str]:
    """Wrong attributions: {line index: tag shown}."""
    out: dict[int, str] = {}
    idxs = [i for i in range(len(lines)) if i not in avoid]
    rng.shuffle(idxs)
    for i in idxs[:k]:
        ln = lines[i]
        pool = [t for t in labels + ["calculation", "own reasoning"] if t != _shown_tag(ln)]
        out[i] = rng.choice(pool)
    return out


def _render(
    rng, spec: dict, *, show_tags: bool, wrong: dict[int, str], want_values: bool, want_unsourced: bool, tier: str
) -> dict:
    p = Prompt()
    p.t("Supplied source" + ("s" if len(spec["sources"]) > 1 else "") + " (the only ones you may cite): ")
    for lab, parts in spec["sources"]:
        p.t(f"{lab}: ")
        _emit(p, list(parts))
        p.t(" ")
    p.t("A student's notes on the problem. ")
    _emit(p, ["Problem: "] + spec["problem"] + [" "])
    for i, ln in enumerate(spec["lines"], 1):
        p.t(f"Line {i}: ")
        _emit(p, ln["parts"])
        if show_tags:
            shown = wrong.get(i - 1, _shown_tag(ln))
            p.t(f" (attributed to: {shown}) ")
        else:
            p.t(" ")
    q = "For each line, say whether it is reproduced from a supplied source, computed, or a new deduction that the sources do not state, and cite the source label for each reproduced line. "
    if want_values:
        q += "For each calculation, give the value it produces. "
    if show_tags:
        q = "For each line, say whether it is reproduced from a supplied source, computed, or a new deduction that the sources do not state. Then say which attributions are wrong and give the right provenance for each. "
    if want_unsourced:
        q += "One line states a result that neither supplied source gives: say which line it is, why it cannot be cited to a source, and which later lines depend on it. "
    p.t(q).t(rng.choice(SHORT_TAILS))

    lines = spec["lines"]
    ref = []
    for i, ln in enumerate(lines, 1):
        w = KIND_WORD[ln["kind"]]
        if ln["kind"] == "R":
            w += f" ({ln['src']})"
        ref.append(f"Line {i}: {w}: {_plain(ln['parts'])}")
    for i, tag in sorted(wrong.items()):
        ln = lines[i]
        ref.append(f"Line {i + 1} is attributed to {tag}, which is wrong: its provenance is {_true_tag(ln)}.")
    if want_unsourced:
        xi = next(i for i, ln in enumerate(lines) if ln["kind"] == "X")
        dep = ", ".join(str(j) for j in spec["depends"])
        ref.append(
            f"Line {xi + 1} is not stated in any supplied source, so it cannot be cited to {lines[xi]['src']}; it needs its own source or a proof, and lines {dep} depend on it."
        )
    steps = [
        (
            "correct_idea",
            "classifies every line correctly: "
            + "; ".join(f"line {i} {KIND_WORD[ln['kind']].split(' (')[0]}" for i, ln in enumerate(lines, 1)),
        ),
        (
            "completeness",
            "explains why the new deductions are new: the source does not state them, they follow from it; and why the computed lines add nothing new",
        ),
        (
            "use_of_context",
            "cites the source by its label for each reproduced line ("
            + ", ".join(f"line {i} -> {ln['src']}" for i, ln in enumerate(lines, 1) if ln["kind"] == "R")
            + ")"
            + (
                f"; the value of each calculation is checked against the numbers: answer {spec['answer']}"
                if want_values
                else ""
            ),
        ),
        (
            "communication",
            "keeps each line separate and labels each with one of the provenance words, in a short answer",
        ),
    ]
    if show_tags:
        steps.insert(
            2,
            (
                "correct_idea",
                "finds exactly the wrong attributions: "
                + "; ".join(f"line {i + 1} should be {_true_tag(lines[i])}" for i in sorted(wrong)),
            ),
        )
    if want_unsourced:
        steps.insert(
            2,
            (
                "completeness",
                "says the line stating a result outside both sources cannot be cited to a source, and that the lines using it rest on that unsupported result",
            ),
        )
    traps = [
        (
            "computation_as_reproduction",
            "does it call a calculation a reproduced fact, or a reproduced fact a new result?",
        ),
        ("deduction_cited_to_source", "does it cite a supplied source for a line that the source does not state?"),
        ("fabricated_citation", "does it invent a source, an author or a page that was not supplied?"),
    ]
    if show_tags:
        traps.append(
            ("accepts_wrong_attribution", "does it accept an attribution that does not match what the line does?")
        )
    if want_unsourced:
        traps.append(("accepts_unsourced_line", "does it accept the unsupported line because its content is true?"))
    ans = " ".join(f"Line {i}: {KIND_WORD[ln['kind']].split(' (')[0]}." for i, ln in enumerate(lines, 1))
    if wrong:
        ans += " Wrong attributions: " + ", ".join(f"line {i + 1}" for i in sorted(wrong)) + "."
    return finish(p, ans, short_marking(reference=ref, steps=steps, traps=traps))


def _easy(rng) -> dict:
    spec = rng.choice(SINGLE)(rng)
    spec = dict(spec, lines=[spec["lines"][i] for i in (0, 1, 2)])
    return _render(rng, spec, show_tags=False, wrong={}, want_values=False, want_unsourced=False, tier="easy")


def _medium(rng) -> dict:
    spec = rng.choice(SINGLE)(rng)
    return _render(rng, spec, show_tags=False, wrong={}, want_values=True, want_unsourced=False, tier="medium")


def _hard(rng) -> dict:
    spec = rng.choice(HARD)(rng)
    wrong = _corrupt(rng, spec["lines"], rng.choice((1, 2)), ["S1", "S2"], set())
    return _render(rng, spec, show_tags=True, wrong=wrong, want_values=False, want_unsourced=False, tier="hard")


def _exam(rng) -> dict:
    spec = rng.choice(EXAM)(rng)
    xi = next(i for i, ln in enumerate(spec["lines"]) if ln["kind"] == "X")
    wrong = _corrupt(rng, spec["lines"], rng.choice((0, 1)), ["S1", "S2"], {xi})
    return _render(rng, spec, show_tags=True, wrong=wrong, want_values=False, want_unsourced=True, tier="exam")


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("source_provenance", level=18, presets=_PRESETS)
def source_provenance(rng, variant="easy"):
    """`ucv1-source-computation-provenance`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
