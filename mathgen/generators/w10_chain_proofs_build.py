"""w10: school-level proofs written as typed steps, marked line by line by the `step_chain` grader.

Four proof families, each with easy / medium / hard tiers and a two-part exam chain:

* ``w10_proof_irrational_surd``: proof by contradiction that a surd is irrational (a square root of a prime, a
  rational plus a multiple of one, a sum of two surds). The learner types the algebraic step that carries the
  argument; the supposition (``p = q√2``) is a hypothesis the grader reads every line under.
* ``w10_proof_parity``: proof by contradiction / cases about odd and even whole numbers. The learner shows the
  algebra that puts an expression in the form ``2(...) + 1`` or ``2(...)``.
* ``w10_proof_am_gm``: inequality proofs that reduce to "this difference is a square": a² + b² − 2ab = (a − b)²,
  its weighted and fractional forms, and the AM-GM inequality for two positive numbers.
* ``w10_proof_triangle``: the triangle inequality |a + b| ≤ |a| + |b| by squaring.

Every question gives the statement, the start and the form to finish in, and asks for one expression per line.
The proof's conclusion ("a square is never negative, so ...") is in the prompt, not asked: nothing here is marked
by a person. A question's `answer` is the chain-proof marking dict of ``_chain_proof_kit``:
``{start, target, symbols, reference_lines, given?, traps, checks}``, graded by ``mathgen.grading.chain``.

The answer is built first (a reference chain), checked a second way here (exact simplification, then 40-digit
numeric evaluation at fresh points) and then every wrong chain offered as a distractor is checked to grade wrong.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import sympy as sp

from ..answers import Distractor
from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import M, T

_ASSUME = {
    "real": {"real": True},
    "nonzero": {"real": True, "nonzero": True},
    "positive": {"positive": True},
    "integer": {"integer": True},
}
_SECOND_POINTS = {
    "real": ("13/7", "-9/4", "17/5", "-3/11"),
    "nonzero": ("13/7", "-9/4", "17/5", "-3/11"),
    "positive": ("13/7", "9/4", "17/5", "3/11"),
    "integer": ("-11", "8", "13", "-4"),
}

NOTE_PLAIN = "Use ^ for powers and write products side by side, as in 2xy."
NOTE_ROOT = "Use ^ for powers, write a square root as √2 or sqrt(2), and products side by side, as in 2xy."
NOTE_ABS = "Use ^ for powers, write an absolute value as |x| or abs(x), and products side by side, as in 2xy."


# ---------------------------------------------------------------------------------------------------------------
# Display and rendering
# ---------------------------------------------------------------------------------------------------------------
def show(sym: str) -> str:
    """The learner-facing, typeable form of a sympy-style line: ``^`` powers, ``√`` roots, ``|x|`` bars."""
    s = kit.disp(sym)
    s = re.sub(r"sqrt\((\d+|[A-Za-z])\)", r"√\1", s)
    s = s.replace("sqrt(", "√(")
    s = re.sub(r"abs\(([^()]*)\)", r"|\1|", s)
    return s


def render(parts: list[tuple]) -> tuple[str, str]:
    text, latex = [], []
    for part in parts:
        if part[0] == "t":
            text.append(part[1])
            latex.append(kit._text_verbatim(part[1]))
        elif part[0] == "m":
            text.append(show(part[1]))
            latex.append(kit.tex(part[1]))
        else:
            text.append(part[1])
            latex.append(part[2])
    return "".join(text), "".join(latex)


def _chain_text(lines: list[str]) -> str:
    return " = ".join(show(line) for line in lines)


# ---------------------------------------------------------------------------------------------------------------
# Checking a family
# ---------------------------------------------------------------------------------------------------------------
def _locals(symbols: dict, given: dict) -> dict:
    ns = {name: sp.Symbol(name, **_ASSUME[kind]) for name, kind in symbols.items()}
    for name in given:
        ns.setdefault(name, sp.Symbol(name, real=True))
    ns.update({"sqrt": sp.sqrt, "abs": sp.Abs})
    return ns


def _second_route(ref: list[str], symbols: dict, given: dict, key: str) -> None:
    """Every consecutive pair of the reference chain is equal at 40 digits, at points the grader does not use.
    Independent of `mathgen.grading.chain`."""
    ns = _locals(symbols, given)
    sub = {ns[name]: sp.sympify(expr, locals=ns) for name, expr in given.items()}
    exprs = [sp.sympify(line, locals=ns).xreplace(sub) for line in ref]
    names = sorted(symbols)
    for i in range(1, len(exprs)):
        diff = exprs[i] - exprs[i - 1]
        for p in range(4):
            env = {ns[n]: sp.Rational(_SECOND_POINTS[symbols[n]][(p + j) % 4]) for j, n in enumerate(names)}
            value = sp.N(diff.xreplace(env), 40)
            check(abs(value) < sp.Float("1e-30"), f"{key}: line {i + 1} differs from line {i} at {env}")


def _grades(spec: dict, answer_text: str, reply: str) -> bool:
    from ..grading.chain import grade_step_chain

    stub = SimpleNamespace(answer=spec, answer_text=answer_text)
    return grade_step_chain(stub, reply).correct


def _spec(fam: dict) -> dict:
    ref = list(fam["ref"])
    spec = {
        "start": ref[0],
        "target": ref[-1],
        "symbols": dict(fam["symbols"]),
        "reference_lines": ref,
    }
    if fam.get("given"):
        spec["given"] = dict(fam["given"])
    spec["checks"] = [
        {
            "type": "equivalence_chain",
            "start": ref[0],
            "target": ref[-1],
            "symbols": dict(fam["symbols"]),
            "authoritative": True,
        }
    ]
    if fam.get("given"):
        spec["checks"][0]["given"] = dict(fam["given"])
    return spec


def _tagged(fam: dict, spec: dict, answer_text: str) -> tuple[list[Distractor], list[dict]]:
    """The wrong chains as tagged distractors; each is proved wrong by the grader."""
    key = fam["key"]
    check(_grades(spec, answer_text, answer_text), f"{key}: the grader rejects the reference chain")
    out, traps, seen = [], [], {answer_text}
    for mid, lines, why in fam["wrongs"]:
        text = _chain_text(lines)
        check(text not in seen, f"{key}: repeated or answer-equal wrong chain {text!r}")
        check(not _grades(spec, answer_text, text), f"{key}: wrong chain {text!r} is accepted")
        check("*" not in text, f"{key}: star in {text!r}")
        seen.add(text)
        out.append(Distractor(text, misconception=mid))
        traps.append(
            {"name": mid, "question": why, "example_lines": list(lines), "expected_check": "authoritative_failure"}
        )
    check(len(out) >= 3, f"{key}: fewer than three wrong chains")
    return out, traps


def _prompt_parts(rng, fam: dict) -> list[tuple]:
    ref = fam["ref"]
    openers = (
        (" Start from ", " and finish at ", ", writing one expression per line. "),
        (" Begin with ", " and end at ", ", one expression on each line. "),
        (" Write a chain of equal expressions, one per line, from ", " to ", ". "),
    )
    a, b, c = openers[rng.randrange(len(openers))]
    return [*fam["lead"], T(a), M(ref[0]), T(b), M(ref[-1]), T(c + fam["note"])]


_ONE_TIMES = re.compile(r"(?<![\w.)])1\*")


def tidy(line: str) -> str:
    """`line` without a coefficient of 1 (``1*u`` is ``u``): the learner is shown, and asked to type, no such factor."""
    return _ONE_TIMES.sub("", line)


def build(rng, fam: dict) -> dict:
    """The question dict for one family: the prompt, the marking dict, the typed answer and the tagged wrong chains."""
    fam = {
        **fam,
        "ref": [tidy(line) for line in fam["ref"]],
        "wrongs": [(mid, [tidy(line) for line in lines], why) for mid, lines, why in fam["wrongs"]],
    }
    ref = list(fam["ref"])
    _second_route(ref, fam["symbols"], fam.get("given") or {}, fam["key"])
    spec = _spec(fam)
    answer_text = _chain_text(ref)
    distractors, traps = _tagged(fam, spec, answer_text)
    spec["traps"] = traps
    text, latex = render(_prompt_parts(rng, fam))
    check("*" not in text and "**" not in text, f"{fam['key']}: star in the prompt")
    return dict(
        text=text,
        latex=latex,
        answer=spec,
        answer_text=answer_text,
        answer_kind="step_chain",
        distractors=distractors,
        working=[f"{show(a)} = {show(b)}: {why}" for a, b, why in zip(ref, ref[1:], fam["reasons"])],
    )


def build_parts(rng, stem: str, stem_latex: str, fams: list[dict], marks: list[int]) -> dict:
    """A multi-part exam chain: each family becomes one `step_chain` part."""
    parts = []
    for fam, mark in zip(fams, marks):
        item = build(rng, fam)
        parts.append(
            dict(
                prompt=item["text"],
                prompt_latex=item["latex"],
                answer_kind="step_chain",
                answer=item["answer"],
                answer_text=item["answer_text"],
                marks=mark,
                distractors=item["distractors"],
                working=item["working"],
            )
        )
    return dict(text=stem, latex=stem_latex, parts=parts)


def pick(rng, items):
    return items[rng.randrange(len(items))]


def tier_families(table: dict, tier: str):
    """The families of one tier; an unknown tier is a `ValueError`, as for every tiered generator."""
    if tier not in table:
        raise ValueError(f"unknown tier {tier!r}; known: {', '.join(table)}")
    return table[tier]


# ---------------------------------------------------------------------------------------------------------------
# Proof by contradiction: irrational surds
# ---------------------------------------------------------------------------------------------------------------
_ROOTS = (2, 3, 5, 6, 7, 10, 11, 13)
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23)
_PQ = (("p", "q"), ("m", "d"), ("x", "y"), ("u", "v"), ("r", "s"))
_PQK = (("p", "q", "k"), ("m", "d", "t"), ("x", "y", "z"), ("u", "v", "w"), ("r", "s", "h"))


def surd_square(rng, n=None, pq=None, form=None) -> dict:
    n = pick(rng, _ROOTS) if n is None else n
    p, q = pick(rng, _PQ) if pq is None else pq
    form = rng.randrange(2) if form is None else form
    if form == 0:
        ref = [f"{p}**2", f"({q}*sqrt({n}))**2", f"{n}*{q}**2"]
        lead = (
            f"To prove by contradiction that √{n} is irrational, suppose √{n} = {p}/{q} for whole numbers "
            f"{p} and {q}, so that {p} = {q}√{n}. Show that {p}^2 = {n}{q}^2."
        )
        symbols, given = {q: "positive"}, {p: f"{q}*sqrt({n})"}
        reasons = [f"replace {p} by {q}√{n}.", f"square the product: (√{n})^2 = {n}, and {q} is squared too."]
        wrongs = [
            (
                "surd-root-not-squared",
                [ref[0], ref[1], f"{q}**2*sqrt({n})"],
                "is the root squared as well as the letter?",
            ),
            (
                "surd-forgot-square-on-q",
                [ref[0], ref[1], f"{n}*{q}"],
                "is the whole product squared, including the letter?",
            ),
            ("surd-square-of-product-as-sum", [ref[0], f"({q}+sqrt({n}))**2", ref[2]], "is (q√n)^2 a product squared?"),
            ("surd-root-not-squared", [ref[0], ref[1], f"{2 * n}*{q}**2"], "does squaring √n give n, not 2n?"),
        ]
    else:
        ref = [f"{q}**2", f"({p}/sqrt({n}))**2", f"{p}**2/{n}"]
        lead = (
            f"To prove by contradiction that √{n} is irrational, suppose √{n} = {p}/{q} for whole numbers "
            f"{p} and {q}, so that {q} = {p}/√{n}. Show that {q}^2 = {p}^2/{n}."
        )
        symbols, given = {p: "positive"}, {q: f"{p}/sqrt({n})"}
        reasons = [
            f"replace {q} by {p}/√{n}.",
            f"square the quotient: top and bottom are both squared, (√{n})^2 = {n}.",
        ]
        wrongs = [
            ("surd-root-not-squared", [ref[0], ref[1], f"{p}**2/sqrt({n})"], "is the root in the denominator squared?"),
            ("surd-forgot-square-on-q", [ref[0], ref[1], f"{p}/{n}"], "is the letter in the numerator squared?"),
            ("surd-quotient-inverted", [ref[0], ref[1], f"{n}/{p}**2"], "is the quotient the right way up?"),
            ("surd-square-of-product-as-sum", [ref[0], f"({p}+sqrt({n}))**2", ref[2]], "is the square of a quotient?"),
        ]
    return dict(
        key=f"surd_square_{form}",
        lead=[T(lead)],
        note=NOTE_ROOT,
        ref=ref,
        symbols=symbols,
        given=given,
        reasons=reasons,
        wrongs=wrongs,
    )


def surd_cancel(rng, n=None, pqk=None, form=None) -> dict:
    n = pick(rng, _PRIMES) if n is None else n
    p, q, k = pick(rng, _PQK) if pqk is None else pqk
    form = rng.randrange(2) if form is None else form
    if form == 0:
        ref = [f"({n}*{k})**2/{n}", f"{n * n}*{k}**2/{n}", f"{n}*{k}**2"]
        lead = (
            f"In the proof that √{n} is irrational, {p}^2 = {n}{q}^2 shows {n} divides {p}, so {p} = {n}{k} and "
            f"{q}^2 = ({n}{k})^2/{n}. Simplify to write {q}^2 in terms of {k}."
        )
        wrongs = [
            ("surd-cancels-factor-wrongly", [ref[0], ref[1], f"{n * n}*{k}**2"], "was the division by n carried out?"),
            ("surd-cancels-factor-wrongly", [ref[0], ref[1], f"{n * n}*{k}"], "is k still squared after cancelling?"),
            (
                "surd-square-of-product-as-sum",
                [ref[0], f"({n * n}+{k}**2)/{n}", ref[2]],
                "is (nk)^2 a product squared?",
            ),
            ("surd-cancels-factor-wrongly", [ref[0], ref[1], f"{n}*{k}"], "is the whole k^2 kept?"),
        ]
        reasons = [f"square both factors of {n}{k}.", f"divide {n}^2 by {n}, leaving {n}."]
    else:
        ref = [f"({n}*{k})**2", f"{n * n}*{k}**2"]
        lead = (
            f"In the proof that √{n} is irrational, {p} = {n}{k}, so {p}^2 = ({n}{k})^2. Use {p}^2 = {n}{q}^2 to "
            f"write {n}{q}^2 in terms of {k}: first expand ({n}{k})^2."
        )
        wrongs = [
            ("surd-forgot-square-on-q", [ref[0], f"{n}*{k}**2"], "is the number squared as well as the letter?"),
            ("surd-forgot-square-on-q", [ref[0], f"{n * n}*{k}"], "is the letter squared as well as the number?"),
            ("surd-square-of-product-as-sum", [ref[0], f"{n * n}+{k}**2"], "is a product squared, not a sum?"),
            ("surd-forgot-square-on-q", [ref[0], f"{n + 1}*{k}**2"], "does squaring n·k give n^2·k^2?"),
        ]
        reasons = [f"square each factor of {n}{k}."]
    return dict(
        key="surd_cancel",
        lead=[T(lead)],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={k: "integer"},
        wrongs=wrongs,
        reasons=reasons,
    )


def surd_rational_plus(rng) -> dict:
    n = pick(rng, (2, 3, 5, 6, 7))
    a = pick(rng, (1, 2, 3, 4, 5))
    b = pick(rng, (2, 3, 4, 5))
    ref = [f"(r-{a})/{b}", f"({a}+{b}*sqrt({n})-{a})/{b}", f"({b}*sqrt({n}))/{b}", f"sqrt({n})"]
    return dict(
        key="surd_rational_plus",
        lead=[
            T(
                f"Given that √{n} is irrational, prove that {a} + {b}√{n} is irrational. Suppose "
                f"{a} + {b}√{n} = r for a rational number r. Show that (r - {a})/{b} = √{n}, which would make "
                f"√{n} rational."
            )
        ],
        note=NOTE_ROOT,
        ref=ref,
        symbols={},
        given={"r": f"{a}+{b}*sqrt({n})"},
        reasons=["replace r by its value.", f"the {a}s cancel.", f"divide the {b} by the {b}."],
        wrongs=[
            ("surd-divides-only-the-root", [ref[0], f"sqrt({n})-{a}/{b}", f"sqrt({n})"], "is every term divided by b?"),
            ("surd-forgot-to-divide-by-b", [ref[0], f"{b}*sqrt({n})"], "was r - a divided by b?"),
            ("surd-sign-of-a-flipped", [f"(r+{a})/{b}", ref[1], ref[3]], "does the working start from (r - a)/b?"),
            (
                "surd-divides-only-the-root",
                [ref[0], f"({b}*sqrt({n}))/{b}", f"{b}*sqrt({n})/{b}**2"],
                "does b/b cancel to 1?",
            ),
        ],
    )


def surd_sum_of_roots(rng) -> dict:
    m, n = rng.sample((2, 3, 5, 7), 2)
    mn = m * n
    ref = [
        f"(r**2+{m}-{n})/(2*r)",
        f"((sqrt({m})+sqrt({n}))**2+{m}-{n})/(2*(sqrt({m})+sqrt({n})))",
        f"(2*{m}+2*sqrt({mn}))/(2*(sqrt({m})+sqrt({n})))",
        f"(2*sqrt({m})*(sqrt({m})+sqrt({n})))/(2*(sqrt({m})+sqrt({n})))",
        f"sqrt({m})",
    ]
    return dict(
        key="surd_sum_of_roots",
        lead=[
            T(
                f"Suppose r = √{m} + √{n} were rational. Then r - √{m} = √{n}, and squaring gives "
                f"r^2 - 2r√{m} + {m} = {n}. Show that (r^2 + {m} - {n})/(2r) = √{m}, so √{m} would be a "
                f"quotient of rational numbers."
            )
        ],
        note=NOTE_ROOT,
        ref=ref,
        symbols={},
        given={"r": f"sqrt({m})+sqrt({n})"},
        reasons=[
            "replace r by √m + √n.",
            f"expand the square: (√{m} + √{n})^2 = {m} + {n} + 2√{mn}, then {m} + {n} + {m} - {n} = 2·{m} + 2√{mn}.",
            f"factor the numerator: {m} + √{mn} = √{m}(√{m} + √{n}).",
            "cancel the common factor.",
        ],
        wrongs=[
            (
                "surd-sum-cross-term-forgotten",
                [ref[0], ref[1], f"(2*{m})/(2*(sqrt({m})+sqrt({n})))", f"sqrt({m})"],
                "does squaring a sum keep the cross term?",
            ),
            (
                "surd-cancels-factor-wrongly",
                [ref[0], ref[1], ref[2], f"sqrt({m})+sqrt({n})", f"sqrt({m})"],
                "does the numerator factor as √m(√m + √n)?",
            ),
            (
                "surd-sign-of-a-flipped",
                [ref[0], ref[1], f"(2*{m}-2*sqrt({mn}))/(2*(sqrt({m})+sqrt({n})))", ref[4]],
                "is the cross term added, not subtracted?",
            ),
        ],
    )


_SURD = {
    "easy": (surd_square,),
    "medium": (surd_cancel,),
    "hard": (surd_rational_plus, surd_sum_of_roots),
}


@generator("w10_proof_irrational_surd", level=13, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
def w10_proof_irrational_surd(rng, tier: str = "easy"):
    return build(rng, pick(rng, tier_families(_SURD, tier))(rng))


@generator("w10_proof_irrational_surd_exam", level=13, presets={"exam": {}})
def w10_proof_irrational_surd_exam(rng):
    n = pick(rng, _PRIMES)
    p, q, k = pick(rng, _PQK)
    first = surd_square(rng, n=n, pq=(p, q))
    second = surd_cancel(rng, n=n, pqk=(p, q, k))
    return build_parts(
        rng,
        f"A proof that √{n} is irrational begins by supposing the opposite.",
        r"\text{A proof that }\sqrt{" + str(n) + r"}\text{ is irrational begins by supposing the opposite.}",
        [first, second],
        [2, 3],
    )


# ---------------------------------------------------------------------------------------------------------------
# Odd and even
# ---------------------------------------------------------------------------------------------------------------
_K = ("k", "m", "t", "u")
_KM = (("k", "m"), ("t", "u"), ("v", "w"))


def parity_odd_square(rng, k=None, c=None) -> dict:
    c = pick(rng, (1, 3, 5, 7)) if c is None else c
    k = pick(rng, _K) if k is None else k
    half = (c * c - 1) // 2
    inner = f"2*{k}**2+{2 * c}*{k}" + (f"+{half}" if half else "")
    ref = [f"(2*{k}+{c})**2", f"4*{k}**2+{4 * c}*{k}+{c * c}", f"2*({inner})+1"]
    return dict(
        key="parity_odd_square",
        lead=[
            T(
                f"To prove by contradiction that if n^2 is even then n is even, suppose n is odd, so n = 2{k} + {c} "
                f"for a whole number {k}. Show that n^2 is one more than an even number, writing it as 2(...) + 1."
            )
        ],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={k: "integer"},
        reasons=["expand the square, keeping the middle term.", "take a factor of 2 out of the first terms."],
        wrongs=[
            (
                "parity-odd-square-forgets-cross-term",
                [ref[0], f"4*{k}**2+{c * c}", f"2*(2*{k}**2)+{c * c}"],
                "does squaring a sum keep the middle term?",
            ),
            (
                "parity-square-of-sum-termwise",
                [ref[0], f"2*{k}**2+{c * c}", f"2*({k}**2)+{c * c}"],
                "is (2k + c)^2 the same as (2k)^2 + c^2?",
            ),
            (
                "parity-odd-form-off-by-one",
                [ref[0], ref[1], f"2*({inner}+1)"],
                "does the last line equal the expanded square, one more than an even number?",
            ),
        ],
    )


def parity_odd_plus_even(rng) -> dict:
    c = pick(rng, (1, 3, 5, 7))
    k, m = pick(rng, _KM)
    half = (c - 1) // 2
    last = f"2*({k}+{m}" + (f"+{half}" if half else "") + ")+1"
    ref = [f"(2*{k}+{c})+2*{m}", f"2*{k}+2*{m}+{c}", last]
    return dict(
        key="parity_odd_plus_even",
        lead=[
            T(
                f"Show that the sum of an odd number 2{k} + {c} and an even number 2{m} is odd, by writing it as "
                f"2(...) + 1."
            )
        ],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={k: "integer", m: "integer"},
        reasons=["remove the brackets.", "take a factor of 2 out of the terms that have one; the odd leftover is 1."],
        wrongs=[
            (
                "parity-odd-form-off-by-one",
                [ref[0], ref[1], f"2*({k}+{m}+{c})+1"],
                "is the odd part c left outside the bracket as +1 only?",
            ),
            (
                "parity-odd-form-off-by-one",
                [ref[0], ref[1], f"2*({k}+{m}) + {c + 1}"],
                "does the last line equal the line before it?",
            ),
            (
                "parity-factor-not-taken-out-fully",
                [ref[0], f"2*{k}+{m}+{c}", last],
                "is the even number 2m kept whole?",
            ),
        ],
    )


def parity_product_of_odds(rng) -> dict:
    r, s = pick(rng, (1, 3, 5, 7)), pick(rng, (1, 3, 5, 7))
    k, m = pick(rng, _KM)
    mid = f"4*{k}*{m}+{2 * s}*{k}+{2 * r}*{m}+{r * s}"
    inner = f"2*{k}*{m}+{s}*{k}+{r}*{m}" + (f"+{(r * s - 1) // 2}" if (r * s - 1) // 2 else "")
    ref = [f"(2*{k}+{r})*(2*{m}+{s})", mid, f"2*({inner})+1"]
    return dict(
        key="parity_product_of_odds",
        lead=[
            T(
                f"Show that the product of the odd numbers 2{k} + {r} and 2{m} + {s} is odd, by writing it as "
                f"2(...) + 1."
            )
        ],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={k: "integer", m: "integer"},
        reasons=["expand: every pair of terms is multiplied.", "take a factor of 2 out of every term but the last."],
        wrongs=[
            (
                "parity-product-misses-cross-term",
                [ref[0], f"4*{k}*{m}+{r * s}", f"2*(2*{k}*{m})+{r * s}"],
                "does expanding two brackets give four products?",
            ),
            (
                "parity-product-misses-cross-term",
                [ref[0], f"4*{k}*{m}+{s}*{k}+{r}*{m}+{r * s}", ref[2]],
                "is each cross term multiplied by 2 as well as by the letter?",
            ),
            (
                "parity-odd-form-off-by-one",
                [ref[0], mid, f"2*({inner}+1)"],
                "does the last line equal the expanded product?",
            ),
        ],
    )


def parity_odd_square_minus_one(rng, k=None, c=None) -> dict:
    c = pick(rng, (1, 3, 5, 7)) if c is None else c
    k = pick(rng, _K) if k is None else k
    lo, hi = (c - 1) // 2, (c + 1) // 2
    a = f"{k}" if lo == 0 else f"({k}+{lo})"
    ref = [f"(2*{k}+{c})**2-1", f"4*{k}**2+{4 * c}*{k}+{c * c - 1}", f"4*{a}*({k}+{hi})"]
    return dict(
        key="parity_odd_square_minus_one",
        lead=[
            T(
                f"For the odd number n = 2{k} + {c}, show that n^2 - 1 is a multiple of 4, by factorising it as "
                f"4 times a product of two brackets."
            )
        ],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={k: "integer"},
        reasons=[
            "expand the square, then subtract 1.",
            f"take out 4, then factorise {k}^2 + {c}{k} + {(c * c - 1) // 4}.",
        ],
        wrongs=[
            (
                "parity-odd-square-forgets-cross-term",
                [ref[0], f"4*{k}**2+{c * c - 1}", f"4*({k}**2+{(c * c - 1) // 4})"],
                "does squaring a sum keep the middle term?",
            ),
            (
                "parity-factor-not-taken-out-fully",
                [ref[0], ref[1], f"2*{a}*({k}+{hi})"],
                "is the whole factor 4 taken out?",
            ),
            (
                "parity-factor-not-taken-out-fully",
                [ref[0], ref[1], f"4*({k}+{hi})"],
                "are both brackets of the factorisation kept?",
            ),
        ],
    )


def parity_quadratic_even(rng, k=None, c=None) -> dict:
    c = pick(rng, (1, 3, 5, 7)) if c is None else c
    k = pick(rng, _K) if k is None else k
    half = (c + 1) // 2
    ref = [
        f"(2*{k}+1)**2+{c}*(2*{k}+1)",
        f"4*{k}**2+{4 + 2 * c}*{k}+{1 + c}",
        f"2*(2*{k}**2+{2 + c}*{k}+{half})",
    ]
    return dict(
        key="parity_quadratic_even",
        lead=[
            T(
                f"Show that n^2 + {c}n is even when n is odd, by putting n = 2{k} + 1 and writing the result as "
                f"2 times a bracket."
            )
        ],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={k: "integer"},
        reasons=["expand both brackets and collect like terms.", "every term is even, so take out a factor of 2."],
        wrongs=[
            (
                "parity-odd-square-forgets-cross-term",
                [ref[0], f"4*{k}**2+1+{c * 2}*{k}+{c}", f"2*(2*{k}**2+{c}*{k})+{1 + c}"],
                "does squaring 2k + 1 keep the middle term 4k?",
            ),
            (
                "parity-factor-not-taken-out-fully",
                [ref[0], ref[1], f"2*(2*{k}**2+{4 + 2 * c}*{k}+{half})"],
                "is the factor 2 taken out of every term?",
            ),
            (
                "parity-factor-not-taken-out-fully",
                [ref[0], ref[1], f"2*(2*{k}**2+{2 + c}*{k}+{1 + c})"],
                "is the last term halved as well?",
            ),
        ],
    )


def parity_odd_cube(rng) -> dict:
    k = pick(rng, _K)
    ref = [
        f"(2*{k}+1)**3+(2*{k}+1)",
        f"8*{k}**3+12*{k}**2+8*{k}+2",
        f"2*(4*{k}**3+6*{k}**2+4*{k}+1)",
    ]
    return dict(
        key="parity_odd_cube",
        lead=[
            T(
                f"Show that n^3 + n is even when n is odd, by putting n = 2{k} + 1 and writing the result as 2 times a bracket."
            )
        ],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={k: "integer"},
        reasons=[
            "expand the cube (binomial theorem) and add the second bracket.",
            "every term is even, so take out a factor of 2.",
        ],
        wrongs=[
            (
                "parity-square-of-sum-termwise",
                [ref[0], f"8*{k}**3+1+2*{k}+1", f"2*(4*{k}**3+{k}+1)"],
                "is (2k + 1)^3 the same as (2k)^3 + 1?",
            ),
            (
                "parity-factor-not-taken-out-fully",
                [ref[0], ref[1], f"2*(4*{k}**3+6*{k}**2+4*{k})+2"],
                "does the last line equal the line before it, as 2 times a bracket?",
            ),
            (
                "parity-factor-not-taken-out-fully",
                [ref[0], ref[1], f"2*(4*{k}**3+6*{k}**2+4*{k}+2)"],
                "is the last term halved as well?",
            ),
        ],
    )


def parity_even_plus_one(rng, k=None, c=None) -> dict:
    c = pick(rng, (1, 3, 5, 7)) if c is None else c
    k = pick(rng, _K) if k is None else k
    ref = [f"(2*{k})**2+{c}*(2*{k})+1", f"4*{k}**2+{2 * c}*{k}+1", f"2*(2*{k}**2+{c}*{k})+1"]
    return dict(
        key="parity_even_plus_one",
        lead=[
            T(
                f"Show that n^2 + {c}n + 1 is odd when n is even, by putting n = 2{k} and writing the result as "
                f"2(...) + 1."
            )
        ],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={k: "integer"},
        reasons=["square 2k and multiply out.", "take a factor of 2 out of the terms that have one."],
        wrongs=[
            (
                "parity-square-of-sum-termwise",
                [ref[0], f"2*{k}**2+{2 * c}*{k}+1", f"2*({k}**2+{c}*{k})+1"],
                "is (2k)^2 equal to 2k^2?",
            ),
            (
                "parity-factor-not-taken-out-fully",
                [ref[0], ref[1], f"2*(2*{k}**2+{2 * c}*{k})+1"],
                "is the factor 2 taken out of every term?",
            ),
            (
                "parity-odd-form-off-by-one",
                [ref[0], ref[1], f"2*(2*{k}**2+{c}*{k}+1)+1"],
                "does the last line equal the line before it?",
            ),
        ],
    )


_PARITY = {
    "easy": (parity_odd_square, parity_odd_plus_even),
    "medium": (parity_product_of_odds, parity_odd_square_minus_one),
    "hard": (parity_quadratic_even, parity_odd_cube, parity_even_plus_one),
}


@generator("w10_proof_parity", level=11, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
def w10_proof_parity(rng, tier: str = "easy"):
    return build(rng, pick(rng, tier_families(_PARITY, tier))(rng))


@generator("w10_proof_parity_exam", level=11, presets={"exam": {}})
def w10_proof_parity_exam(rng):
    k = pick(rng, _K)
    first = pick(rng, (parity_odd_square, parity_odd_square_minus_one))(rng, k=k)
    second = pick(rng, (parity_quadratic_even, parity_even_plus_one))(rng, k=k)
    return build_parts(
        rng,
        "A proof by contradiction about odd and even whole numbers n takes the odd case and the even case in turn.",
        r"\text{A proof by contradiction about odd and even whole numbers }n\text{ takes the odd case and the even "
        r"case in turn.}",
        [first, second],
        [2, 3],
    )


# ---------------------------------------------------------------------------------------------------------------
# Inequalities that reduce to a square: AM-GM
# ---------------------------------------------------------------------------------------------------------------
_AB = (("a", "b"), ("x", "y"), ("p", "q"), ("u", "v"))
_XY3 = (("x", "y"), ("u", "v"), ("s", "t"))


def amgm_sum_squares(rng) -> dict:
    a, b = pick(rng, _AB)
    form = rng.randrange(4)
    if form == 0:
        ref = [f"{a}**2+{b}**2-2*{a}*{b}", f"({a}-{b})**2"]
        lead = f"To prove {a}^2 + {b}^2 ≥ 2{a}{b}, first show that {a}^2 + {b}^2 - 2{a}{b} is a perfect square."
        reasons = ["a perfect square: the square of the difference."]
        wrongs = [
            ("amgm-cross-term-sign", [ref[0], f"({a}+{b})**2"], "is it the square of the difference?"),
            ("amgm-difference-of-squares", [ref[0], f"({a}-{b})*({a}+{b})"], "is a square two equal brackets?"),
            (
                "amgm-cross-term-halved",
                [ref[0], f"{a}**2-{a}*{b}+{b}**2", f"({a}-{b})**2"],
                "does (a - b)^2 expand with 2ab?",
            ),
        ]
    elif form == 1:
        ref = [
            f"({a}+{b})**2-4*{a}*{b}",
            f"{a}**2+2*{a}*{b}+{b}**2-4*{a}*{b}",
            f"{a}**2-2*{a}*{b}+{b}**2",
            f"({a}-{b})**2",
        ]
        lead = f"To prove ({a} + {b})^2 ≥ 4{a}{b}, first show that ({a} + {b})^2 - 4{a}{b} is a perfect square."
        reasons = ["expand the square.", "collect the like terms.", "recognise the square of the difference."]
        wrongs = [
            ("amgm-cross-term-sign", [ref[0], f"({a}+{b})**2"], "does each line equal the line before it?"),
            ("amgm-difference-of-squares", [ref[0], f"({a}-{b})*({a}+{b})"], "is a square two equal brackets?"),
            (
                "amgm-cross-term-halved",
                [ref[0], ref[1], f"{a}**2-{a}*{b}+{b}**2", ref[3]],
                "does collecting give -2ab?",
            ),
        ]
    elif form == 2:
        ref = [
            f"({a}-{b})**2+4*{a}*{b}",
            f"{a}**2-2*{a}*{b}+{b}**2+4*{a}*{b}",
            f"{a}**2+2*{a}*{b}+{b}**2",
            f"({a}+{b})**2",
        ]
        lead = f"To prove ({a} + {b})^2 ≥ 4{a}{b}, first show that ({a} - {b})^2 + 4{a}{b} equals ({a} + {b})^2."
        reasons = ["expand the square.", "collect the like terms.", "recognise the square of the sum."]
        wrongs = [
            ("amgm-cross-term-sign", [ref[0], f"({a}-{b})**2"], "does each line equal the line before it?"),
            (
                "amgm-difference-of-squares",
                [ref[0], f"({a}-{b})*({a}+{b})"],
                "does each line equal the line before it?",
            ),
            (
                "amgm-cross-term-halved",
                [ref[0], ref[1], f"{a}**2+{a}*{b}+{b}**2", ref[3]],
                "does collecting give +2ab?",
            ),
        ]
    else:
        a = pick(rng, ("x", "t", "u", "w", "s"))
        ref = [f"{a}**2+1-2*{a}", f"({a}-1)**2"]
        lead = f"To prove {a}^2 + 1 ≥ 2{a}, first show that {a}^2 + 1 - 2{a} is a perfect square."
        reasons = ["a perfect square: the square of the difference."]
        wrongs = [
            ("amgm-cross-term-sign", [ref[0], f"({a}+1)**2"], "is it the square of the difference?"),
            ("amgm-difference-of-squares", [ref[0], f"({a}-1)*({a}+1)"], "is a square two equal brackets?"),
            ("amgm-cross-term-halved", [ref[0], f"{a}**2-{a}+1", f"({a}-1)**2"], "does (x - 1)^2 expand with 2x?"),
        ]
        b = a
    return dict(
        key=f"amgm_sum_squares_{form}",
        lead=[T(lead + " A square is never negative, which will finish the proof.")],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={a: "real", b: "real"},
        reasons=reasons,
        wrongs=wrongs,
    )


def amgm_scaled(rng, x=None, y=None, c=None, d=None, form=None) -> dict:
    if x is None:
        x, y = pick(rng, _XY3)
    if c is None:
        c, d = rng.sample((1, 2, 3, 4), 2)
    if form is None:
        form = rng.randrange(2)
    if form:
        cx = f"{c}*{x}" if c > 1 else x
        dy = f"{d}*{y}" if d > 1 else y
        ref = [f"({c}*{x})**2+({d}*{y})**2-{2 * c * d}*{x}*{y}", f"({cx}-{dy})**2"]
        start_t = f"{c * c}{x}^2 + {d * d}{y}^2 - {2 * c * d}{x}{y}"
        lead = f"Show that {start_t} is a perfect square. This proves {c * c}{x}^2 + {d * d}{y}^2 ≥ {2 * c * d}{x}{y}."
        reasons = ["the first two terms are squares and the middle term is twice their product."]
        wrongs = [
            ("amgm-cross-term-sign", [ref[0], f"({cx}+{dy})**2"], "is the square of a difference?"),
            (
                "amgm-cross-term-halved",
                [ref[0], f"({c * c}*{x}-{d * d}*{y})**2"],
                "are the brackets squared roots of the terms?",
            ),
            ("amgm-difference-of-squares", [ref[0], f"({cx}-{dy})*({cx}+{dy})"], "is a square two equal brackets?"),
        ]
    else:
        ref = [
            f"({c}*{x}+{d}*{y})**2-{4 * c * d}*{x}*{y}",
            f"{c * c}*{x}**2+{2 * c * d}*{x}*{y}+{d * d}*{y}**2-{4 * c * d}*{x}*{y}",
            f"{c * c}*{x}**2-{2 * c * d}*{x}*{y}+{d * d}*{y}**2",
            f"({c}*{x}-{d}*{y})**2",
        ]
        lead = (
            f"Show that ({c}{x} + {d}{y})^2 - {4 * c * d}{x}{y} is a perfect square. This proves "
            f"({c}{x} + {d}{y})^2 ≥ {4 * c * d}{x}{y}."
        )
        reasons = ["expand the square.", "collect the like terms.", "recognise the square of the difference."]
        wrongs = [
            ("amgm-cross-term-sign", [ref[0], f"({c}*{x}+{d}*{y})**2"], "does each line equal the line before it?"),
            (
                "amgm-cross-term-halved",
                [ref[0], ref[1], f"{c * c}*{x}**2-{c * d}*{x}*{y}+{d * d}*{y}**2", ref[3]],
                "is the middle term 2cd?",
            ),
            (
                "amgm-difference-of-squares",
                [ref[0], ref[1], ref[2], f"({c}*{x})**2-({d}*{y})**2"],
                "is the last line the square of a difference?",
            ),
        ]
    return dict(
        key="amgm_scaled",
        lead=[T(lead + " A square is never negative.")],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={x: "real", y: "real"},
        reasons=reasons,
        wrongs=wrongs,
    )


def amgm_reciprocal(rng) -> dict:
    x = pick(rng, ("x", "t", "u"))
    c = pick(rng, (2, 3, 4, 5, 6))
    ref = [f"{x}+{c * c}/{x}-{2 * c}", f"({x}**2+{c * c}-{2 * c}*{x})/{x}", f"({x}-{c})**2/{x}"]
    return dict(
        key="amgm_reciprocal",
        lead=[
            T(
                f"For {x} > 0, prove {x} + {c * c}/{x} ≥ {2 * c}. First show that {x} + {c * c}/{x} - {2 * c} equals a square "
                f"divided by {x}, which is never negative."
            )
        ],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={x: "positive"},
        reasons=[f"write everything over {x}.", "the numerator is the square of a difference."],
        wrongs=[
            (
                "amgm-drops-denominator",
                [ref[0], f"{x}**2+{c * c}-{2 * c}*{x}", f"({x}-{c})**2"],
                "was the whole expression multiplied by x?",
            ),
            ("amgm-cross-term-sign", [ref[0], ref[1], f"({x}+{c})**2/{x}"], "is the square of the difference?"),
            (
                "amgm-cross-term-halved",
                [ref[0], f"({x}**2+{c * c}-{c}*{x})/{x}", f"({x}-{c})**2/{x}"],
                "is the middle term 2c·x over x, i.e. 2c?",
            ),
        ],
    )


def amgm_ratio(rng) -> dict:
    a, b = pick(rng, (("a", "b"), ("x", "y"), ("p", "q")))
    ref = [f"{a}/{b}+{b}/{a}-2", f"({a}**2+{b}**2-2*{a}*{b})/({a}*{b})", f"({a}-{b})**2/({a}*{b})"]
    return dict(
        key="amgm_ratio",
        lead=[
            T(
                f"For positive {a} and {b}, prove {a}/{b} + {b}/{a} ≥ 2. First show that {a}/{b} + {b}/{a} - 2 equals a "
                f"square divided by {a}{b}."
            )
        ],
        note=NOTE_PLAIN,
        ref=ref,
        symbols={a: "positive", b: "positive"},
        reasons=[f"use the common denominator {a}{b}.", "the numerator is the square of a difference."],
        wrongs=[
            (
                "amgm-drops-denominator",
                [ref[0], f"({a}**2+{b}**2-2)/({a}*{b})", ref[2]],
                "is the 2 over the common denominator too?",
            ),
            ("amgm-cross-term-sign", [ref[0], ref[1], f"({a}+{b})**2/({a}*{b})"], "is the square of the difference?"),
            (
                "amgm-drops-denominator",
                [ref[0], f"({a}**2+{b}**2-2*{a}*{b})/{a}", f"({a}-{b})**2/{a}"],
                "is the common denominator ab?",
            ),
        ],
    )


def amgm_roots(rng) -> dict:
    a, b = pick(rng, _AB)
    ref = [
        f"({a}+{b})/2-sqrt({a}*{b})",
        f"((sqrt({a}))**2+(sqrt({b}))**2-2*sqrt({a})*sqrt({b}))/2",
        f"(sqrt({a})-sqrt({b}))**2/2",
    ]
    return dict(
        key="amgm_roots",
        lead=[
            T(
                f"For positive {a} and {b}, prove the AM-GM inequality ({a} + {b})/2 ≥ √({a}{b}). First show that "
                f"({a} + {b})/2 - √({a}{b}) equals a square divided by 2."
            )
        ],
        note=NOTE_ROOT,
        ref=ref,
        symbols={a: "positive", b: "positive"},
        reasons=[
            f"write {a} as (√{a})^2 and {b} as (√{b})^2, and √({a}{b}) as √{a}√{b}.",
            "the numerator is the square of a difference.",
        ],
        wrongs=[
            (
                "amgm-cross-term-sign",
                [ref[0], ref[1], f"(sqrt({a})+sqrt({b}))**2/2"],
                "is the square of the difference?",
            ),
            (
                "amgm-cross-term-halved",
                [ref[0], f"((sqrt({a}))**2+(sqrt({b}))**2-sqrt({a})*sqrt({b}))/2", ref[2]],
                "is the cross term 2√a√b?",
            ),
            (
                "amgm-difference-of-squares",
                [ref[0], ref[1], f"(sqrt({a})-sqrt({b}))*(sqrt({a})+sqrt({b}))/2"],
                "is a square two equal brackets?",
            ),
        ],
    )


_AMGM = {
    "easy": (amgm_sum_squares,),
    "medium": (amgm_scaled,),
    "hard": (amgm_reciprocal, amgm_ratio, amgm_roots),
}


@generator("w10_proof_am_gm", level=12, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
def w10_proof_am_gm(rng, tier: str = "easy"):
    return build(rng, pick(rng, tier_families(_AMGM, tier))(rng))


@generator("w10_proof_am_gm_exam", level=12, presets={"exam": {}})
def w10_proof_am_gm_exam(rng):
    x, y = pick(rng, _XY3)
    c, d = rng.sample((1, 2, 3, 4, 5), 2)
    first = amgm_scaled(rng, x=x, y=y, c=c, d=d, form=rng.randrange(2))
    cx = f"{c}*{x}" if c > 1 else x
    dy = f"{d}*{y}" if d > 1 else y
    rb = [
        f"({c}*{x}+{d}*{y})/2-sqrt({c * d}*{x}*{y})",
        f"((sqrt({c}*{x}))**2+(sqrt({d}*{y}))**2-2*sqrt({c}*{x})*sqrt({d}*{y}))/2",
        f"(sqrt({c}*{x})-sqrt({d}*{y}))**2/2",
    ]
    cs = f"{c}{x}" if c > 1 else x
    ds = f"{d}{y}" if d > 1 else y
    second = dict(
        key="amgm_exam_b",
        lead=[
            T(
                f"Now let {x} and {y} be positive. Show that ({cs} + {ds})/2 - √({c * d}{x}{y}) equals a square divided by 2."
            )
        ],
        note=NOTE_ROOT,
        ref=rb,
        symbols={x: "positive", y: "positive"},
        reasons=[f"write {cs} as (√({cs}))^2 and {ds} as (√({ds}))^2.", "the numerator is the square of a difference."],
        wrongs=[
            (
                "amgm-cross-term-sign",
                [rb[0], rb[1], f"(sqrt({c}*{x})+sqrt({d}*{y}))**2/2"],
                "is it the square of the difference?",
            ),
            (
                "amgm-cross-term-halved",
                [rb[0], f"((sqrt({c}*{x}))**2+(sqrt({d}*{y}))**2-sqrt({c}*{x})*sqrt({d}*{y}))/2", rb[2]],
                "is the cross term twice the product of the roots?",
            ),
            (
                "amgm-difference-of-squares",
                [rb[0], rb[1], f"(sqrt({c}*{x})-sqrt({d}*{y}))*(sqrt({c}*{x})+sqrt({d}*{y}))/2"],
                "is it a square?",
            ),
        ],
    )
    return build_parts(
        rng,
        f"A proof that the mean of two positive numbers is at least their geometric mean uses a difference that is a square.",
        r"\text{A proof that the mean of two positive numbers is at least their geometric mean uses a difference "
        r"that is a square.}",
        [first, second],
        [2, 3],
    )


# ---------------------------------------------------------------------------------------------------------------
# The triangle inequality
# ---------------------------------------------------------------------------------------------------------------
_AB_TRI = (("a", "b"), ("x", "y"), ("u", "v"), ("m", "n"))


def tri_square_of_sum(rng) -> dict:
    a, b = pick(rng, _AB_TRI)
    c, d = pick(rng, (1, 2, 3)), pick(rng, (1, 2, 3))
    ca = f"{c}*abs({a})" if c > 1 else f"abs({a})"
    db = f"{d}*abs({b})" if d > 1 else f"abs({b})"
    shown_a = f"{c}|{a}|" if c > 1 else f"|{a}|"
    shown_b = f"{d}|{b}|" if d > 1 else f"|{b}|"
    cross = 2 * c * d
    mid = f"{c * c}*abs({a})**2+{cross}*abs({a})*abs({b})+{d * d}*abs({b})**2"
    ref = [f"({ca}+{db})**2", mid, f"{c * c}*{a}**2+{cross}*abs({a})*abs({b})+{d * d}*{b}**2"]
    ref = [r.replace("1*", "") if r.startswith("1*") else r for r in ref]
    ref[1] = ref[1].replace(f"1*abs({a})**2", f"abs({a})**2").replace(f"1*abs({b})**2", f"abs({b})**2")
    ref[2] = ref[2].replace(f"1*{a}**2", f"{a}**2").replace(f"1*{b}**2", f"{b}**2")
    cross_bad = f"{c * c}*{a}**2+{cross}*{a}*{b}+{d * d}*{b}**2"
    return dict(
        key="tri_square_of_sum",
        lead=[
            T(
                f"To prove a triangle inequality by squaring, we need ({shown_a} + {shown_b})^2 in terms of {a} and "
                f"{b}. Expand it, keeping the middle term as an absolute value, and use |{a}|^2 = {a}^2 and "
                f"|{b}|^2 = {b}^2."
            )
        ],
        note=NOTE_ABS,
        ref=ref,
        symbols={a: "real", b: "real"},
        reasons=[
            "expand the square, keeping the middle term.",
            "the square of an absolute value is the square of the number.",
        ],
        wrongs=[
            (
                "tri-abs-squared-not-squared",
                [ref[0], ref[1], f"{c * c}*abs({a})+{cross}*abs({a})*abs({b})+{d * d}*abs({b})"],
                "is |a|^2 equal to a^2, not |a|?",
            ),
            (
                "tri-cross-term-dropped",
                [ref[0], f"{c * c}*abs({a})**2+{d * d}*abs({b})**2", f"{c * c}*{a}**2+{d * d}*{b}**2"],
                "does squaring a sum keep the middle term?",
            ),
            (
                "tri-abs-squared-not-squared",
                [ref[0], ref[1], cross_bad],
                "is the middle term kept as an absolute value?",
            ),
        ],
    )


def tri_difference(rng, a=None, b=None, c=None) -> dict:
    if a is None:
        a, b = pick(rng, _AB_TRI)
    if c is None:
        c = pick(rng, (1, 2, 3, 4))
    ca = f"{c}*abs({a})" if c > 1 else f"abs({a})"
    cb = f"{c}*{a}" if c > 1 else a
    ref = [
        f"({ca}+abs({b}))**2-({cb}+{b})**2",
        f"{c * c}*{a}**2+{2 * c}*abs({a})*abs({b})+{b}**2-({cb}+{b})**2",
        f"{2 * c}*(abs({a})*abs({b})-{a}*{b})",
    ]
    ref = [r.replace("1*", "") for r in ref]
    wrong_end = f"{2 * c}*(abs({a})*abs({b})+{a}*{b})"
    return dict(
        key="tri_difference",
        lead=[
            T(
                f"To prove |{cb.replace('*', '') if c > 1 else a} + {b}| ≤ {c if c > 1 else ''}|{a}| + |{b}| by squaring, show that "
                f"the difference of the squares equals {2 * c}(|{a}||{b}| - {a}{b}), which is never negative."
            )
        ],
        note=NOTE_ABS,
        ref=ref,
        symbols={a: "real", b: "real"},
        reasons=[
            "expand the first square; use |a|^2 = a^2 and expand the second square.",
            "the squared terms cancel; factor what is left.",
        ],
        wrongs=[
            ("tri-cross-term-sign", [ref[0], ref[1], wrong_end], "is the second cross term subtracted?"),
            (
                "tri-abs-squared-not-squared",
                [ref[0], f"{c * c}*abs({a})+{2 * c}*abs({a})*abs({b})+abs({b})-({cb}+{b})**2", ref[2]],
                "is |a|^2 equal to a^2, not |a|?",
            ),
            ("tri-cross-term-dropped", [ref[0], ref[1], f"{2 * c}*abs({a})*abs({b})"], "does the term -2ab remain?"),
        ],
    )


def tri_three_points(rng, form=None, letters=None, c=None) -> dict:
    if form is None:
        form = rng.randrange(2)
    if form:
        x, y, z = letters or pick(rng, (("x", "y", "z"), ("a", "b", "c"), ("p", "q", "r"), ("u", "v", "w")))
        symbols = {x: "real", y: "real", z: "real"}
        end_a, end_b, via = x, z, y
        u_, v_ = f"({x}-{y})", f"({y}-{z})"
        au, av = f"abs({x}-{y})", f"abs({y}-{z})"
        shown = dict(u=f"{x} - {y}", v=f"{y} - {z}", w=f"{x} - {z}")
    else:
        x, y = letters or pick(rng, (("x", "y"), ("a", "b"), ("p", "q")))
        if c is None:
            c = pick(rng, (1, 2, 3, 4, 5, 6))
        symbols = {x: "real", y: "real"}
        end_a, end_b, via = x, y, str(c)
        u_, v_ = f"({x}-{c})", f"({c}-{y})"
        au, av = f"abs({x}-{c})", f"abs({c}-{y})"
        shown = dict(u=f"{x} - {c}", v=f"{c} - {y}", w=f"{x} - {y}")
    w_ = f"({end_a}-{end_b})"
    ref = [
        f"({au}+{av})**2-{w_}**2",
        f"{u_}**2+2*{au}*{av}+{v_}**2-{w_}**2",
        f"2*({au}*{av}-{u_}*{v_})",
    ]
    return dict(
        key="tri_three_points",
        lead=[
            T(
                f"To prove the distance inequality |{shown['w']}| ≤ |{shown['u']}| + |{shown['v']}| by squaring, show that "
                f"(|{shown['u']}| + |{shown['v']}|)^2 - ({shown['w']})^2 equals 2(|{shown['u']}||{shown['v']}| - "
                f"({shown['u']})({shown['v']})), which is never negative. Use that {shown['w']} = ({shown['u']}) + ({shown['v']})."
            )
        ],
        note=NOTE_ABS,
        ref=ref,
        symbols=symbols,
        reasons=[
            "expand the first square and use |t|^2 = t^2.",
            "expand the second square as the square of a sum; the squares cancel.",
        ],
        wrongs=[
            (
                "tri-cross-term-sign",
                [ref[0], ref[1], f"2*({au}*{av}+{u_}*{v_})"],
                "is the second cross term subtracted?",
            ),
            (
                "tri-abs-squared-not-squared",
                [ref[0], f"{au}+2*{au}*{av}+{av}-{w_}**2", ref[2]],
                "is |t|^2 equal to t^2?",
            ),
            ("tri-cross-term-dropped", [ref[0], ref[1], f"{au}*{av}-{u_}*{v_}"], "is the factor 2 kept?"),
        ],
    )


_TRI = {"easy": (tri_square_of_sum,), "medium": (tri_difference,), "hard": (tri_three_points,)}


@generator("w10_proof_triangle", level=12, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
def w10_proof_triangle(rng, tier: str = "easy"):
    return build(rng, pick(rng, tier_families(_TRI, tier))(rng))


@generator("w10_proof_triangle_exam", level=12, presets={"exam": {}})
def w10_proof_triangle_exam(rng):
    a, b = pick(rng, (("x", "y"), ("u", "v"), ("m", "n")))
    first = tri_difference(rng, a=a, b=b, c=pick(rng, (1, 2, 3, 4)))
    if rng.randrange(5) == 0:
        second = tri_three_points(rng, form=1, letters=pick(rng, (("p", "q", "r"), ("s", "t", "w"))))
    else:
        second = tri_three_points(
            rng, form=0, letters=pick(rng, (("p", "q"), ("s", "t"), ("r", "w"))), c=pick(rng, (1, 2, 3, 4, 5, 6))
        )
    return build_parts(
        rng,
        "The triangle inequality for numbers is proved by squaring both sides and showing the difference is not negative.",
        r"\text{The triangle inequality for numbers is proved by squaring both sides and showing the difference is "
        r"not negative.}",
        [first, second],
        [2, 3],
    )
