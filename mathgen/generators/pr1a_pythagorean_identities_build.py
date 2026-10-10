"""Generates `pr1a_pythagorean_identity_proof` items: prove a consequence of sin^2(a) + cos^2(a) = 1, assessed
with a rubric.

Topic ``pr1a_pythagorean_identity_proof`` (rubric-assessed, ``level=14``).
Two families of item:

- PROVE: derive one Pythagorean consequence from ``sin^2(a) + cos^2(a) = 1``
  (easy: ``1 + tan^2(a) = sec^2(a)`` / ``1 + cot^2(a) = csc^2(a)``; medium/hard:
  one-step and multi-step consequences; exam: an identity needing two
  different Pythagorean forms).
- APPLY: an exact value from one stated special-angle value
  (easy: one identity step; exam: a two-identity route, e.g. ``cot = 1/tan``
  then ``1 + cot^2 = csc^2``).

Every reference chain below is an identity in its own stated symbols (never
the right-triangle a^2 + b^2 = c^2 CONSTRAINT form -- that is not
chainable in free a, b, c). Every step is confirmed
CAS-confirmable by direct sweep; no step needed
splitting for `symbolically_equal` to accept it.

The generator never runs a full CAS chain check. It verifies each
reference chain cheaply, with the Python `math` module only, at three fixed-but-irregular real points (prove
items) or once at the concrete stated value (apply items).
"""

from __future__ import annotations

import math
import re

from ..core import check, generator, phrase


#: `\text{...}` escaping for the one special character these strings ever
#: carry (`^`, this module's typeable power notation) -- `\textasciicircum{}`
#: is the KaTeX-verified pattern (`\text{use \textasciicircum{} for powers}`
#: renders; a bare `^` inside `\text{}` does not).
def _text_verbatim(s: str) -> str:
    return r"\text{" + s.replace("^", r"\textasciicircum{}") + "}"


#: Real-LaTeX conversion for an `lhs_disp`/`rhs_disp` string (this module's
#: own typeable-ASCII display forms: `sin(x)`, `1 + tan^2(x)`, `sec(x) csc(x)`,
#: never `**`/`*`) -- backslash-prefix every trig function name; `^` already
#: reads as a valid KaTeX superscript operator in maths mode (`tan^2(x)`),
#: so nothing else needs converting.
_TRIG_NAME_RE = re.compile(r"\b(sin|cos|tan|sec|csc|cot)(?=[\^(])")


def _to_latex_math(s: str) -> str:
    return _TRIG_NAME_RE.sub(lambda m: "\\" + m.group(1), s)


# ---------------------------------------------------------------------------
# Symbol pool (safe in both sympy.sympify and mathgen.grading._parse_expression)
# ---------------------------------------------------------------------------
ARGS: tuple[tuple[str, str], ...] = (("x", "x"), ("t", "t"), ("u", "u"))

_PROBE_POINTS = (0.37, 1.19, 2.53)  # irregular reals; none near a multiple of pi/2

_TRIG_NS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sec": lambda v: 1.0 / math.cos(v),
    "csc": lambda v: 1.0 / math.sin(v),
    "cot": lambda v: 1.0 / math.tan(v),
}
_NUMERIC_NS = dict(_TRIG_NS, pi=math.pi, sqrt=math.sqrt)


def _eval_symbolic(line: str, arg_sym: str, value: float) -> float:
    ns = dict(_TRIG_NS)
    ns[arg_sym] = value
    return eval(line, {"__builtins__": {}}, ns)  # noqa: S307 -- fixed trusted templates only


def _eval_numeric(line: str) -> float:
    return eval(line, {"__builtins__": {}}, _NUMERIC_NS)  # noqa: S307


def _verify_chain_numerically(ref_lines: list[str], arg_sym: str, msg: str) -> None:
    """The cheap in-generator check: every consecutive pair of reference
    lines agrees at three fixed-but-irregular real points, evaluated with the
    `math` module only (never sympy/CAS)."""
    for point in _PROBE_POINTS:
        values = [_eval_symbolic(line, arg_sym, point) for line in ref_lines]
        for i in range(1, len(values)):
            check(math.isclose(values[i], values[i - 1], rel_tol=1e-6, abs_tol=1e-9), msg)


def _verify_numeric_chain(ref_lines: list[str], msg: str) -> None:
    values = [_eval_numeric(line) for line in ref_lines]
    for i in range(1, len(values)):
        check(math.isclose(values[i], values[i - 1], rel_tol=1e-9, abs_tol=1e-12), msg)


def _flip_sign(expr: str) -> str:
    """Flip the one +/- in a hardcoded surd string (e.g. "8 - 4*sqrt(3)" ->
    "8 + 4*sqrt(3)") -- always a genuinely different value for every target
    this module builds (each is `a +/- b*sqrt(c)` or `(a +/- b)/d` with
    `b, c != 0`), used only to build the `wrong_stated_value` trap."""
    if " - " in expr:
        return expr.replace(" - ", " + ", 1)
    return expr.replace(" + ", " - ", 1)


def _fmt_deg(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


# ---------------------------------------------------------------------------
# PROVE families -- one entry per identity. `{a}` is the argument placeholder,
# filled with the sympy-safe symbol text for `*_sym` fields and the display
# text for `*_disp` fields. `ref_sym[0]`/`ref_sym[-1]` MUST equal
# `lhs_sym`/`rhs_sym` verbatim (checked at build time).
# ---------------------------------------------------------------------------
FAMILIES: tuple[dict, ...] = (
    dict(
        key="tan_sec",
        tier="easy",
        lhs_sym="1 + tan({a})**2",
        rhs_sym="sec({a})**2",
        ref_sym=(
            "1 + tan({a})**2",
            "1 + sin({a})**2/cos({a})**2",
            "(cos({a})**2 + sin({a})**2)/cos({a})**2",
            "1/cos({a})**2",
            "sec({a})**2",
        ),
        lhs_disp="1 + tan^2({a})",
        rhs_disp="sec^2({a})",
        drop_sym="1 + tan({a})",
        wrong_sym="1 - tan({a})**2",
        given_disp="sin^2({a}) + cos^2({a}) = 1",
        domain_disp="cos({a}) is not 0",
    ),
    dict(
        key="cot_csc",
        tier="easy",
        lhs_sym="1 + cot({a})**2",
        rhs_sym="csc({a})**2",
        ref_sym=(
            "1 + cot({a})**2",
            "1 + cos({a})**2/sin({a})**2",
            "(sin({a})**2 + cos({a})**2)/sin({a})**2",
            "1/sin({a})**2",
            "csc({a})**2",
        ),
        lhs_disp="1 + cot^2({a})",
        rhs_disp="csc^2({a})",
        drop_sym="1 + cot({a})",
        wrong_sym="1 - cot({a})**2",
        given_disp="sin^2({a}) + cos^2({a}) = 1",
        domain_disp="sin({a}) is not 0",
    ),
    dict(
        key="sec_tan_diff",
        tier="medium",
        lhs_sym="sec({a})**2 - tan({a})**2",
        rhs_sym="1",
        ref_sym=(
            "sec({a})**2 - tan({a})**2",
            "1/cos({a})**2 - sin({a})**2/cos({a})**2",
            "(1 - sin({a})**2)/cos({a})**2",
            "cos({a})**2/cos({a})**2",
            "1",
        ),
        lhs_disp="sec^2({a}) - tan^2({a})",
        rhs_disp="1",
        drop_sym="sec({a}) - tan({a})**2",
        wrong_sym="sec({a})**2 + tan({a})**2",
        given_disp="tan({a}) = sin({a})/cos({a}), sec({a}) = 1/cos({a}), and sin^2({a}) + cos^2({a}) = 1",
        domain_disp="cos({a}) is not 0",
    ),
    dict(
        key="conj_prod",
        tier="medium",
        lhs_sym="(1 + sin({a}))*(1 - sin({a}))",
        rhs_sym="cos({a})**2",
        ref_sym=("(1 + sin({a}))*(1 - sin({a}))", "1 - sin({a})**2", "cos({a})**2"),
        lhs_disp="(1 + sin({a}))(1 - sin({a}))",
        rhs_disp="cos^2({a})",
        drop_sym="1 - sin({a})",
        wrong_sym="1 + sin({a})**2",
        given_disp="sin^2({a}) + cos^2({a}) = 1",
    ),
    dict(
        key="quartic_diff",
        tier="medium",
        lhs_sym="sin({a})**4 - cos({a})**4",
        rhs_sym="sin({a})**2 - cos({a})**2",
        ref_sym=(
            "sin({a})**4 - cos({a})**4",
            "(sin({a})**2 - cos({a})**2)*(sin({a})**2 + cos({a})**2)",
            "sin({a})**2 - cos({a})**2",
        ),
        lhs_disp="sin^4({a}) - cos^4({a})",
        rhs_disp="sin^2({a}) - cos^2({a})",
        drop_sym="(sin({a}) - cos({a})**2)*(sin({a})**2 + cos({a})**2)",
        wrong_sym="(sin({a})**2 + cos({a})**2)*(sin({a})**2 + cos({a})**2)",
        given_disp="sin^2({a}) + cos^2({a}) = 1 and a^2 - b^2 = (a - b)(a + b)",
    ),
    dict(
        key="fraction_sum",
        tier="hard",
        lhs_sym="1/(1 - sin({a})) + 1/(1 + sin({a}))",
        rhs_sym="2*sec({a})**2",
        ref_sym=(
            "1/(1 - sin({a})) + 1/(1 + sin({a}))",
            "((1 + sin({a})) + (1 - sin({a})))/((1 - sin({a}))*(1 + sin({a})))",
            "2/(1 - sin({a})**2)",
            "2/cos({a})**2",
            "2*sec({a})**2",
        ),
        lhs_disp="1/(1 - sin({a})) + 1/(1 + sin({a}))",
        rhs_disp="2 sec^2({a})",
        drop_sym="2/(1 - sin({a}))",
        wrong_sym="2/(1 + sin({a})**2)",
        given_disp="sec({a}) = 1/cos({a}) and sin^2({a}) + cos^2({a}) = 1",
        domain_disp="cos({a}) is not 0",
    ),
    dict(
        key="tan_cot_prod",
        tier="hard",
        lhs_sym="tan({a}) + cot({a})",
        rhs_sym="sec({a})*csc({a})",
        ref_sym=(
            "tan({a}) + cot({a})",
            "sin({a})/cos({a}) + cos({a})/sin({a})",
            "(sin({a})**2 + cos({a})**2)/(sin({a})*cos({a}))",
            "1/(sin({a})*cos({a}))",
            "sec({a})*csc({a})",
        ),
        lhs_disp="tan({a}) + cot({a})",
        rhs_disp="sec({a}) csc({a})",
        drop_sym="(sin({a}) + cos({a})**2)/(sin({a})*cos({a}))",
        wrong_sym="(sin({a})**2 - cos({a})**2)/(sin({a})*cos({a}))",
        given_disp=(
            "tan({a}) = sin({a})/cos({a}), cot({a}) = cos({a})/sin({a}), "
            "sec({a}) = 1/cos({a}), csc({a}) = 1/sin({a}), and sin^2({a}) + cos^2({a}) = 1"
        ),
        domain_disp="sin({a}) is not 0 and cos({a}) is not 0",
    ),
    dict(
        key="conjugate_fraction",
        tier="hard",
        lhs_sym="cos({a})/(1 - sin({a}))",
        rhs_sym="(1 + sin({a}))/cos({a})",
        ref_sym=(
            "cos({a})/(1 - sin({a}))",
            "cos({a})*(1 + sin({a}))/((1 - sin({a}))*(1 + sin({a})))",
            "cos({a})*(1 + sin({a}))/(1 - sin({a})**2)",
            "cos({a})*(1 + sin({a}))/cos({a})**2",
            "(1 + sin({a}))/cos({a})",
        ),
        lhs_disp="cos({a})/(1 - sin({a}))",
        rhs_disp="(1 + sin({a}))/cos({a})",
        drop_sym="cos({a})*(1 + sin({a}))/(1 - sin({a}))",
        wrong_sym="cos({a})*(1 + sin({a}))/(1 + sin({a})**2)",
        given_disp=("sin^2({a}) + cos^2({a}) = 1 (multiply top and bottom by the conjugate 1 + sin({a}))"),
        domain_disp="cos({a}) is not 0",
    ),
    dict(
        key="sec_csc_prod",
        tier="hard",
        lhs_sym="sec({a})**2 + csc({a})**2",
        rhs_sym="sec({a})**2*csc({a})**2",
        ref_sym=(
            "sec({a})**2 + csc({a})**2",
            "1/cos({a})**2 + 1/sin({a})**2",
            "(sin({a})**2 + cos({a})**2)/(sin({a})**2*cos({a})**2)",
            "1/(sin({a})**2*cos({a})**2)",
            "sec({a})**2*csc({a})**2",
        ),
        lhs_disp="sec^2({a}) + csc^2({a})",
        rhs_disp="sec^2({a}) csc^2({a})",
        drop_sym="(sin({a}) + cos({a})**2)/(sin({a})**2*cos({a})**2)",
        wrong_sym="(sin({a})**2 - cos({a})**2)/(sin({a})**2*cos({a})**2)",
        given_disp="sec({a}) = 1/cos({a}), csc({a}) = 1/sin({a}), and sin^2({a}) + cos^2({a}) = 1",
        domain_disp="sin({a}) is not 0 and cos({a}) is not 0",
    ),
    dict(
        key="exam_proof_a",
        tier="exam",
        lhs_sym="(sec({a}) - tan({a}))*(sec({a}) + tan({a}))",
        rhs_sym="1",
        ref_sym=(
            "(sec({a}) - tan({a}))*(sec({a}) + tan({a}))",
            "sec({a})**2 - tan({a})**2",
            "1/cos({a})**2 - sin({a})**2/cos({a})**2",
            "(1 - sin({a})**2)/cos({a})**2",
            "cos({a})**2/cos({a})**2",
            "1",
        ),
        lhs_disp="(sec({a}) - tan({a}))(sec({a}) + tan({a}))",
        rhs_disp="1",
        drop_sym="sec({a}) - tan({a})**2",
        wrong_sym="sec({a})**2 + tan({a})**2",
        given_disp=("sin^2({a}) + cos^2({a}) = 1, and the difference-of-squares expansion (a - b)(a + b) = a^2 - b^2"),
        domain_disp="cos({a}) is not 0",
    ),
    dict(
        key="exam_proof_b",
        tier="exam",
        lhs_sym="(1 + tan({a})**2)*(1 - sin({a})**2)",
        rhs_sym="1",
        ref_sym=(
            "(1 + tan({a})**2)*(1 - sin({a})**2)",
            "sec({a})**2*cos({a})**2",
            "(1/cos({a})**2)*cos({a})**2",
            "1",
        ),
        lhs_disp="(1 + tan^2({a}))(1 - sin^2({a}))",
        rhs_disp="1",
        drop_sym="sec({a})*cos({a})**2",
        wrong_sym="csc({a})**2*cos({a})**2",
        given_disp="1 + tan^2({a}) = sec^2({a}) and sin^2({a}) + cos^2({a}) = 1",
        domain_disp="cos({a}) is not 0",
    ),
)

FAMILIES_BY_TIER: dict[str, tuple[dict, ...]] = {
    tier: tuple(f for f in FAMILIES if f["tier"] == tier) for tier in ("easy", "medium", "hard", "exam")
}

PROVE_TEMPLATES = (
    "Prove that {lhs} = {rhs}. You may cite {given} as a known result.{domain} "
    "Start your proof from {lhs} and finish at {rhs}, writing one equivalent expression per line. "
    "Put brackets round every angle, as in sin(x), and use ^ for powers, as in cos(x)^2.",
    "Show that the identity {lhs} = {rhs} holds, given that {given}.{domain} "
    "Begin your proof from {lhs} and end at {rhs}, one typed expression per line -- "
    "brackets round every angle, e.g. sin(x), and ^ for powers, e.g. cos(x)^2.",
    "Using the fact that {given}, prove that {lhs} = {rhs}.{domain} "
    "Write your proof starting from {lhs}, finishing at {rhs}, one expression per line, "
    "using brackets round every angle (sin(x)) and ^ for powers (cos(x)^2).",
)

#: `latex` twin of `PROVE_TEMPLATES` above -- SAME wording, SAME order (index
#: for index), one function per entry rather than a `.format()` string so
#: the `\text{{...}}` double-brace bookkeeping never has to fight this
#: module's own `{a}`-style slots. `lhs`/`rhs` render as real LaTeX (`\tan^2
#: x`-style backslash-prefixed function names via `_to_latex_math`); the
#: quoted "start from .../finish at..." clause, the given, the domain
#: clause and the notation sentence are the exact typeable string wrapped
#: in escaped `\text{...}` -- never a second, differently-worded
#: rendering, and never a bare `^` or trig-function letters sitting
#: unescaped in maths mode.
_NOTATION_TAIL = (
    "writing one equivalent expression per line. Put brackets round every angle, "
    "as in sin(x), and use ^ for powers, as in cos(x)^2."
)
_NOTATION_TAIL_2 = (
    "one typed expression per line -- brackets round every angle, e.g. sin(x), and ^ for powers, e.g. cos(x)^2."
)
_NOTATION_TAIL_3 = "one expression per line, using brackets round every angle (sin(x)) and ^ for powers (cos(x)^2)."


def _prove_latex_0(lhs_disp: str, rhs_disp: str, given_disp: str, domain_clause: str) -> str:
    lhs_m, rhs_m = _to_latex_math(lhs_disp), _to_latex_math(rhs_disp)
    domain_part = _text_verbatim(domain_clause) if domain_clause else ""
    return (
        _text_verbatim("Prove that ")
        + f"{lhs_m} = {rhs_m}"
        + _text_verbatim(". You may cite ")
        + _text_verbatim(given_disp)
        + _text_verbatim(" as a known result.")
        + domain_part
        + _text_verbatim(" Start your proof from ")
        + _text_verbatim(lhs_disp)
        + _text_verbatim(" and finish at ")
        + _text_verbatim(rhs_disp)
        + _text_verbatim(", " + _NOTATION_TAIL)
    )


def _prove_latex_1(lhs_disp: str, rhs_disp: str, given_disp: str, domain_clause: str) -> str:
    lhs_m, rhs_m = _to_latex_math(lhs_disp), _to_latex_math(rhs_disp)
    domain_part = _text_verbatim(domain_clause) if domain_clause else ""
    return (
        _text_verbatim("Show that the identity ")
        + f"{lhs_m} = {rhs_m}"
        + _text_verbatim(" holds, given that ")
        + _text_verbatim(given_disp)
        + _text_verbatim(".")
        + domain_part
        + _text_verbatim(" Begin your proof from ")
        + _text_verbatim(lhs_disp)
        + _text_verbatim(" and end at ")
        + _text_verbatim(rhs_disp)
        + _text_verbatim(", " + _NOTATION_TAIL_2)
    )


def _prove_latex_2(lhs_disp: str, rhs_disp: str, given_disp: str, domain_clause: str) -> str:
    lhs_m, rhs_m = _to_latex_math(lhs_disp), _to_latex_math(rhs_disp)
    domain_part = _text_verbatim(domain_clause) if domain_clause else ""
    return (
        _text_verbatim("Using the fact that ")
        + _text_verbatim(given_disp)
        + _text_verbatim(", prove that ")
        + f"{lhs_m} = {rhs_m}"
        + _text_verbatim(".")
        + domain_part
        + _text_verbatim(" Write your proof starting from ")
        + _text_verbatim(lhs_disp)
        + _text_verbatim(", finishing at ")
        + _text_verbatim(rhs_disp)
        + _text_verbatim(", " + _NOTATION_TAIL_3)
    )


PROVE_LATEX_TEMPLATES = (_prove_latex_0, _prove_latex_1, _prove_latex_2)


def _prove_traps(family: dict, arg_sym: str, arg_disp: str) -> list[dict]:
    lhs = family["lhs_sym"].format(a=arg_sym)
    rhs = family["rhs_sym"].format(a=arg_sym)
    ref = [line.format(a=arg_sym) for line in family["ref_sym"]]
    drop_line = family["drop_sym"].format(a=arg_sym)
    wrong_line = family["wrong_sym"].format(a=arg_sym)
    return [
        {
            "name": "drops_the_square",
            "question": "does every squared trig term stay squared all the way through the proof?",
            "example_lines": [lhs, drop_line, rhs],
            "expected_check": "authoritative_failure",
        },
        {
            "name": "wrong_pythagorean_form",
            "question": "has the wrong Pythagorean identity, or a wrong sign inside it, been substituted?",
            "example_lines": [lhs, wrong_line, rhs],
            "expected_check": "authoritative_failure",
        },
        {
            "name": "starts_from_rhs",
            "question": "does the proof genuinely start from the stated left-hand side, not the target?",
            "example_lines": [rhs, lhs],
            "expected_check": "non_authoritative_failure",
        },
        {
            "name": "divides_by_zero_unstated",
            "question": "is the domain condition needed for the division (cos/sin nonzero) stated anywhere?",
            "example_lines": list(ref),
            "expected_check": "passes_chain",
        },
        {
            # A bare one-line jump [lhs, rhs] is not always CAS-confirmable
            # directly (measured: `tan(x) + cot(x) -> sec(x)*csc(x)` comes
            # back "could not confirm" even though the full stepwise chain
            # does confirm) -- keeping the first rewrite-by-definition step
            # (`ref[1]`) and skipping the rest is confirmable for every
            # family in this pool (checked by direct sweep result
            # file) and still represents "skips the shown working".
            "name": "cites_what_it_proves",
            "question": "does the response show the derivation, or just assert the identity being proved?",
            "example_lines": [lhs, ref[1], rhs] if len(ref) > 2 else [lhs, rhs],
            "expected_check": "passes_chain",
        },
    ]


def _prove_item(rng, family: dict) -> dict:
    arg_sym, arg_disp = rng.choice(ARGS)
    lhs_sym = family["lhs_sym"].format(a=arg_sym)
    rhs_sym = family["rhs_sym"].format(a=arg_sym)
    ref_sym = [line.format(a=arg_sym) for line in family["ref_sym"]]
    check(ref_sym[0] == lhs_sym, f"{family['key']}: reference chain does not start at the stated left-hand side")
    check(ref_sym[-1] == rhs_sym, f"{family['key']}: reference chain does not end at the stated right-hand side")
    _verify_chain_numerically(ref_sym, arg_sym, f"{family['key']}: reference chain does not hold numerically")

    lhs_disp = family["lhs_disp"].format(a=arg_disp)
    rhs_disp = family["rhs_disp"].format(a=arg_disp)
    given_disp = family["given_disp"].format(a=arg_disp)
    domain_tmpl = family.get("domain_disp", "")
    domain_clause = (
        f" This identity holds for every {arg_disp} where {domain_tmpl.format(a=arg_disp)}." if domain_tmpl else ""
    )

    # `text` and `latex` share ONE template choice: `rng.randrange`
    # draws exactly the one `_randbelow` call `phrase()`'s own `rng.choice`
    # would have made, so `text` stays byte-identical to before this fix,
    # and `PROVE_LATEX_TEMPLATES[idx]` never independently drifts to a
    # different wording than `text`.
    idx = rng.randrange(len(PROVE_TEMPLATES))
    text = PROVE_TEMPLATES[idx].format(lhs=lhs_disp, rhs=rhs_disp, given=given_disp, domain=domain_clause)
    latex = PROVE_LATEX_TEMPLATES[idx](lhs_disp, rhs_disp, given_disp, domain_clause)
    answer_text = f"{lhs_disp} = {rhs_disp}"

    key_steps = [
        {
            "criterion_id": "correctness",
            "description": f"transforms {lhs_disp} into {rhs_disp} using only valid algebraic steps",
        },
        {
            "criterion_id": "justified_steps",
            "description": f"cites {given_disp} as the given result used, rather than re-deriving it",
        },
        {
            "criterion_id": "communication",
            "description": "writes exactly one equivalent expression per line, ending at the stated target",
        },
    ]
    traps = _prove_traps(family, arg_sym, arg_disp)
    symbols = {arg_sym: "real"}
    checks = [
        {"type": "equivalence_chain", "start": lhs_sym, "target": rhs_sym, "symbols": symbols, "authoritative": True}
    ]
    answer = {
        "start": lhs_sym,
        "target": rhs_sym,
        "symbols": symbols,
        "reference_lines": ref_sym,
        "key_steps": key_steps,
        "traps": traps,
        "checks": checks,
    }
    return dict(text=text, latex=latex, answer=answer, answer_text=answer_text)


# ---------------------------------------------------------------------------
# APPLY families -- exact values from one (tan-only tier) or two-identity
# (exam tier) route, from a fixed pool of special-angle stated values.
# ---------------------------------------------------------------------------
TAN_ANGLES: tuple[dict, ...] = (
    dict(
        angle_sym="pi/12",
        angle_disp="π/12",
        degrees=15,
        stated_sym="2 - sqrt(3)",
        stated_disp="2 - √3",
        sec2_sym="8 - 4*sqrt(3)",
        sec2_disp="8 - 4√3",
        csc2_sym="4*sqrt(3) + 8",
        csc2_disp="4√3 + 8",
        cos2_sym="(sqrt(3) + 2)/4",
        cos2_disp="(√3 + 2)/4",
    ),
    dict(
        angle_sym="5*pi/12",
        angle_disp="5π/12",
        degrees=75,
        stated_sym="2 + sqrt(3)",
        stated_disp="2 + √3",
        sec2_sym="4*sqrt(3) + 8",
        sec2_disp="4√3 + 8",
        csc2_sym="8 - 4*sqrt(3)",
        csc2_disp="8 - 4√3",
        cos2_sym="(2 - sqrt(3))/4",
        cos2_disp="(2 - √3)/4",
    ),
    dict(
        angle_sym="pi/8",
        angle_disp="π/8",
        degrees=22.5,
        stated_sym="sqrt(2) - 1",
        stated_disp="√2 - 1",
        sec2_sym="4 - 2*sqrt(2)",
        sec2_disp="4 - 2√2",
        csc2_sym="2*sqrt(2) + 4",
        csc2_disp="2√2 + 4",
        cos2_sym="(sqrt(2) + 2)/4",
        cos2_disp="(√2 + 2)/4",
    ),
    dict(
        angle_sym="3*pi/8",
        angle_disp="3π/8",
        degrees=67.5,
        stated_sym="sqrt(2) + 1",
        stated_disp="√2 + 1",
        sec2_sym="2*sqrt(2) + 4",
        sec2_disp="2√2 + 4",
        csc2_sym="4 - 2*sqrt(2)",
        csc2_disp="4 - 2√2",
        cos2_sym="(2 - sqrt(2))/4",
        cos2_disp="(2 - √2)/4",
    ),
)

SC_ANGLES: tuple[dict, ...] = (
    dict(
        angle_sym="pi/10",
        angle_disp="π/10",
        degrees=18,
        given_kind="sin",
        stated_sym="(sqrt(5) - 1)/4",
        stated_disp="(√5 - 1)/4",
        target_sym="(5 + sqrt(5))/8",
        target_disp="(5 + √5)/8",
        target_name="cos^2",
    ),
    dict(
        angle_sym="pi/5",
        angle_disp="π/5",
        degrees=36,
        given_kind="cos",
        stated_sym="(1 + sqrt(5))/4",
        stated_disp="(1 + √5)/4",
        target_sym="(5 - sqrt(5))/8",
        target_disp="(5 - √5)/8",
        target_name="sin^2",
    ),
)

_NOTATION_SENTENCE = (
    "Write one expression per line; put brackets round every angle, as in sin(x), and use ^ for powers, as in cos(x)^2."
)

APPLY_TAN_TEMPLATES = (
    "Given that tan({angle_disp}) = {stated_disp} (where {angle_disp} = {degrees}°), use "
    "1 + tan^2({angle_disp}) = sec^2({angle_disp}) to find sec^2({angle_disp}) exactly. "
    "Start from sec^2({angle_disp}) and finish at the exact value. " + _NOTATION_SENTENCE,
    "It is known that tan({angle_disp}) = {stated_disp}. Using "
    "1 + tan^2({angle_disp}) = sec^2({angle_disp}), evaluate sec^2({angle_disp}) exactly, "
    "starting from sec^2({angle_disp}). " + _NOTATION_SENTENCE,
    "Using tan({angle_disp}) = {stated_disp} ({angle_disp} = {degrees}°) and "
    "1 + tan^2({angle_disp}) = sec^2({angle_disp}), find the exact value of sec^2({angle_disp}). " + _NOTATION_SENTENCE,
)

APPLY_SC_TEMPLATES = (
    "Given that {given_kind}({angle_disp}) = {stated_disp} (where {angle_disp} = {degrees}°), "
    "use sin^2({angle_disp}) + cos^2({angle_disp}) = 1 to find {target_name}({angle_disp}) exactly. "
    "Start from {target_name}({angle_disp}) and finish at the exact value. " + _NOTATION_SENTENCE,
    "It is known that {given_kind}({angle_disp}) = {stated_disp}. Using "
    "sin^2({angle_disp}) + cos^2({angle_disp}) = 1, evaluate {target_name}({angle_disp}) exactly. "
    + _NOTATION_SENTENCE,
    "Using {given_kind}({angle_disp}) = {stated_disp} ({angle_disp} = {degrees}°) and "
    "sin^2({angle_disp}) + cos^2({angle_disp}) = 1, find the exact value of {target_name}({angle_disp}). "
    + _NOTATION_SENTENCE,
)

APPLY_CSC_TEMPLATES = (
    "Given that tan({angle_disp}) = {stated_disp} (where {angle_disp} = {degrees}°), use "
    "cot({angle_disp}) = 1/tan({angle_disp}) and then 1 + cot^2({angle_disp}) = csc^2({angle_disp}) "
    "to find csc^2({angle_disp}) exactly. " + _NOTATION_SENTENCE,
    "It is known that tan({angle_disp}) = {stated_disp}. First write "
    "cot({angle_disp}) = 1/tan({angle_disp}), then use "
    "1 + cot^2({angle_disp}) = csc^2({angle_disp}) to evaluate csc^2({angle_disp}) exactly. " + _NOTATION_SENTENCE,
)

APPLY_COS2_TEMPLATES = (
    "Given that tan({angle_disp}) = {stated_disp} (where {angle_disp} = {degrees}°), use "
    "1 + tan^2({angle_disp}) = sec^2({angle_disp}) and then "
    "cos^2({angle_disp}) = 1/sec^2({angle_disp}) to find cos^2({angle_disp}) exactly. " + _NOTATION_SENTENCE,
    "It is known that tan({angle_disp}) = {stated_disp}. First write "
    "sec^2({angle_disp}) = 1 + tan^2({angle_disp}), then use "
    "cos^2({angle_disp}) = 1/sec^2({angle_disp}) to evaluate cos^2({angle_disp}) exactly. " + _NOTATION_SENTENCE,
)


def _apply_traps(start_sym: str, ref: list[str], target_sym: str) -> list[dict]:
    # No `starts_from_rhs` trap here: an apply item's `start`/`target` are both bare numeric
    # constants, and sympy auto-evaluates every special-angle trig constant in this pool to the SAME
    # canonical closed form on parse regardless of which side wrote it -- so a reversed [target,
    # start] chain is numerically indistinguishable from the correct order and trivially
    # `passes_chain` instead of failing. Two traps are enough to satisfy the minimum (one
    # non-authoritative, one passes_chain) for a numeric item.
    return [
        {
            "name": "wrong_stated_value",
            "question": "does the final line match the exact value asked for, not a sign-flipped one?",
            "example_lines": ref[:-1] + [_flip_sign(target_sym)],
            "expected_check": "non_authoritative_failure",
        },
        {
            "name": "cites_what_it_proves",
            "question": "does the response show the substitution, or just assert the final value?",
            "example_lines": [start_sym, target_sym],
            "expected_check": "passes_chain",
        },
    ]


def _tan_apply_item(rng) -> dict:
    ang = rng.choice(TAN_ANGLES)
    start_sym = f"sec({ang['angle_sym']})**2"
    ref = [start_sym, f"1 + tan({ang['angle_sym']})**2", f"1 + ({ang['stated_sym']})**2", ang["sec2_sym"]]
    _verify_numeric_chain(ref, "tan_apply: chain does not hold numerically")
    text = phrase(
        rng,
        APPLY_TAN_TEMPLATES,
        angle_disp=ang["angle_disp"],
        degrees=_fmt_deg(ang["degrees"]),
        stated_disp=ang["stated_disp"],
    )
    # An apply item's whole prompt is typeable/numeric prose (no separate
    # "identity to prove" clause worth real-math treatment), so `latex` is
    # the exact same string, escaped once for `\text{...}` (same
    # prompt as `text`, never a second independently-built rendering).
    latex = _text_verbatim(text)
    answer_text = f"sec^2({ang['angle_disp']}) = {ang['sec2_disp']}"
    key_steps = [
        {
            "criterion_id": "correctness",
            "description": f"substitutes tan({ang['angle_disp']}) = {ang['stated_disp']} into "
            f"1 + tan^2({ang['angle_disp']}) = sec^2({ang['angle_disp']}) and simplifies exactly",
        },
        {
            "criterion_id": "justified_steps",
            "description": f"cites 1 + tan^2({ang['angle_disp']}) = sec^2({ang['angle_disp']}) as the "
            f"identity used and tan({ang['angle_disp']}) = {ang['stated_disp']} as the given",
        },
        {
            "criterion_id": "communication",
            "description": "shows the substitution and the simplification as separate lines, ending at the exact value",
        },
    ]
    traps = _apply_traps(start_sym, ref, ang["sec2_sym"])
    checks = [
        {
            "type": "equivalence_chain",
            "start": start_sym,
            "target": ang["sec2_sym"],
            "symbols": {},
            "authoritative": True,
        }
    ]
    answer = {
        "start": start_sym,
        "target": ang["sec2_sym"],
        "symbols": {},
        "reference_lines": ref,
        "key_steps": key_steps,
        "traps": traps,
        "checks": checks,
    }
    return dict(text=text, latex=latex, answer=answer, answer_text=answer_text)


def _sc_apply_item(rng) -> dict:
    ang = rng.choice(SC_ANGLES)
    if ang["given_kind"] == "sin":
        start_sym = f"cos({ang['angle_sym']})**2"
        ref = [start_sym, f"1 - sin({ang['angle_sym']})**2", f"1 - ({ang['stated_sym']})**2", ang["target_sym"]]
    else:
        start_sym = f"sin({ang['angle_sym']})**2"
        ref = [start_sym, f"1 - cos({ang['angle_sym']})**2", f"1 - ({ang['stated_sym']})**2", ang["target_sym"]]
    _verify_numeric_chain(ref, "sc_apply: chain does not hold numerically")
    text = phrase(
        rng,
        APPLY_SC_TEMPLATES,
        given_kind=ang["given_kind"],
        angle_disp=ang["angle_disp"],
        degrees=_fmt_deg(ang["degrees"]),
        stated_disp=ang["stated_disp"],
        target_name=ang["target_name"],
    )
    latex = _text_verbatim(text)
    answer_text = f"{ang['target_name']}({ang['angle_disp']}) = {ang['target_disp']}"
    key_steps = [
        {
            "criterion_id": "correctness",
            "description": f"substitutes {ang['given_kind']}({ang['angle_disp']}) = {ang['stated_disp']} into "
            f"sin^2({ang['angle_disp']}) + cos^2({ang['angle_disp']}) = 1 and simplifies exactly",
        },
        {
            "criterion_id": "justified_steps",
            "description": f"cites sin^2({ang['angle_disp']}) + cos^2({ang['angle_disp']}) = 1 as the identity "
            f"used and {ang['given_kind']}({ang['angle_disp']}) = {ang['stated_disp']} as the given",
        },
        {
            "criterion_id": "communication",
            "description": "shows the substitution and the simplification as separate lines, ending at the exact value",
        },
    ]
    traps = _apply_traps(start_sym, ref, ang["target_sym"])
    checks = [
        {
            "type": "equivalence_chain",
            "start": start_sym,
            "target": ang["target_sym"],
            "symbols": {},
            "authoritative": True,
        }
    ]
    answer = {
        "start": start_sym,
        "target": ang["target_sym"],
        "symbols": {},
        "reference_lines": ref,
        "key_steps": key_steps,
        "traps": traps,
        "checks": checks,
    }
    return dict(text=text, latex=latex, answer=answer, answer_text=answer_text)


def _two_step_csc_item(rng) -> dict:
    ang = rng.choice(TAN_ANGLES)
    start_sym = f"csc({ang['angle_sym']})**2"
    ref = [
        start_sym,
        f"1 + cot({ang['angle_sym']})**2",
        f"1 + (1/tan({ang['angle_sym']}))**2",
        f"1 + (1/({ang['stated_sym']}))**2",
        ang["csc2_sym"],
    ]
    _verify_numeric_chain(ref, "two_step_csc: chain does not hold numerically")
    text = phrase(
        rng,
        APPLY_CSC_TEMPLATES,
        angle_disp=ang["angle_disp"],
        degrees=_fmt_deg(ang["degrees"]),
        stated_disp=ang["stated_disp"],
    )
    latex = _text_verbatim(text)
    answer_text = f"csc^2({ang['angle_disp']}) = {ang['csc2_disp']}"
    key_steps = [
        {
            "criterion_id": "correctness",
            "description": f"derives cot({ang['angle_disp']}) = 1/tan({ang['angle_disp']}) from "
            f"tan({ang['angle_disp']}) = {ang['stated_disp']}, then applies "
            f"1 + cot^2({ang['angle_disp']}) = csc^2({ang['angle_disp']}) and simplifies exactly",
        },
        {
            "criterion_id": "justified_steps",
            "description": f"cites both cot({ang['angle_disp']}) = 1/tan({ang['angle_disp']}) and "
            f"1 + cot^2({ang['angle_disp']}) = csc^2({ang['angle_disp']}) as the identities used",
        },
        {
            "criterion_id": "communication",
            "description": "shows both substitutions as separate lines, ending at the exact value",
        },
    ]
    traps = _apply_traps(start_sym, ref, ang["csc2_sym"])
    checks = [
        {
            "type": "equivalence_chain",
            "start": start_sym,
            "target": ang["csc2_sym"],
            "symbols": {},
            "authoritative": True,
        }
    ]
    answer = {
        "start": start_sym,
        "target": ang["csc2_sym"],
        "symbols": {},
        "reference_lines": ref,
        "key_steps": key_steps,
        "traps": traps,
        "checks": checks,
    }
    return dict(text=text, latex=latex, answer=answer, answer_text=answer_text)


def _two_step_cos2_item(rng) -> dict:
    ang = rng.choice(TAN_ANGLES)
    start_sym = f"cos({ang['angle_sym']})**2"
    ref = [
        start_sym,
        f"1/sec({ang['angle_sym']})**2",
        f"1/(1 + tan({ang['angle_sym']})**2)",
        f"1/(1 + ({ang['stated_sym']})**2)",
        ang["cos2_sym"],
    ]
    _verify_numeric_chain(ref, "two_step_cos2: chain does not hold numerically")
    text = phrase(
        rng,
        APPLY_COS2_TEMPLATES,
        angle_disp=ang["angle_disp"],
        degrees=_fmt_deg(ang["degrees"]),
        stated_disp=ang["stated_disp"],
    )
    latex = _text_verbatim(text)
    answer_text = f"cos^2({ang['angle_disp']}) = {ang['cos2_disp']}"
    key_steps = [
        {
            "criterion_id": "correctness",
            "description": f"derives sec^2({ang['angle_disp']}) = 1 + tan^2({ang['angle_disp']}) from "
            f"tan({ang['angle_disp']}) = {ang['stated_disp']}, then applies "
            f"cos^2({ang['angle_disp']}) = 1/sec^2({ang['angle_disp']}) and simplifies exactly",
        },
        {
            "criterion_id": "justified_steps",
            "description": f"cites both 1 + tan^2({ang['angle_disp']}) = sec^2({ang['angle_disp']}) and "
            f"cos^2({ang['angle_disp']}) = 1/sec^2({ang['angle_disp']}) as the identities used",
        },
        {
            "criterion_id": "communication",
            "description": "shows both substitutions as separate lines, ending at the exact value",
        },
    ]
    traps = _apply_traps(start_sym, ref, ang["cos2_sym"])
    checks = [
        {
            "type": "equivalence_chain",
            "start": start_sym,
            "target": ang["cos2_sym"],
            "symbols": {},
            "authoritative": True,
        }
    ]
    answer = {
        "start": start_sym,
        "target": ang["cos2_sym"],
        "symbols": {},
        "reference_lines": ref,
        "key_steps": key_steps,
        "traps": traps,
        "checks": checks,
    }
    return dict(text=text, latex=latex, answer=answer, answer_text=answer_text)


@generator(
    "pr1a_pythagorean_identity_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def pr1a_pythagorean_identity_proof(rng, variant="easy"):
    """`senior-prove-and-apply-the-pythagorean-identities`. Rubric-assessed (`answer: rubric`): every question
    returned carries a per-item marking dict (`Question.answer`), never a plain value., the traps and the
    CAS-confirmability sweep."""
    if variant == "easy":
        if rng.random() < 0.5:
            return _prove_item(rng, rng.choice(FAMILIES_BY_TIER["easy"]))
        return _tan_apply_item(rng) if rng.random() < 0.5 else _sc_apply_item(rng)
    if variant == "medium":
        return _prove_item(rng, rng.choice(FAMILIES_BY_TIER["medium"]))
    if variant == "hard":
        return _prove_item(rng, rng.choice(FAMILIES_BY_TIER["hard"]))
    if variant == "exam":
        if rng.random() < 0.5:
            return _prove_item(rng, rng.choice(FAMILIES_BY_TIER["exam"]))
        return _two_step_csc_item(rng) if rng.random() < 0.5 else _two_step_cos2_item(rng)
    raise ValueError(f"unknown variant: {variant}")
