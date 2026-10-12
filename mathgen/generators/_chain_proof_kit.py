"""Shared plumbing for the chain-proof generators (open-response proofs whose working is a chain of equal
expressions).

Each such generator returns, as `Question.answer`, a marking dict:

    {"start", "target", "symbols", "reference_lines", "key_steps", "traps", "checks"}

The learner types one expression per line. `start` and `target` are the first and last line of the reference
chain; `checks` holds one `equivalence_chain` check over them. The parts of a proof a symbolic check cannot
see (a base case, a stated conclusion, a cited law) are named in `key_steps` for whoever marks the rest.

This module is not a generator (its name starts with an underscore, so the generators package skips it). It holds
what every chain-proof generator needs so none of them hand-rolls it: turning a typeable sympy-style line into
the plain-text and LaTeX forms of a prompt, composing a prompt as text and as LaTeX from the same parts, a cheap
numeric check that a reference chain is a chain of equal expressions, and the marking dict with its traps.

A family (one kind of proof) is a plain dict:

    key        short unique name
    tier       "easy" | "medium" | "hard" | "exam"
    symbols    {name: kind} for the check, kind in real / positive / integer / nonzero
    ref        the reference chain, a list of sympy-style lines (first line = start, last line = target)
    lead       the statement to prove, as a list of prompt parts (see `T`, `M`, `R`)
    kind       "identity" or "induction" (what the unnamed parts of the proof are)
    domain     optional sentence stating where the claim holds
    given      optional short description of the law or definition the proof may cite
    probes     optional {name: [values]} overriding the default numeric probe points
    extra      optional extra notation advice appended to the notation sentence
    jump_ok    optional bool; True when [ref[0], ref[1], ref[-1]] is still a valid chain

Apply families (a numeric chain) carry `key`, `tier`, `ref`, `lead`, `wrong_final` and `answer_text`.
"""

from __future__ import annotations

import functools
import math
import re

import sympy as sp

from ..core import check

# ---------------------------------------------------------------------------
# Prompt parts. A prompt is a list of parts; the same list renders as plain text and as LaTeX.
# ---------------------------------------------------------------------------


def T(text: str) -> tuple:
    """Prose. Plain in `text`; `\\text{...}` in `latex`."""
    return ("t", text)


def M(sym: str) -> tuple:
    """A sympy-style expression. Typeable ASCII in `text`; real LaTeX in `latex`."""
    return ("m", sym)


def R(disp_text: str, latex_text: str) -> tuple:
    """A hand-written display form for something a line of sympy syntax cannot say (a series with dots)."""
    return ("r", disp_text, latex_text)


_FUNC_NAMES = "sin|cos|tan|sec|csc|cot|sqrt|exp|log|ln|factorial"
_FUNC_START = re.compile(rf"(?:{_FUNC_NAMES})\b")


def _text_verbatim(s: str) -> str:
    """`\\text{...}` with the one special character these prompts carry (`^`) escaped the KaTeX-safe way."""
    return r"\text{" + s.replace("^", r"\textasciicircum{}") + "}"


def _drop_stars(s: str) -> str:
    """Remove the `*` multiplication signs from a sympy-style line, the way a textbook writes the product."""
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch != "*":
            out.append(ch)
            i += 1
            continue
        prev = out[-1] if out else ""
        # Skip the spaces already emitted before the star.
        j = len(out) - 1
        while j >= 0 and out[j] == " ":
            j -= 1
        prev = out[j] if j >= 0 else ""
        rest = s[i + 1 :].lstrip()
        nxt = rest[:1]
        while out and out[-1] == " ":
            out.pop()
        if prev.isdigit() and nxt.isdigit():
            out.append(" × ")
        elif nxt == "(" and (prev.isalnum() or prev == ")" or prev == "!"):
            out.append("")
        elif prev == ")" and nxt.isdigit():
            out.append(" × ")
        elif nxt.isalpha():
            if (prev.isdigit() or prev == ")") and not _FUNC_START.match(rest):
                out.append("")
            else:
                out.append(" ")
        elif nxt.isdigit():
            out.append(" × ")
        else:
            out.append(" ")
        i += 1
        while i < len(s) and s[i] == " ":
            i += 1
    return "".join(out)


_FACT_RE = re.compile(r"factorial\(([^()]*(?:\([^()]*\)[^()]*)*)\)")


_ONE_TIMES_RE = re.compile(r"(?<![\w.])1\*(?=[A-Za-z(])")  # a literal 1 times something is never printed


def disp(sym: str) -> str:
    """The plain-text form of a sympy-style line: `^` for powers, products side by side, `n!` for factorials."""
    s = _ONE_TIMES_RE.sub("", sym)

    def fact(m: re.Match) -> str:
        arg = m.group(1).strip()
        return arg + "!" if re.fullmatch(r"[A-Za-z0-9]+", arg) else "(" + arg + ")!"

    while "factorial(" in s:
        s = _FACT_RE.sub(fact, s)
    s = s.replace("**", "^")
    s = _drop_stars(s)
    s = re.sub(r"(?<=[\w)!])\s*([+\-])\s*(?=[\w(])", r" \1 ", s)
    s = s.replace("pi", "π") if re.search(r"(?<![A-Za-z])pi(?![A-Za-z])", s) else s
    return re.sub(r"\s+", " ", s).strip()


@functools.lru_cache(maxsize=None)
def tex(sym: str) -> str:
    """The LaTeX form of a sympy-style line, written in the order the line is written."""
    out = sp.latex(sp.sympify(_ONE_TIMES_RE.sub("", sym), evaluate=False), order="none")
    # An unevaluated 1/t prints as the product `1 \\frac{1}{t}`; the leading 1 is only the numerator.
    return re.sub(r"(?<![\d.])1 \\frac\{", r"\\frac{", out)


def render(parts: list[tuple]) -> tuple[str, str]:
    """`(text, latex)` of a list of prompt parts."""
    text, latex = [], []
    for part in parts:
        if part[0] == "t":
            text.append(part[1])
            latex.append(_text_verbatim(part[1]))
        elif part[0] == "m":
            text.append(disp(part[1]))
            latex.append(tex(part[1]))
        else:
            text.append(part[1])
            latex.append(part[2])
    return "".join(text), "".join(latex)


# ---------------------------------------------------------------------------
# The numeric check a generator runs on its own reference chain (math module only, never a CAS).
# ---------------------------------------------------------------------------
_DEFAULT_PROBES = {
    "real": (0.37, 1.19, 2.53),
    "positive": (0.37, 1.19, 2.53),
    "nonzero": (0.37, 1.19, 2.53),
    "integer": (6, 7, 9, 12),
}

_NS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sec": lambda v: 1.0 / math.cos(v),
    "csc": lambda v: 1.0 / math.sin(v),
    "cot": lambda v: 1.0 / math.tan(v),
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "factorial": lambda v: math.gamma(v + 1),
    "pi": math.pi,
}


def _eval(line: str, env: dict) -> float:
    return eval(line, {"__builtins__": {}}, {**_NS, **env})  # noqa: S307 -- fixed trusted templates only


def verify_chain(ref: list[str], symbols: dict, probes: dict | None, msg: str) -> None:
    """Every consecutive pair of lines agrees at every probe point (a fresh point per symbol, all combined)."""
    names = sorted(symbols)
    grids = []
    for name in names:
        values = (probes or {}).get(name) or _DEFAULT_PROBES.get(symbols[name], _DEFAULT_PROBES["real"])
        grids.append(values)
    n_points = max([len(g) for g in grids] + [1])
    for p in range(n_points):
        env = {name: grids[i][(p + i) % len(grids[i])] for i, name in enumerate(names)}
        values = [_eval(line, env) for line in ref]
        for i in range(1, len(values)):
            check(math.isclose(values[i], values[i - 1], rel_tol=1e-7, abs_tol=1e-9), msg)


def verify_numeric(ref: list[str], msg: str) -> None:
    values = [_eval(line, {}) for line in ref]
    for i in range(1, len(values)):
        check(math.isclose(values[i], values[i - 1], rel_tol=1e-9, abs_tol=1e-12), msg)


def differs(line_a: str, line_b: str, symbols: dict, probes: dict | None) -> bool:
    """True when two lines disagree at some probe point (used to prove a trap line really is wrong)."""
    names = sorted(symbols)
    grids = [(probes or {}).get(name) or _DEFAULT_PROBES.get(symbols[name], _DEFAULT_PROBES["real"]) for name in names]
    n_points = max([len(g) for g in grids] + [1])
    for p in range(n_points):
        env = {name: grids[i][(p + i) % len(grids[i])] for i, name in enumerate(names)}
        if not math.isclose(_eval(line_a, env), _eval(line_b, env), rel_tol=1e-7, abs_tol=1e-9):
            return True
    return False


# ---------------------------------------------------------------------------
# Prompt wording
# ---------------------------------------------------------------------------
NOTATION = (
    "Put brackets round every function argument, as in sin(x), use ^ for powers, as in cos(x)^2, "
    "and write products side by side, as in 2n(n + 1)."
)

#: Three wordings of the chain instruction; the start/target parts fill the gaps.
_CHAIN_CLAUSES = (
    (" Start your working from ", " and finish at ", ", writing one expression per line. "),
    (" Begin from ", " and end at ", ", one typed expression per line. "),
    (" Write a chain of equal expressions, one per line, starting at ", " and finishing at ", ". "),
)


def _prompt_parts(rng, fam: dict, start: str, target: str) -> list[tuple]:
    idx = rng.randrange(len(_CHAIN_CLAUSES))
    a, b, c = _CHAIN_CLAUSES[idx]
    parts = list(fam["lead"])
    if fam.get("domain"):
        parts.append(T(" " + fam["domain"]))
    extra = (" " + fam["extra"]) if fam.get("extra") else ""
    parts += [T(a), M(start), T(b), M(target), T(c + NOTATION + extra)]
    return parts


# ---------------------------------------------------------------------------
# Marking dict
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _same_expression(a: str, b: str) -> bool:
    return sp.sympify(a) == sp.sympify(b)


def _traps(fam: dict) -> list[dict]:
    ref = fam["ref"]
    start, target = ref[0], ref[-1]
    symbols = fam["symbols"]
    probes = fam.get("probes")
    mid = ref[1] if len(ref) > 2 else start
    slip = f"({mid})*{sorted(symbols)[0]}"
    scaled = f"2*({mid})"
    check(differs(slip, mid, symbols, probes), f"{fam['key']}: slip trap line equals its source")
    check(differs(scaled, mid, symbols, probes), f"{fam['key']}: scale trap line equals its source")
    traps = [
        {
            "name": "slips_in_a_stray_factor",
            "question": "does every line equal the line before it, or does a stray factor appear partway through?",
            "example_lines": [start, slip, target],
            "expected_check": "authoritative_failure",
        },
        {
            "name": "mis_scales_a_step",
            "question": "is any line a multiple of the line before it (a dropped or doubled factor)?",
            "example_lines": [start, scaled, target],
            "expected_check": "authoritative_failure",
        },
    ]
    traps.append(
        {
            "name": "ends_at_a_different_expression",
            "question": "does the last line match the stated target exactly, not a nearby expression?",
            "example_lines": [*ref[:-1], f"2*({target})"],
            "expected_check": "non_authoritative_failure",
        }
    )
    if not _same_expression(start, target):  # otherwise the CAS reads both ends as one expression
        traps.append(
            {
                "name": "starts_from_the_target",
                "question": "does the working genuinely start from the stated first expression, not the target?",
                "example_lines": [target, start],
                "expected_check": "non_authoritative_failure",
            }
        )
    if fam["kind"] == "induction":
        traps.append(
            {
                "name": "omits_base_case_or_conclusion",
                "question": "does the response check the base case, state the inductive hypothesis and "
                "close with a conclusion for every positive integer?",
                "example_lines": list(ref),
                "expected_check": "passes_chain",
            }
        )
    else:
        traps.append(
            {
                "name": "no_reason_given",
                "question": "does the response name the law or definition behind each step, or only list expressions?",
                "example_lines": list(ref),
                "expected_check": "passes_chain",
            }
        )
    if fam.get("jump_ok") and len(ref) >= 4:
        traps.append(
            {
                "name": "skips_the_working",
                "question": "does the response show the intermediate algebra, or jump to the target?",
                "example_lines": [start, ref[1], target],
                "expected_check": "passes_chain",
            }
        )
    return traps


def _key_steps(fam: dict, start_d: str, target_d: str) -> list[dict]:
    steps = [
        {
            "criterion_id": "correctness",
            "description": f"transforms {start_d} into {target_d} using only valid algebraic steps",
        }
    ]
    if fam["kind"] == "induction":
        steps += [
            {
                "criterion_id": "logical_completeness",
                "description": "checks the base case, states the inductive hypothesis for n = k, uses it in the "
                "step, and concludes the result holds for every positive integer n",
            },
            {
                "criterion_id": "definitions_assumptions",
                "description": "assumes the result for n = k only (not for all n) and says n is a positive integer",
            },
        ]
    else:
        steps += [
            {
                "criterion_id": "logical_completeness",
                "description": "every line follows from the line before it and the last line is exactly the target",
            },
            {
                "criterion_id": "definitions_assumptions",
                "description": fam.get("given_step") or "states the definition or law each step relies on",
            },
        ]
    steps.append(
        {
            "criterion_id": "communication",
            "description": "writes one expression per line, in order, ending at the stated target",
        }
    )
    return steps


def prove_item(rng, fam: dict) -> dict:
    """Build the question dict for a prove family (symbolic chain)."""
    ref = list(fam["ref"])
    start, target = ref[0], ref[-1]
    verify_chain(ref, fam["symbols"], fam.get("probes"), f"{fam['key']}: reference chain does not hold numerically")
    start_d, target_d = disp(start), disp(target)
    text, latex = render(_prompt_parts(rng, fam, start, target))
    check("**" not in text and "*" not in text, f"{fam['key']}: prompt text carries a star")
    symbols = dict(fam["symbols"])
    answer = {
        "start": start,
        "target": target,
        "symbols": symbols,
        "reference_lines": ref,
        "key_steps": _key_steps(fam, start_d, target_d),
        "traps": _traps(fam),
        "checks": [
            {"type": "equivalence_chain", "start": start, "target": target, "symbols": symbols, "authoritative": True}
        ],
    }
    return dict(text=text, latex=latex, answer=answer, answer_text=f"{start_d} = {target_d}")


def apply_item(rng, fam: dict) -> dict:
    """Build the question dict for an apply family (numeric chain; `symbols` is empty)."""
    ref = list(fam["ref"])
    start, target = ref[0], ref[-1]
    verify_numeric(ref, f"{fam['key']}: reference chain does not hold numerically")
    start_d, target_d = disp(start), disp(target)
    parts = list(fam["lead"]) + [
        T(" Start from "),
        M(start),
        T(
            " and finish at the exact value, one expression per line. "
            + NOTATION
            + (" " + fam["extra"] if fam.get("extra") else "")
        ),
    ]
    text, latex = render(parts)
    check("**" not in text and "*" not in text, f"{fam['key']}: prompt text carries a star")
    wrong = fam["wrong_final"]
    check(not math.isclose(_eval(wrong, {}), _eval(target, {}), rel_tol=1e-9), f"{fam['key']}: wrong value is right")
    traps = [
        {
            "name": "wrong_final_value",
            "question": "does the final line match the exact value asked for?",
            "example_lines": ref[:-1] + [wrong],
            "expected_check": "non_authoritative_failure",
        },
        {
            "name": "no_substitution_shown",
            "question": "does the response show the substitution, or just assert the final value?",
            "example_lines": [start, target],
            "expected_check": "passes_chain",
        },
    ]
    key_steps = [
        {
            "criterion_id": "correctness",
            "description": f"evaluates {start_d} correctly and finishes at {target_d}",
        },
        {
            "criterion_id": "logical_completeness",
            "description": "shows the substitution and each simplification as separate lines",
        },
        {
            "criterion_id": "definitions_assumptions",
            "description": fam.get("given_step") or "names the formula or law being applied",
        },
        {
            "criterion_id": "communication",
            "description": "writes one expression per line, ending at the exact value",
        },
    ]
    answer = {
        "start": start,
        "target": target,
        "symbols": {},
        "reference_lines": ref,
        "key_steps": key_steps,
        "traps": traps,
        "checks": [
            {"type": "equivalence_chain", "start": start, "target": target, "symbols": {}, "authoritative": True}
        ],
    }
    return dict(text=text, latex=latex, answer=answer, answer_text=fam["answer_text"])


def pick(rng, items):
    return items[rng.randrange(len(items))]


def auto_ref(start: str, target: str) -> list[str]:
    """A reference chain `start`, a single fraction with the numerator multiplied out, `target`."""
    numer, denom = sp.fraction(sp.together(sp.sympify(start)))
    middle = f"({sp.expand(numer)})/({denom})" if denom != 1 else str(sp.expand(numer))
    lines = [start]
    if middle.replace(" ", "") not in (start.replace(" ", ""), target.replace(" ", "")):
        lines.append(middle)
    lines.append(target)
    return lines
