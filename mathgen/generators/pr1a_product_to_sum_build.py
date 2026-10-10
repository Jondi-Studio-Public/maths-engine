"""One topic, ``pr1a_product_to_sum_proof``, four presets (easy/medium/hard/
exam). PROVE items derive one of the four product-to-sum identities

    sin(A)cos(B) = (1/2)[sin(A + B) + sin(A - B)]
    cos(A)sin(B) = (1/2)[sin(A + B) - sin(A - B)]
    cos(A)cos(B) = (1/2)[cos(A - B) + cos(A + B)]
    sin(A)sin(B) = (1/2)[cos(A - B) - cos(A + B)]

by adding/subtracting the (cited, never re-proved -- that is ACMMM041's
kernel) angle-sum and angle-difference formulas. APPLY items
express a stated product of sines/cosines of numeric multiples of x as a
sum or difference (medium), or evaluate a stated product of special-angle
values exactly (exam). The hard tier proves the four SUM-TO-PRODUCT
siblings (derived from the product-to-sum results above by substituting
A = (P + R)/2, B = (P - R)/2 -- cited, working backwards) and applies them
to numeric instances (sin(5x) + sin(3x) = 2 sin(4x)cos(x)). The exam tier
also proves a composite identity chaining two sum-to-product steps
((sin(3x) + sin(x))/(cos(3x) + cos(x)) = tan(2x)).

This skill NEVER re-derives the angle-sum/difference formulas themselves
(ACMMM041's own territory) and never asks for a double-angle
identity (ACMSM044's territory).

Only pairs whose reference chain is directly confirmed by `mathgen.core.symbolically_equal` are kept in the
pools below.

Cheap in-generator verification only: every reference chain here
is a FIXED pool entry (finite draw space), so each is checked once, at
import time, with the `math` module at a small set of fixed-but-irregular
real points -- never a full CAS chain check, which costs 0.1-0.4s per
chain.
"""

from __future__ import annotations

import math
import random
import re

from ..core import check, generator, phrase

# ---------------------------------------------------------------------------
# Cheap numeric verification helpers -- `math`-only, no sympy here.
# ---------------------------------------------------------------------------

_PROBE_AB = ((0.4, 0.7), (1.0, -0.5), (2.3, 0.9))
_PROBE_X = (0.4, 1.1, 2.3)

_EVAL_NS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "sqrt": math.sqrt}


def _eval_line(line: str, values: dict[str, float] | None = None) -> float:
    ns = dict(_EVAL_NS)
    if values:
        ns.update(values)
    return eval(line, {"__builtins__": {}}, ns)  # noqa: S307 -- fixed internal strings only


def _verify_chain(reference_lines: list[str], label: str, values_seq) -> None:
    for values in values_seq:
        computed = [_eval_line(line, values) for line in reference_lines]
        for i in range(len(computed) - 1):
            check(
                math.isclose(computed[i], computed[i + 1], rel_tol=1e-9, abs_tol=1e-9),
                f"{label}: reference line {i + 1} does not numerically match line {i}",
            )


def _verify_symbolic_chain(reference_lines: list[str], label: str, letters: tuple[str, str] = ("A", "B")) -> None:
    l1, l2 = letters
    _verify_chain(reference_lines, label, ({l1: a, l2: b} for a, b in _PROBE_AB))


def _verify_single_symbol_chain(reference_lines: list[str], label: str, symbol: str = "x") -> None:
    _verify_chain(reference_lines, label, ({symbol: x} for x in _PROBE_X))


def _verify_numeric_chain(reference_lines: list[str], label: str) -> None:
    _verify_chain(reference_lines, label, ({},))


# ---------------------------------------------------------------------------
# Shared marking-dict / key-step / trap builders (same shape as the sibling pr1a modules).
# ---------------------------------------------------------------------------


def _marking(start, target, symbols, reference_lines, key_steps, traps):
    return {
        "start": start,
        "target": target,
        "symbols": symbols,
        "reference_lines": list(reference_lines),
        "key_steps": key_steps,
        "traps": traps,
        "checks": [
            {
                "type": "equivalence_chain",
                "start": start,
                "target": target,
                "symbols": symbols,
                "authoritative": True,
            }
        ],
    }


def _trap(name, question, example_lines, expected_check):
    return {
        "name": name,
        "question": question,
        "example_lines": list(example_lines),
        "expected_check": expected_check,
    }


def _key_steps(given_names: str, cited_step: str):
    return [
        {
            "criterion_id": "correctness",
            "description": "Every line follows from the previous by a valid algebraic or trigonometric step.",
        },
        {
            "criterion_id": "justified_steps",
            "description": f"Cites {given_names} by name at the step where it is used ({cited_step}).",
        },
        {
            "criterion_id": "communication",
            "description": "Each line is displayed as its own step and the final line matches the stated target exactly.",
        },
    ]


# ---------------------------------------------------------------------------
# Letter-pair substitution (the safe list): every prove entry is
# written once in a canonical two-letter form, then expanded over 5 Latin
# letter pairs so the pool has real letter variety instead of one fixed
# pair repeated at every seed.
# ---------------------------------------------------------------------------

_LETTER_PAIRS = [("A", "B", "ab"), ("P", "R", "pr"), ("x", "t", "xt"), ("u", "v", "uv"), ("x", "u", "xu")]

_TOKEN_RE_CACHE: dict[str, re.Pattern] = {}


def _token_re(letter: str) -> re.Pattern:
    if letter not in _TOKEN_RE_CACHE:
        _TOKEN_RE_CACHE[letter] = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(letter)}(?![A-Za-z0-9_])")
    return _TOKEN_RE_CACHE[letter]


def _sub_two_letters(text: str, old1: str, old2: str, new1: str, new2: str) -> str:
    ph1, ph2 = "\x00A\x00", "\x00B\x00"
    text = _token_re(old1).sub(ph1, text)
    text = _token_re(old2).sub(ph2, text)
    return text.replace(ph1, new1).replace(ph2, new2)


def _subst_deep(obj, old1: str, old2: str, new1: str, new2: str):
    if isinstance(obj, str):
        return _sub_two_letters(obj, old1, old2, new1, new2)
    if isinstance(obj, list):
        return [_subst_deep(item, old1, old2, new1, new2) for item in obj]
    if isinstance(obj, dict):
        return {k: _subst_deep(v, old1, old2, new1, new2) for k, v in obj.items()}
    return obj


def _expand_family(entry: dict, canon: tuple[str, str] = ("A", "B")) -> list[dict]:
    canon1, canon2 = canon
    variants = []
    for l1, l2, tag in _LETTER_PAIRS:
        variant = _subst_deep(entry, canon1, canon2, l1, l2)
        variant["id"] = f"{entry['id']}_{tag}"
        variant["symbols"] = {l1: "real", l2: "real"}
        variant["letters"] = (l1, l2)
        variants.append(variant)
    return variants


# ---------------------------------------------------------------------------
# EASY -- prove one of the four product-to-sum identities, starting from
# the (given) sum/difference side, ending at the product (rule: "state
# which side the chain starts from" -- always the left of `identity_read`).
# Canonical letters A, B.
# ---------------------------------------------------------------------------

_SINCOS_TO_PRODUCT = dict(
    id="sincos_to_product",
    tier="easy",
    start="(sin(A + B) + sin(A - B))/2",
    target="sin(A)*cos(B)",
    identity_read="(sin(A + B) + sin(A - B))/2 = sin(A)cos(B)",
    identity_tex=r"\frac{\sin(A + B) + \sin(A - B)}{2} = \sin(A)\cos(B)",
    given_read=("sin(A + B) = sin(A)cos(B) + cos(A)sin(B) and sin(A - B) = sin(A)cos(B) - cos(A)sin(B)"),
    given_tex=(
        r"\sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B)\text{ and }"
        r"\sin(A - B) = \sin(A)\cos(B) - \cos(A)\sin(B)"
    ),
    reference_lines=[
        "(sin(A + B) + sin(A - B))/2",
        "(sin(A)*cos(B) + cos(A)*sin(B) + sin(A)*cos(B) - cos(A)*sin(B))/2",
        "(2*sin(A)*cos(B))/2",
        "sin(A)*cos(B)",
    ],
    key_steps=_key_steps(
        "the given sin(A + B) and sin(A - B) formulas",
        "substituting both given formulas and cancelling the cos(A)sin(B) terms",
    ),
    traps=[
        _trap(
            "forgets_the_half",
            "Does the response actually halve the sum of the two given formulas, or drop the 1/2?",
            ["(sin(A + B) + sin(A - B))/2", "2*sin(A)*cos(B)", "sin(A)*cos(B)"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_wrong_side",
            "Does the proof actually start from the left-hand side (1/2)[sin(A + B) + sin(A - B)]?",
            [
                "sin(A)*cos(B)",
                "(2*sin(A)*cos(B))/2",
                "(sin(A)*cos(B) + cos(A)*sin(B) + sin(A)*cos(B) - cos(A)*sin(B))/2",
                "(sin(A + B) + sin(A - B))/2",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "reproves_angle_sum",
            "Does the response cite the given sin(A + B)/sin(A - B) formulas, or re-derive them from scratch?",
            [
                "(sin(A + B) + sin(A - B))/2",
                "(sin(A)*cos(B) + cos(A)*sin(B) + sin(A)*cos(B) - cos(A)*sin(B))/2",
                "(2*sin(A)*cos(B))/2",
                "sin(A)*cos(B)",
            ],
            "passes_chain",
        ),
    ],
)

_COSSIN_TO_PRODUCT = dict(
    id="cossin_to_product",
    tier="easy",
    start="(sin(A + B) - sin(A - B))/2",
    target="cos(A)*sin(B)",
    identity_read="(sin(A + B) - sin(A - B))/2 = cos(A)sin(B)",
    identity_tex=r"\frac{\sin(A + B) - \sin(A - B)}{2} = \cos(A)\sin(B)",
    given_read=("sin(A + B) = sin(A)cos(B) + cos(A)sin(B) and sin(A - B) = sin(A)cos(B) - cos(A)sin(B)"),
    given_tex=(
        r"\sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B)\text{ and }"
        r"\sin(A - B) = \sin(A)\cos(B) - \cos(A)\sin(B)"
    ),
    reference_lines=[
        "(sin(A + B) - sin(A - B))/2",
        "(sin(A)*cos(B) + cos(A)*sin(B) - (sin(A)*cos(B) - cos(A)*sin(B)))/2",
        "(2*cos(A)*sin(B))/2",
        "cos(A)*sin(B)",
    ],
    key_steps=_key_steps(
        "the given sin(A + B) and sin(A - B) formulas",
        "substituting both given formulas and cancelling the sin(A)cos(B) terms",
    ),
    traps=[
        _trap(
            "forgets_the_half",
            "Does the response actually halve the difference of the two given formulas, or drop the 1/2?",
            ["(sin(A + B) - sin(A - B))/2", "2*cos(A)*sin(B)", "cos(A)*sin(B)"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_wrong_side",
            "Does the proof actually start from the left-hand side (1/2)[sin(A + B) - sin(A - B)]?",
            [
                "cos(A)*sin(B)",
                "(2*cos(A)*sin(B))/2",
                "(sin(A)*cos(B) + cos(A)*sin(B) - (sin(A)*cos(B) - cos(A)*sin(B)))/2",
                "(sin(A + B) - sin(A - B))/2",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "reproves_angle_sum",
            "Does the response cite the given sin(A + B)/sin(A - B) formulas, or re-derive them from scratch?",
            [
                "(sin(A + B) - sin(A - B))/2",
                "(sin(A)*cos(B) + cos(A)*sin(B) - (sin(A)*cos(B) - cos(A)*sin(B)))/2",
                "(2*cos(A)*sin(B))/2",
                "cos(A)*sin(B)",
            ],
            "passes_chain",
        ),
    ],
)

_COSCOS_TO_SUM = dict(
    id="coscos_to_sum",
    tier="easy",
    start="(cos(A - B) + cos(A + B))/2",
    target="cos(A)*cos(B)",
    identity_read="(cos(A - B) + cos(A + B))/2 = cos(A)cos(B)",
    identity_tex=r"\frac{\cos(A - B) + \cos(A + B)}{2} = \cos(A)\cos(B)",
    given_read=("cos(A - B) = cos(A)cos(B) + sin(A)sin(B) and cos(A + B) = cos(A)cos(B) - sin(A)sin(B)"),
    given_tex=(
        r"\cos(A - B) = \cos(A)\cos(B) + \sin(A)\sin(B)\text{ and }"
        r"\cos(A + B) = \cos(A)\cos(B) - \sin(A)\sin(B)"
    ),
    reference_lines=[
        "(cos(A - B) + cos(A + B))/2",
        "(cos(A)*cos(B) + sin(A)*sin(B) + cos(A)*cos(B) - sin(A)*sin(B))/2",
        "(2*cos(A)*cos(B))/2",
        "cos(A)*cos(B)",
    ],
    key_steps=_key_steps(
        "the given cos(A - B) and cos(A + B) formulas",
        "substituting both given formulas and cancelling the sin(A)sin(B) terms",
    ),
    traps=[
        _trap(
            "pairs_the_wrong_sum_and_difference",
            "Does the response use the correct pair of formulas cos(A - B) AND cos(A + B), or swap in the wrong sign?",
            ["(cos(A - B) + cos(A + B))/2", "sin(A)*sin(B)", "cos(A)*cos(B)"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_wrong_side",
            "Does the proof actually start from the left-hand side (1/2)[cos(A - B) + cos(A + B)]?",
            [
                "cos(A)*cos(B)",
                "(2*cos(A)*cos(B))/2",
                "(cos(A)*cos(B) + sin(A)*sin(B) + cos(A)*cos(B) - sin(A)*sin(B))/2",
                "(cos(A - B) + cos(A + B))/2",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "reproves_angle_sum",
            "Does the response cite the given cos(A - B)/cos(A + B) formulas, or re-derive them from scratch?",
            [
                "(cos(A - B) + cos(A + B))/2",
                "(cos(A)*cos(B) + sin(A)*sin(B) + cos(A)*cos(B) - sin(A)*sin(B))/2",
                "(2*cos(A)*cos(B))/2",
                "cos(A)*cos(B)",
            ],
            "passes_chain",
        ),
    ],
)

_SINSIN_TO_DIFF = dict(
    id="sinsin_to_diff",
    tier="easy",
    start="(cos(A - B) - cos(A + B))/2",
    target="sin(A)*sin(B)",
    identity_read="(cos(A - B) - cos(A + B))/2 = sin(A)sin(B)",
    identity_tex=r"\frac{\cos(A - B) - \cos(A + B)}{2} = \sin(A)\sin(B)",
    given_read=("cos(A - B) = cos(A)cos(B) + sin(A)sin(B) and cos(A + B) = cos(A)cos(B) - sin(A)sin(B)"),
    given_tex=(
        r"\cos(A - B) = \cos(A)\cos(B) + \sin(A)\sin(B)\text{ and }"
        r"\cos(A + B) = \cos(A)\cos(B) - \sin(A)\sin(B)"
    ),
    reference_lines=[
        "(cos(A - B) - cos(A + B))/2",
        "(cos(A)*cos(B) + sin(A)*sin(B) - (cos(A)*cos(B) - sin(A)*sin(B)))/2",
        "(2*sin(A)*sin(B))/2",
        "sin(A)*sin(B)",
    ],
    key_steps=_key_steps(
        "the given cos(A - B) and cos(A + B) formulas",
        "substituting both given formulas and cancelling the cos(A)cos(B) terms",
    ),
    traps=[
        _trap(
            "pairs_the_wrong_sum_and_difference",
            "Does the response use the correct pair of formulas cos(A - B) AND cos(A + B), or swap in the wrong sign?",
            ["(cos(A - B) - cos(A + B))/2", "cos(A)*cos(B)", "sin(A)*sin(B)"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_wrong_side",
            "Does the proof actually start from the left-hand side (1/2)[cos(A - B) - cos(A + B)]?",
            [
                "sin(A)*sin(B)",
                "(2*sin(A)*sin(B))/2",
                "(cos(A)*cos(B) + sin(A)*sin(B) - (cos(A)*cos(B) - sin(A)*sin(B)))/2",
                "(cos(A - B) - cos(A + B))/2",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "reproves_angle_sum",
            "Does the response cite the given cos(A - B)/cos(A + B) formulas, or re-derive them from scratch?",
            [
                "(cos(A - B) - cos(A + B))/2",
                "(cos(A)*cos(B) + sin(A)*sin(B) - (cos(A)*cos(B) - sin(A)*sin(B)))/2",
                "(2*sin(A)*sin(B))/2",
                "sin(A)*sin(B)",
            ],
            "passes_chain",
        ),
    ],
)

_EASY_POOL: list[dict] = []
for _fam in (_SINCOS_TO_PRODUCT, _COSSIN_TO_PRODUCT, _COSCOS_TO_SUM, _SINSIN_TO_DIFF):
    _EASY_POOL.extend(_expand_family(_fam, canon=("A", "B")))

for _entry in _EASY_POOL:
    _verify_symbolic_chain(_entry["reference_lines"], _entry["id"], letters=_entry["letters"])


# MEDIUM -- apply to numeric multiples of x, starting from the PRODUCT (the reverse decision from
# easy). Sign decision (rule "include cases where A < B") lives in the sincos family, whose sin(A -
# B) term is odd: when m < n, sin((m - n)x) = -sin((n - m)x). The coscos/sinsin families never need
# that decision (cos is even), so their authoritative trap is a pairing/sign swap instead. Only (m,
# n) pairs whose 4-line reference chain was directly confirmed against `symbolically_equal` are
# kept.


def _mult_read(k: int) -> str:
    return "x" if k == 1 else f"{k}x"


def _sincos_medium_entry(m: int, n: int) -> dict:
    sum_m, diff_m = m + n, abs(m - n)
    a, b = f"{m}*x", f"{n}*x"
    start = f"2*sin({a})*cos({b})"
    mid = f"sin({a} + {b}) + sin({a} - {b})"
    if m >= n:
        target = f"sin({sum_m}*x) + sin({diff_m}*x)"
        wrong_target = f"sin({sum_m}*x) - sin({diff_m}*x)"
    else:
        target = f"sin({sum_m}*x) - sin({diff_m}*x)"
        wrong_target = f"sin({sum_m}*x) + sin({diff_m}*x)"
    start_read = f"2 sin({_mult_read(m)})cos({_mult_read(n)})"
    target_read = target.replace("*x", "x") if m >= n else target.replace("*x", "x")
    target_read = re.sub(r"(\d)x", r"\1x", target.replace(" ", "")).replace("*", "")
    return dict(
        id=f"sincos_med_{m}_{n}",
        tier="medium",
        symbols={"x": "real"},
        start=start,
        target=target,
        start_read=start_read,
        target_read=target_read,
        given_read="2 sin(A)cos(B) = sin(A + B) + sin(A - B)",
        given_tex=r"2\sin(A)\cos(B) = \sin(A + B) + \sin(A - B)",
        reference_lines=[start, mid, target],
        key_steps=_key_steps(
            "the given 2 sin(A)cos(B) = sin(A + B) + sin(A - B) identity",
            f"substituting A = {_mult_read(m)}, B = {_mult_read(n)} and simplifying the arguments",
        ),
        traps=[
            _trap(
                "argument_slip",
                f"Does the response get the SIGN of sin({_mult_read(diff_m)}) right (odd symmetry of sine), "
                "not just its size?",
                [start, wrong_target, target],
                "authoritative_failure",
            ),
            _trap(
                "starts_from_wrong_side",
                "Does the response start from the stated PRODUCT, not the sum it is asked to derive?",
                [target, mid, start],
                "non_authoritative_failure",
            ),
            _trap(
                "cites_the_result_being_proved",
                "Does the response show the substitution step, or just assert the sum directly?",
                [start, target],
                "passes_chain",
            ),
        ],
    )


def _coscos_medium_entry(m: int, n: int) -> dict:
    sum_m, diff_m = m + n, abs(m - n)
    a, b = f"{m}*x", f"{n}*x"
    start = f"2*cos({a})*cos({b})"
    mid = f"cos({a} - {b}) + cos({a} + {b})"
    target = f"cos({diff_m}*x) + cos({sum_m}*x)"
    wrong_target = f"cos({diff_m}*x) - cos({sum_m}*x)"
    start_read = f"2 cos({_mult_read(m)})cos({_mult_read(n)})"
    target_read = f"cos({_mult_read(diff_m)}) + cos({_mult_read(sum_m)})"
    return dict(
        id=f"coscos_med_{m}_{n}",
        tier="medium",
        symbols={"x": "real"},
        start=start,
        target=target,
        start_read=start_read,
        target_read=target_read,
        given_read="2 cos(A)cos(B) = cos(A - B) + cos(A + B)",
        given_tex=r"2\cos(A)\cos(B) = \cos(A - B) + \cos(A + B)",
        reference_lines=[start, mid, target],
        key_steps=_key_steps(
            "the given 2 cos(A)cos(B) = cos(A - B) + cos(A + B) identity",
            f"substituting A = {_mult_read(m)}, B = {_mult_read(n)} and simplifying the arguments",
        ),
        traps=[
            _trap(
                "pairs_the_wrong_sum_and_difference",
                "Does the response add cos(A - B) AND cos(A + B), or use the wrong sign between them?",
                [start, wrong_target, target],
                "authoritative_failure",
            ),
            _trap(
                "starts_from_wrong_side",
                "Does the response start from the stated PRODUCT, not the sum it is asked to derive?",
                [target, mid, start],
                "non_authoritative_failure",
            ),
            _trap(
                "cites_the_result_being_proved",
                "Does the response show the substitution step, or just assert the sum directly?",
                [start, target],
                "passes_chain",
            ),
        ],
    )


def _sinsin_medium_entry(m: int, n: int) -> dict:
    sum_m, diff_m = m + n, abs(m - n)
    a, b = f"{m}*x", f"{n}*x"
    start = f"2*sin({a})*sin({b})"
    mid = f"cos({a} - {b}) - cos({a} + {b})"
    target = f"cos({diff_m}*x) - cos({sum_m}*x)"
    wrong_target = f"cos({diff_m}*x) + cos({sum_m}*x)"
    start_read = f"2 sin({_mult_read(m)})sin({_mult_read(n)})"
    target_read = f"cos({_mult_read(diff_m)}) - cos({_mult_read(sum_m)})"
    return dict(
        id=f"sinsin_med_{m}_{n}",
        tier="medium",
        symbols={"x": "real"},
        start=start,
        target=target,
        start_read=start_read,
        target_read=target_read,
        given_read="2 sin(A)sin(B) = cos(A - B) - cos(A + B)",
        given_tex=r"2\sin(A)\sin(B) = \cos(A - B) - \cos(A + B)",
        reference_lines=[start, mid, target],
        key_steps=_key_steps(
            "the given 2 sin(A)sin(B) = cos(A - B) - cos(A + B) identity",
            f"substituting A = {_mult_read(m)}, B = {_mult_read(n)} and simplifying the arguments",
        ),
        traps=[
            _trap(
                "pairs_the_wrong_sum_and_difference",
                "Does the response subtract cos(A + B) from cos(A - B), or use the wrong sign between them?",
                [start, wrong_target, target],
                "authoritative_failure",
            ),
            _trap(
                "starts_from_wrong_side",
                "Does the response start from the stated PRODUCT, not the difference it is asked to derive?",
                [target, mid, start],
                "non_authoritative_failure",
            ),
            _trap(
                "cites_the_result_being_proved",
                "Does the response show the substitution step, or just assert the difference directly?",
                [start, target],
                "passes_chain",
            ),
        ],
    )


_SINCOS_MEDIUM_PAIRS = [(3, 1), (1, 3), (5, 2), (2, 5)]
_COSCOS_MEDIUM_PAIRS = [(5, 2), (2, 5), (6, 1), (1, 6)]
_SINSIN_MEDIUM_PAIRS = [(3, 1), (1, 3), (5, 2), (2, 5)]

_MEDIUM_POOL: list[dict] = (
    [_sincos_medium_entry(m, n) for m, n in _SINCOS_MEDIUM_PAIRS]
    + [_coscos_medium_entry(m, n) for m, n in _COSCOS_MEDIUM_PAIRS]
    + [_sinsin_medium_entry(m, n) for m, n in _SINSIN_MEDIUM_PAIRS]
)

for _entry in _MEDIUM_POOL:
    _verify_single_symbol_chain(_entry["reference_lines"], _entry["id"])


# ---------------------------------------------------------------------------
# HARD (i) -- prove the sum-to-product forms, derived from the (given)
# product-to-sum results by substituting A = (P + R)/2, B = (P - R)/2
# (canonical letters P, R -- Q is excluded, so R stands in for
# "Q"). Working backwards (recognising P as (P+R)/2 + (P-R)/2) is
# the decision this tier adds over easy/medium.
# ---------------------------------------------------------------------------

_SIN_SUM_TO_PRODUCT = dict(
    id="sin_sum_to_product",
    tier="hard",
    start="sin(P) + sin(R)",
    target="2*sin((P + R)/2)*cos((P - R)/2)",
    identity_read="sin(P) + sin(R) = 2 sin((P + R)/2)cos((P - R)/2)",
    identity_tex=r"\sin(P) + \sin(R) = 2\sin\!\left(\tfrac{P + R}{2}\right)\cos\!\left(\tfrac{P - R}{2}\right)",
    given_read="2 sin(A)cos(B) = sin(A + B) + sin(A - B), with A = (P + R)/2, B = (P - R)/2",
    given_tex=(
        r"2\sin(A)\cos(B) = \sin(A + B) + \sin(A - B)\text{, with }"
        r"A = \tfrac{P + R}{2},\ B = \tfrac{P - R}{2}"
    ),
    reference_lines=[
        "sin(P) + sin(R)",
        "sin((P + R)/2 + (P - R)/2) + sin((P + R)/2 - (P - R)/2)",
        "2*sin((P + R)/2)*cos((P - R)/2)",
    ],
    key_steps=_key_steps(
        "the given 2 sin(A)cos(B) = sin(A + B) + sin(A - B) identity",
        "writing P as (P + R)/2 + (P - R)/2 and R as (P + R)/2 - (P - R)/2 before applying it",
    ),
    traps=[
        _trap(
            "mixes_up_half_sum_and_half_difference",
            "Does the response keep (P + R)/2 and (P - R)/2 in the roles A and B, not swapped?",
            [
                "sin(P) + sin(R)",
                "sin((P + R)/2 + (P - R)/2) + sin((P + R)/2 - (P - R)/2)",
                "2*sin((P - R)/2)*cos((P + R)/2)",
                "2*sin((P + R)/2)*cos((P - R)/2)",
            ],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_wrong_side",
            "Does the response start from sin(P) + sin(R), not the product it is asked to derive?",
            [
                "2*sin((P + R)/2)*cos((P - R)/2)",
                "sin((P + R)/2 + (P - R)/2) + sin((P + R)/2 - (P - R)/2)",
                "sin(P) + sin(R)",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response show the half-sum/half-difference substitution, or just assert the product?",
            ["sin(P) + sin(R)", "2*sin((P + R)/2)*cos((P - R)/2)"],
            "passes_chain",
        ),
    ],
)

_SIN_DIFF_TO_PRODUCT = dict(
    id="sin_diff_to_product",
    tier="hard",
    start="sin(P) - sin(R)",
    target="2*cos((P + R)/2)*sin((P - R)/2)",
    identity_read="sin(P) - sin(R) = 2 cos((P + R)/2)sin((P - R)/2)",
    identity_tex=r"\sin(P) - \sin(R) = 2\cos\!\left(\tfrac{P + R}{2}\right)\sin\!\left(\tfrac{P - R}{2}\right)",
    given_read="2 cos(A)sin(B) = sin(A + B) - sin(A - B), with A = (P + R)/2, B = (P - R)/2",
    given_tex=(
        r"2\cos(A)\sin(B) = \sin(A + B) - \sin(A - B)\text{, with }"
        r"A = \tfrac{P + R}{2},\ B = \tfrac{P - R}{2}"
    ),
    reference_lines=[
        "sin(P) - sin(R)",
        "sin((P + R)/2 + (P - R)/2) - sin((P + R)/2 - (P - R)/2)",
        "2*cos((P + R)/2)*sin((P - R)/2)",
    ],
    key_steps=_key_steps(
        "the given 2 cos(A)sin(B) = sin(A + B) - sin(A - B) identity",
        "writing P as (P + R)/2 + (P - R)/2 and R as (P + R)/2 - (P - R)/2 before applying it",
    ),
    traps=[
        _trap(
            "mixes_up_half_sum_and_half_difference",
            "Does the response keep (P + R)/2 and (P - R)/2 in the roles A and B, not swapped?",
            [
                "sin(P) - sin(R)",
                "sin((P + R)/2 + (P - R)/2) - sin((P + R)/2 - (P - R)/2)",
                "2*cos((P - R)/2)*sin((P + R)/2)",
                "2*cos((P + R)/2)*sin((P - R)/2)",
            ],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_wrong_side",
            "Does the response start from sin(P) - sin(R), not the product it is asked to derive?",
            [
                "2*cos((P + R)/2)*sin((P - R)/2)",
                "sin((P + R)/2 + (P - R)/2) - sin((P + R)/2 - (P - R)/2)",
                "sin(P) - sin(R)",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response show the half-sum/half-difference substitution, or just assert the product?",
            ["sin(P) - sin(R)", "2*cos((P + R)/2)*sin((P - R)/2)"],
            "passes_chain",
        ),
    ],
)

_COS_SUM_TO_PRODUCT = dict(
    id="cos_sum_to_product",
    tier="hard",
    start="cos(P) + cos(R)",
    target="2*cos((P + R)/2)*cos((P - R)/2)",
    identity_read="cos(P) + cos(R) = 2 cos((P + R)/2)cos((P - R)/2)",
    identity_tex=r"\cos(P) + \cos(R) = 2\cos\!\left(\tfrac{P + R}{2}\right)\cos\!\left(\tfrac{P - R}{2}\right)",
    given_read="2 cos(A)cos(B) = cos(A - B) + cos(A + B), with A = (P + R)/2, B = (P - R)/2",
    given_tex=(
        r"2\cos(A)\cos(B) = \cos(A - B) + \cos(A + B)\text{, with }"
        r"A = \tfrac{P + R}{2},\ B = \tfrac{P - R}{2}"
    ),
    reference_lines=[
        "cos(P) + cos(R)",
        "cos((P + R)/2 + (P - R)/2) + cos((P + R)/2 - (P - R)/2)",
        "2*cos((P + R)/2)*cos((P - R)/2)",
    ],
    key_steps=_key_steps(
        "the given 2 cos(A)cos(B) = cos(A - B) + cos(A + B) identity",
        "writing P as (P + R)/2 + (P - R)/2 and R as (P + R)/2 - (P - R)/2 before applying it",
    ),
    traps=[
        _trap(
            "mixes_up_half_sum_and_half_difference",
            "Does the response keep (P + R)/2 and (P - R)/2 in the roles A and B, not swapped?",
            [
                "cos(P) + cos(R)",
                "cos((P + R)/2 + (P - R)/2) + cos((P + R)/2 - (P - R)/2)",
                "2*cos((P - R)/2)*cos((P + R)/2 + 1)",
                "2*cos((P + R)/2)*cos((P - R)/2)",
            ],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_wrong_side",
            "Does the response start from cos(P) + cos(R), not the product it is asked to derive?",
            [
                "2*cos((P + R)/2)*cos((P - R)/2)",
                "cos((P + R)/2 + (P - R)/2) + cos((P + R)/2 - (P - R)/2)",
                "cos(P) + cos(R)",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response show the half-sum/half-difference substitution, or just assert the product?",
            ["cos(P) + cos(R)", "2*cos((P + R)/2)*cos((P - R)/2)"],
            "passes_chain",
        ),
    ],
)

_COS_DIFF_TO_PRODUCT = dict(
    id="cos_diff_to_product",
    tier="hard",
    start="cos(P) - cos(R)",
    target="-2*sin((P + R)/2)*sin((P - R)/2)",
    identity_read="cos(P) - cos(R) = -2 sin((P + R)/2)sin((P - R)/2)",
    identity_tex=r"\cos(P) - \cos(R) = -2\sin\!\left(\tfrac{P + R}{2}\right)\sin\!\left(\tfrac{P - R}{2}\right)",
    given_read="2 sin(A)sin(B) = cos(A - B) - cos(A + B), with A = (P + R)/2, B = (P - R)/2",
    given_tex=(
        r"2\sin(A)\sin(B) = \cos(A - B) - \cos(A + B)\text{, with }"
        r"A = \tfrac{P + R}{2},\ B = \tfrac{P - R}{2}"
    ),
    reference_lines=[
        "cos(P) - cos(R)",
        "-(cos((P + R)/2 - (P - R)/2) - cos((P + R)/2 + (P - R)/2))",
        "-2*sin((P + R)/2)*sin((P - R)/2)",
    ],
    key_steps=_key_steps(
        "the given 2 sin(A)sin(B) = cos(A - B) - cos(A + B) identity",
        "writing cos(P) - cos(R) as -(cos(R) - cos(P)) and substituting A = (P + R)/2, B = (P - R)/2",
    ),
    traps=[
        _trap(
            "mixes_up_half_sum_and_half_difference",
            "Does the response keep (P + R)/2 and (P - R)/2 in the roles A and B, not swapped?",
            [
                "cos(P) - cos(R)",
                "-(cos((P + R)/2 - (P - R)/2) - cos((P + R)/2 + (P - R)/2))",
                "-2*sin((P - R)/2)*sin((P + R)/2 + 1)",
                "-2*sin((P + R)/2)*sin((P - R)/2)",
            ],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_wrong_side",
            "Does the response start from cos(P) - cos(R), not the product it is asked to derive?",
            [
                "-2*sin((P + R)/2)*sin((P - R)/2)",
                "-(cos((P + R)/2 - (P - R)/2) - cos((P + R)/2 + (P - R)/2))",
                "cos(P) - cos(R)",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response show the half-sum/half-difference substitution, or just assert the product?",
            ["cos(P) - cos(R)", "-2*sin((P + R)/2)*sin((P - R)/2)"],
            "passes_chain",
        ),
    ],
)

_HARD_PROVE_POOL: list[dict] = []
for _fam in (_SIN_SUM_TO_PRODUCT, _SIN_DIFF_TO_PRODUCT, _COS_SUM_TO_PRODUCT, _COS_DIFF_TO_PRODUCT):
    _HARD_PROVE_POOL.extend(_expand_family(_fam, canon=("P", "R")))

for _entry in _HARD_PROVE_POOL:
    _verify_symbolic_chain(_entry["reference_lines"], _entry["id"], letters=_entry["letters"])


# ---------------------------------------------------------------------------
# HARD (ii) -- numeric instances of the sum-to-product forms (e.g.
# sin(5x) + sin(3x) = 2 sin(4x)cos(x)). Only (m, n) pairs (m > n, same
# parity so (m +- n)/2 is an integer multiple of x) whose 3-line reference
# chain was directly confirmed against `symbolically_equal` are kept.
# ---------------------------------------------------------------------------


def _sinsum_hard_entry(m: int, n: int) -> dict:
    a, b = (m + n) // 2, (m - n) // 2
    start = f"sin({m}*x) + sin({n}*x)"
    mid = f"sin(({m}*x + {n}*x)/2 + ({m}*x - {n}*x)/2) + sin(({m}*x + {n}*x)/2 - ({m}*x - {n}*x)/2)"
    target = f"2*sin({a}*x)*cos({b}*x)"
    wrong_target = f"2*sin({b}*x)*cos({a}*x)"
    return dict(
        id=f"sinsum_hard_{m}_{n}",
        tier="hard",
        symbols={"x": "real"},
        start=start,
        target=target,
        start_read=f"sin({_mult_read(m)}) + sin({_mult_read(n)})",
        target_read=f"2 sin({_mult_read(a)})cos({_mult_read(b)})",
        given_read="2 sin(A)cos(B) = sin(A + B) + sin(A - B)",
        given_tex=r"2\sin(A)\cos(B) = \sin(A + B) + \sin(A - B)",
        reference_lines=[start, mid, target],
        key_steps=_key_steps(
            "the given 2 sin(A)cos(B) = sin(A + B) + sin(A - B) identity",
            f"writing {m}x as {a}x + {b}x and {n}x as {a}x - {b}x before applying it",
        ),
        traps=[
            _trap(
                "mixes_up_half_sum_and_half_difference",
                "Does the response keep the half-sum and half-difference multiples in the right order?",
                [start, mid, wrong_target, target],
                "authoritative_failure",
            ),
            _trap(
                "starts_from_wrong_side",
                "Does the response start from the stated sum, not the product it is asked to derive?",
                [target, mid, start],
                "non_authoritative_failure",
            ),
            _trap(
                "cites_the_result_being_proved",
                "Does the response show the half-sum/half-difference substitution, or just assert the product?",
                [start, target],
                "passes_chain",
            ),
        ],
    )


def _sindiff_hard_entry(m: int, n: int) -> dict:
    a, b = (m + n) // 2, (m - n) // 2
    start = f"sin({m}*x) - sin({n}*x)"
    mid = f"sin(({m}*x + {n}*x)/2 + ({m}*x - {n}*x)/2) - sin(({m}*x + {n}*x)/2 - ({m}*x - {n}*x)/2)"
    target = f"2*cos({a}*x)*sin({b}*x)"
    wrong_target = f"2*cos({b}*x)*sin({a}*x)"
    return dict(
        id=f"sindiff_hard_{m}_{n}",
        tier="hard",
        symbols={"x": "real"},
        start=start,
        target=target,
        start_read=f"sin({_mult_read(m)}) - sin({_mult_read(n)})",
        target_read=f"2 cos({_mult_read(a)})sin({_mult_read(b)})",
        given_read="2 cos(A)sin(B) = sin(A + B) - sin(A - B)",
        given_tex=r"2\cos(A)\sin(B) = \sin(A + B) - \sin(A - B)",
        reference_lines=[start, mid, target],
        key_steps=_key_steps(
            "the given 2 cos(A)sin(B) = sin(A + B) - sin(A - B) identity",
            f"writing {m}x as {a}x + {b}x and {n}x as {a}x - {b}x before applying it",
        ),
        traps=[
            _trap(
                "mixes_up_half_sum_and_half_difference",
                "Does the response keep the half-sum and half-difference multiples in the right order?",
                [start, mid, wrong_target, target],
                "authoritative_failure",
            ),
            _trap(
                "starts_from_wrong_side",
                "Does the response start from the stated difference, not the product it is asked to derive?",
                [target, mid, start],
                "non_authoritative_failure",
            ),
            _trap(
                "cites_the_result_being_proved",
                "Does the response show the half-sum/half-difference substitution, or just assert the product?",
                [start, target],
                "passes_chain",
            ),
        ],
    )


def _cossum_hard_entry(m: int, n: int) -> dict:
    a, b = (m + n) // 2, (m - n) // 2
    start = f"cos({m}*x) + cos({n}*x)"
    mid = f"cos(({m}*x + {n}*x)/2 + ({m}*x - {n}*x)/2) + cos(({m}*x + {n}*x)/2 - ({m}*x - {n}*x)/2)"
    target = f"2*cos({a}*x)*cos({b}*x)"
    wrong_target = f"2*sin({a}*x)*sin({b}*x)"
    return dict(
        id=f"cossum_hard_{m}_{n}",
        tier="hard",
        symbols={"x": "real"},
        start=start,
        target=target,
        start_read=f"cos({_mult_read(m)}) + cos({_mult_read(n)})",
        target_read=f"2 cos({_mult_read(a)})cos({_mult_read(b)})",
        given_read="2 cos(A)cos(B) = cos(A - B) + cos(A + B)",
        given_tex=r"2\cos(A)\cos(B) = \cos(A - B) + \cos(A + B)",
        reference_lines=[start, mid, target],
        key_steps=_key_steps(
            "the given 2 cos(A)cos(B) = cos(A - B) + cos(A + B) identity",
            f"writing {m}x as {a}x + {b}x and {n}x as {a}x - {b}x before applying it",
        ),
        traps=[
            _trap(
                "pairs_the_wrong_sum_and_difference",
                "Does the response use cos(...)cos(...), not sin(...)sin(...)?",
                [start, mid, wrong_target, target],
                "authoritative_failure",
            ),
            _trap(
                "starts_from_wrong_side",
                "Does the response start from the stated sum, not the product it is asked to derive?",
                [target, mid, start],
                "non_authoritative_failure",
            ),
            _trap(
                "cites_the_result_being_proved",
                "Does the response show the half-sum/half-difference substitution, or just assert the product?",
                [start, target],
                "passes_chain",
            ),
        ],
    )


def _cosdiff_hard_entry(m: int, n: int) -> dict:
    a, b = (m + n) // 2, (m - n) // 2
    start = f"cos({m}*x) - cos({n}*x)"
    mid = f"-(cos(({m}*x + {n}*x)/2 - ({m}*x - {n}*x)/2) - cos(({m}*x + {n}*x)/2 + ({m}*x - {n}*x)/2))"
    target = f"-2*sin({a}*x)*sin({b}*x)"
    wrong_target = f"2*sin({a}*x)*sin({b}*x)"
    return dict(
        id=f"cosdiff_hard_{m}_{n}",
        tier="hard",
        symbols={"x": "real"},
        start=start,
        target=target,
        start_read=f"cos({_mult_read(m)}) - cos({_mult_read(n)})",
        target_read=f"-2 sin({_mult_read(a)})sin({_mult_read(b)})",
        given_read="2 sin(A)sin(B) = cos(A - B) - cos(A + B)",
        given_tex=r"2\sin(A)\sin(B) = \cos(A - B) - \cos(A + B)",
        reference_lines=[start, mid, target],
        key_steps=_key_steps(
            "the given 2 sin(A)sin(B) = cos(A - B) - cos(A + B) identity",
            f"writing cos({m}x) - cos({n}x) as -(cos({n}x) - cos({m}x)) and substituting",
        ),
        traps=[
            _trap(
                "forgets_the_overall_sign",
                "Does the response keep the leading minus sign from cos(P) - cos(R) = -(cos(R) - cos(P))?",
                [start, mid, wrong_target, target],
                "authoritative_failure",
            ),
            _trap(
                "starts_from_wrong_side",
                "Does the response start from the stated difference, not the product it is asked to derive?",
                [target, mid, start],
                "non_authoritative_failure",
            ),
            _trap(
                "cites_the_result_being_proved",
                "Does the response show the half-sum/half-difference substitution, or just assert the product?",
                [start, target],
                "passes_chain",
            ),
        ],
    )


# Every pair below was checked against `mathgen.core.symbolically_equal` for ALL FOUR families at
# once while writing this module; pairs such as (5, 3), (11, 5) and (9, 1) were tried and rejected
# because at least one family's reference chain came back "could not confirm".
_HARD_NUMERIC_PAIRS = [(3, 1), (7, 3), (9, 5)]

_HARD_NUMERIC_POOL: list[dict] = (
    [_sinsum_hard_entry(m, n) for m, n in _HARD_NUMERIC_PAIRS]
    + [_sindiff_hard_entry(m, n) for m, n in _HARD_NUMERIC_PAIRS]
    + [_cossum_hard_entry(m, n) for m, n in _HARD_NUMERIC_PAIRS]
    + [_cosdiff_hard_entry(m, n) for m, n in _HARD_NUMERIC_PAIRS]
)

for _entry in _HARD_NUMERIC_POOL:
    _verify_single_symbol_chain(_entry["reference_lines"], _entry["id"])

_HARD_POOL: list[dict] = _HARD_PROVE_POOL + _HARD_NUMERIC_POOL


# ---------------------------------------------------------------------------
# EXAM (a) -- evaluate a stated product of special-angle values exactly
# (numeric, no free symbols). Every value below was checked directly with
# the `math` module and against `symbolically_equal` (module
# docstring) while writing this module.
# ---------------------------------------------------------------------------

_EXAM_APPLY_POOL: list[dict] = [
    dict(
        id="exam_apply_sincos_5_1",
        tier="exam",
        symbols={},
        start="sin(5*pi/12)*cos(pi/12)",
        target="(2 + sqrt(3))/4",
        start_read="sin(5pi/12)cos(pi/12)",
        target_read="(2 + sqrt3)/4",
        given_read="2 sin(A)cos(B) = sin(A + B) + sin(A - B), and sin(pi/2) = 1, sin(pi/3) = sqrt(3)/2",
        given_tex=r"2\sin(A)\cos(B) = \sin(A + B) + \sin(A - B)\text{, and }\sin(\tfrac{\pi}{2}) = 1,\ "
        r"\sin(\tfrac{\pi}{3}) = \tfrac{\sqrt{3}}{2}",
        reference_lines=[
            "sin(5*pi/12)*cos(pi/12)",
            "(sin(5*pi/12 + pi/12) + sin(5*pi/12 - pi/12))/2",
            "(sin(pi/2) + sin(pi/3))/2",
            "(2 + sqrt(3))/4",
        ],
        key_steps=_key_steps(
            "the given product-to-sum identity and the standard values at pi/2, pi/3",
            "adding pi/12 to and from 5pi/12 before applying the identity",
        ),
    ),
    dict(
        id="exam_apply_coscos_5_1",
        tier="exam",
        symbols={},
        start="cos(5*pi/12)*cos(pi/12)",
        target="1/4",
        start_read="cos(5pi/12)cos(pi/12)",
        target_read="1/4",
        given_read="2 cos(A)cos(B) = cos(A - B) + cos(A + B), and cos(pi/3) = 1/2, cos(pi/2) = 0",
        given_tex=r"2\cos(A)\cos(B) = \cos(A - B) + \cos(A + B)\text{, and }\cos(\tfrac{\pi}{3}) = \tfrac{1}{2},\ "
        r"\cos(\tfrac{\pi}{2}) = 0",
        reference_lines=[
            "cos(5*pi/12)*cos(pi/12)",
            "(cos(5*pi/12 - pi/12) + cos(5*pi/12 + pi/12))/2",
            "(cos(pi/3) + cos(pi/2))/2",
            "1/4",
        ],
        key_steps=_key_steps(
            "the given product-to-sum identity and the standard values at pi/3, pi/2",
            "subtracting and adding pi/12 to 5pi/12 before applying the identity",
        ),
    ),
    dict(
        id="exam_apply_sinsin_3_1",
        tier="exam",
        symbols={},
        start="sin(3*pi/8)*sin(pi/8)",
        target="sqrt(2)/4",
        start_read="sin(3pi/8)sin(pi/8)",
        target_read="sqrt2/4",
        given_read="2 sin(A)sin(B) = cos(A - B) - cos(A + B), and cos(pi/4) = sqrt(2)/2, cos(pi/2) = 0",
        given_tex=r"2\sin(A)\sin(B) = \cos(A - B) - \cos(A + B)\text{, and }\cos(\tfrac{\pi}{4}) = \tfrac{\sqrt{2}}{2},\ "
        r"\cos(\tfrac{\pi}{2}) = 0",
        reference_lines=[
            "sin(3*pi/8)*sin(pi/8)",
            "(cos(3*pi/8 - pi/8) - cos(3*pi/8 + pi/8))/2",
            "(cos(pi/4) - cos(pi/2))/2",
            "sqrt(2)/4",
        ],
        key_steps=_key_steps(
            "the given product-to-sum identity and the standard values at pi/4, pi/2",
            "subtracting and adding pi/8 to 3pi/8 before applying the identity",
        ),
    ),
    dict(
        id="exam_apply_sincos2_7_1",
        tier="exam",
        symbols={},
        start="2*sin(7*pi/12)*cos(pi/12)",
        target="(sqrt(3) + 2)/2",
        start_read="2 sin(7pi/12)cos(pi/12)",
        target_read="(sqrt3 + 2)/2",
        given_read="2 sin(A)cos(B) = sin(A + B) + sin(A - B), and sin(2pi/3) = sqrt(3)/2, sin(pi/2) = 1",
        given_tex=r"2\sin(A)\cos(B) = \sin(A + B) + \sin(A - B)\text{, and }\sin(\tfrac{2\pi}{3}) = \tfrac{\sqrt{3}}{2},\ "
        r"\sin(\tfrac{\pi}{2}) = 1",
        reference_lines=[
            "2*sin(7*pi/12)*cos(pi/12)",
            "sin(7*pi/12 + pi/12) + sin(7*pi/12 - pi/12)",
            "sin(2*pi/3) + sin(pi/2)",
            "(sqrt(3) + 2)/2",
        ],
        key_steps=_key_steps(
            "the given product-to-sum identity and the standard values at 2pi/3, pi/2",
            "adding pi/12 to and from 7pi/12 before applying the identity",
        ),
    ),
    dict(
        id="exam_apply_sincos_1_5",
        tier="exam",
        symbols={},
        start="sin(pi/12)*cos(5*pi/12)",
        target="(2 - sqrt(3))/4",
        start_read="sin(pi/12)cos(5pi/12)",
        target_read="(2 - sqrt3)/4",
        given_read=(
            "2 sin(A)cos(B) = sin(A + B) + sin(A - B), and sin(pi/2) = 1, sin(-pi/3) = -sqrt(3)/2 (sine is odd)"
        ),
        given_tex=r"2\sin(A)\cos(B) = \sin(A + B) + \sin(A - B)\text{, and }\sin(\tfrac{\pi}{2}) = 1,\ "
        r"\sin(-\tfrac{\pi}{3}) = -\tfrac{\sqrt{3}}{2}\text{ (sine is odd)}",
        reference_lines=[
            "sin(pi/12)*cos(5*pi/12)",
            "(sin(pi/12 + 5*pi/12) + sin(pi/12 - 5*pi/12))/2",
            "(sin(pi/2) + sin(-pi/3))/2",
            "(2 - sqrt(3))/4",
        ],
        key_steps=_key_steps(
            "the given product-to-sum identity and the odd symmetry of sine",
            "adding pi/12 to and subtracting 5pi/12 from pi/12 before applying the identity",
        ),
    ),
    dict(
        id="exam_apply_cossin_1_3",
        tier="exam",
        symbols={},
        start="cos(pi/8)*sin(3*pi/8)",
        target="(2 + sqrt(2))/4",
        start_read="cos(pi/8)sin(3pi/8)",
        target_read="(2 + sqrt2)/4",
        given_read=(
            "2 cos(A)sin(B) = sin(A + B) - sin(A - B), and sin(pi/2) = 1, sin(-pi/4) = -sqrt(2)/2 (sine is odd)"
        ),
        given_tex=r"2\cos(A)\sin(B) = \sin(A + B) - \sin(A - B)\text{, and }\sin(\tfrac{\pi}{2}) = 1,\ "
        r"\sin(-\tfrac{\pi}{4}) = -\tfrac{\sqrt{2}}{2}\text{ (sine is odd)}",
        reference_lines=[
            "cos(pi/8)*sin(3*pi/8)",
            "(sin(pi/8 + 3*pi/8) - sin(pi/8 - 3*pi/8))/2",
            "(sin(pi/2) - sin(-pi/4))/2",
            "(2 + sqrt(2))/4",
        ],
        key_steps=_key_steps(
            "the given product-to-sum identity and the odd symmetry of sine",
            "adding and subtracting pi/8 and 3pi/8 before applying the identity",
        ),
    ),
]

for _entry in _EXAM_APPLY_POOL:
    _entry["traps"] = [
        _trap(
            "wrong_special_value",
            "Does the response use the CORRECT standard value at each reference angle, not a nearby one?",
            [_entry["start"], "0"],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response show the decomposition and standard values, or just assert the exact value?",
            [_entry["start"], _entry["target"]],
            "passes_chain",
        ),
    ]

for _entry in _EXAM_APPLY_POOL:
    _verify_numeric_chain(_entry["reference_lines"], _entry["id"])


# ---------------------------------------------------------------------------
# EXAM (b) -- a composite identity needing a sum-to-product step TWICE
# (numerator and denominator), then cancelling. tan is a CAS risk; both chains below were checked directly against
# `symbolically_equal` while writing this module (module docstring / CAS
# table). Domain must be stated: both denominators must be
# nonzero.
# ---------------------------------------------------------------------------

_EXAM_COMPOSITE_POOL: list[dict] = [
    dict(
        id="composite_tan2x",
        tier="exam",
        symbols={"x": "real"},
        start="(sin(3*x) + sin(x))/(cos(3*x) + cos(x))",
        target="tan(2*x)",
        start_read="(sin(3x) + sin(x))/(cos(3x) + cos(x))",
        target_read="tan(2x)",
        identity_read="(sin(3x) + sin(x))/(cos(3x) + cos(x)) = tan(2x)",
        identity_tex=r"\frac{\sin(3x) + \sin(x)}{\cos(3x) + \cos(x)} = \tan(2x)",
        given_read=(
            "sin(P) + sin(R) = 2 sin((P + R)/2)cos((P - R)/2) and "
            "cos(P) + cos(R) = 2 cos((P + R)/2)cos((P - R)/2), valid where cos(2x) and cos(x) are nonzero"
        ),
        given_tex=(
            r"\sin(P) + \sin(R) = 2\sin(\tfrac{P+R}{2})\cos(\tfrac{P-R}{2})\text{ and }"
            r"\cos(P) + \cos(R) = 2\cos(\tfrac{P+R}{2})\cos(\tfrac{P-R}{2})\text{, valid where }"
            r"\cos(2x)\text{ and }\cos(x)\text{ are nonzero}"
        ),
        reference_lines=[
            "(sin(3*x) + sin(x))/(cos(3*x) + cos(x))",
            "(2*sin(2*x)*cos(x))/(2*cos(2*x)*cos(x))",
            "sin(2*x)/cos(2*x)",
            "tan(2*x)",
        ],
        key_steps=_key_steps(
            "the given sin(P) + sin(R) and cos(P) + cos(R) sum-to-product identities",
            "applying both with P = 3x, R = x, then cancelling the common factor 2cos(x) (nonzero)",
        ),
        traps=[
            _trap(
                "mixes_up_the_two_sum_to_product_forms",
                "Does the response apply the sin-sum formula to the numerator and the cos-sum formula to the "
                "denominator, not the same one to both?",
                ["(sin(3*x) + sin(x))/(cos(3*x) + cos(x))", "(2*sin(2*x)*cos(x))/(2*sin(2*x)*cos(x))", "tan(2*x)"],
                "authoritative_failure",
            ),
            _trap(
                "starts_from_wrong_side",
                "Does the response start from the stated ratio, not tan(2x)?",
                [
                    "tan(2*x)",
                    "sin(2*x)/cos(2*x)",
                    "(2*sin(2*x)*cos(x))/(2*cos(2*x)*cos(x))",
                    "(sin(3*x) + sin(x))/(cos(3*x) + cos(x))",
                ],
                "non_authoritative_failure",
            ),
            _trap(
                "cites_the_result_being_proved",
                "Does the response show both sum-to-product steps and the cancellation, or just assert tan(2x)?",
                [
                    "(sin(3*x) + sin(x))/(cos(3*x) + cos(x))",
                    "(2*sin(2*x)*cos(x))/(2*cos(2*x)*cos(x))",
                    "sin(2*x)/cos(2*x)",
                    "tan(2*x)",
                ],
                "passes_chain",
            ),
            _trap(
                "divides_without_domain_check",
                "Does the response state that cos(2x) and cos(x) are nonzero before cancelling and simplifying?",
                [
                    "(sin(3*x) + sin(x))/(cos(3*x) + cos(x))",
                    "(2*sin(2*x)*cos(x))/(2*cos(2*x)*cos(x))",
                    "sin(2*x)/cos(2*x)",
                    "tan(2*x)",
                ],
                "passes_chain",
            ),
        ],
    ),
    dict(
        id="composite_tanx",
        tier="exam",
        symbols={"x": "real"},
        start="(sin(5*x) - sin(3*x))/(cos(5*x) + cos(3*x))",
        target="tan(x)",
        start_read="(sin(5x) - sin(3x))/(cos(5x) + cos(3x))",
        target_read="tan(x)",
        identity_read="(sin(5x) - sin(3x))/(cos(5x) + cos(3x)) = tan(x)",
        identity_tex=r"\frac{\sin(5x) - \sin(3x)}{\cos(5x) + \cos(3x)} = \tan(x)",
        given_read=(
            "sin(P) - sin(R) = 2 cos((P + R)/2)sin((P - R)/2) and "
            "cos(P) + cos(R) = 2 cos((P + R)/2)cos((P - R)/2), valid where cos(4x) and cos(x) are nonzero"
        ),
        given_tex=(
            r"\sin(P) - \sin(R) = 2\cos(\tfrac{P+R}{2})\sin(\tfrac{P-R}{2})\text{ and }"
            r"\cos(P) + \cos(R) = 2\cos(\tfrac{P+R}{2})\cos(\tfrac{P-R}{2})\text{, valid where }"
            r"\cos(4x)\text{ and }\cos(x)\text{ are nonzero}"
        ),
        reference_lines=[
            "(sin(5*x) - sin(3*x))/(cos(5*x) + cos(3*x))",
            "(2*cos(4*x)*sin(x))/(2*cos(4*x)*cos(x))",
            "sin(x)/cos(x)",
            "tan(x)",
        ],
        key_steps=_key_steps(
            "the given sin(P) - sin(R) and cos(P) + cos(R) sum-to-product identities",
            "applying both with P = 5x, R = 3x, then cancelling the common factor 2cos(4x) (nonzero)",
        ),
        traps=[
            _trap(
                "mixes_up_the_two_sum_to_product_forms",
                "Does the response apply the sin-difference formula to the numerator and the cos-sum formula "
                "to the denominator, not the same one to both?",
                ["(sin(5*x) - sin(3*x))/(cos(5*x) + cos(3*x))", "(2*cos(4*x)*sin(x))/(2*cos(4*x)*sin(x))", "tan(x)"],
                "authoritative_failure",
            ),
            _trap(
                "starts_from_wrong_side",
                "Does the response start from the stated ratio, not tan(x)?",
                [
                    "tan(x)",
                    "sin(x)/cos(x)",
                    "(2*cos(4*x)*sin(x))/(2*cos(4*x)*cos(x))",
                    "(sin(5*x) - sin(3*x))/(cos(5*x) + cos(3*x))",
                ],
                "non_authoritative_failure",
            ),
            _trap(
                "cites_the_result_being_proved",
                "Does the response show both sum-to-product steps and the cancellation, or just assert tan(x)?",
                [
                    "(sin(5*x) - sin(3*x))/(cos(5*x) + cos(3*x))",
                    "(2*cos(4*x)*sin(x))/(2*cos(4*x)*cos(x))",
                    "sin(x)/cos(x)",
                    "tan(x)",
                ],
                "passes_chain",
            ),
            _trap(
                "divides_without_domain_check",
                "Does the response state that cos(4x) and cos(x) are nonzero before cancelling and simplifying?",
                [
                    "(sin(5*x) - sin(3*x))/(cos(5*x) + cos(3*x))",
                    "(2*cos(4*x)*sin(x))/(2*cos(4*x)*cos(x))",
                    "sin(x)/cos(x)",
                    "tan(x)",
                ],
                "passes_chain",
            ),
        ],
    ),
]

for _entry in _EXAM_COMPOSITE_POOL:
    _verify_single_symbol_chain(_entry["reference_lines"], _entry["id"])

_EXAM_POOL: list[dict] = _EXAM_APPLY_POOL + _EXAM_COMPOSITE_POOL


# ---------------------------------------------------------------------------
# Prompt wording. `.format()` for the text side; the latex side uses
# `%TOKEN%` markers with `.replace()`, because the latex fragments contain
# literal `{`/`}` from `\text{...}`/`\frac{...}{...}` that `.format()` would
# misread. The notation sentence is always appended (both text and
# latex) after the main template.
# ---------------------------------------------------------------------------

_PROVE_TEMPLATES = [
    (
        "Prove that {identity}. You are given {given}. Start from the left-hand side, {start}, and finish at {target}.",
        r"\text{Prove that }%IDENTITY%.\text{ You are given }%GIVEN%"
        r".\text{ Start from the left-hand side, }%START%,\text{ and finish at }%TARGET%.",
    ),
    (
        "Using {given}, prove that {identity}. Begin your proof at {start} and end at {target}.",
        r"\text{Using }%GIVEN%,\text{ prove that }%IDENTITY%"
        r".\text{ Begin your proof at }%START%\text{ and end at }%TARGET%.",
    ),
    (
        "Show that {identity} follows from {given}. Write a chain of equalities starting at "
        "{start} and ending at {target}.",
        r"\text{Show that }%IDENTITY%\text{ follows from }%GIVEN%"
        r".\text{ Write a chain of equalities starting at }%START%\text{ and ending at }%TARGET%.",
    ),
    (
        "Give a complete proof of {identity}, citing {given} at the step where each is used. "
        "Start at {start}; end at {target}.",
        r"\text{Give a complete proof of }%IDENTITY%,\text{ citing }%GIVEN%"
        r"\text{ at the step where each is used. Start at }%START%;\text{ end at }%TARGET%.",
    ),
]

_APPLY_TEMPLATES = [
    (
        "Express {start} as a sum or difference, using {given}.",
        r"\text{Express }%START%\text{ as a sum or difference, using }%GIVEN%.",
    ),
    (
        "Write {start} as a sum or difference of two terms, citing {given} at the step where it is used.",
        r"\text{Write }%START%\text{ as a sum or difference of two terms, citing }%GIVEN%"
        r"\text{ at the step where it is used.}",
    ),
    (
        "Show how {start} can be rewritten as a sum or difference, using {given}.",
        r"\text{Show how }%START%\text{ can be rewritten as a sum or difference, using }%GIVEN%.",
    ),
]

_EXAM_APPLY_TEMPLATES = [
    (
        "Find the exact value of {start}. Use {given}.",
        r"\text{Find the exact value of }%START%.\text{ Use }%GIVEN%.",
    ),
    (
        "Evaluate {start} exactly, citing {given} at the step where it is used.",
        r"\text{Evaluate }%START%\text{ exactly, citing }%GIVEN%\text{ at the step where it is used.}",
    ),
    (
        "Give the exact value of {start}, using {given}.",
        r"\text{Give the exact value of }%START%,\text{ using }%GIVEN%.",
    ),
]

_NOTATION_SENTENCES = [
    (
        "Write one expression per line; put brackets round every angle, as in sin(x), and "
        "use ^ for powers, as in cos(x)^2.",
        r"\text{Write one expression per line; put brackets round every angle, as in }\sin(x)"
        r"\text{, and use \textasciicircum{} for powers, as in }\cos(x)^2\text{.}",
    ),
    (
        "Type one line per step; bracket every angle (for example sin(x)) and use ^ for powers (for example cos(x)^2).",
        r"\text{Type one line per step; bracket every angle (for example }\sin(x)"
        r"\text{) and use \textasciicircum{} for powers (for example }\cos(x)^2\text{).}",
    ),
]


def _notation_sentence(rng: random.Random) -> tuple[str, str]:
    return _NOTATION_SENTENCES[rng.randrange(len(_NOTATION_SENTENCES))]


def _fill_tex(template: str, **slots) -> str:
    out = template
    for key, value in slots.items():
        if value is not None:
            out = out.replace(f"%{key.upper()}%", value)
    return out


def _split_identity(identity_read: str) -> tuple[str, str]:
    left, right = identity_read.split(" = ", 1)
    return left, right


def _read_to_tex(s: str) -> str:
    """Small hand rule, not a general parser: every `*_read` string this
    module builds uses only sin/cos/tan/sqrt/pi of x, A, B, P, R and
    +-/() -- so a fixed set of textual substitutions gets real LaTeX
    (never a bare identifier run KaTeX would render as italic maths
    letters) without pulling in sympy's own printer."""
    out = re.sub(r"sqrt(\d+)", r"\\sqrt{\1}", s)
    out = re.sub(r"\b(sin|cos|tan)\(", r"\\\1(", out)
    out = re.sub(r"(?<![A-Za-z\\])pi\b", r"\\pi", out)
    return out


def _fmt_prove(entry: dict, rng: random.Random) -> tuple[str, str]:
    idx = rng.randrange(len(_PROVE_TEMPLATES))
    text_t, latex_t = _PROVE_TEMPLATES[idx]
    start_read, target_read = _split_identity(entry["identity_read"])
    start_tex, target_tex = _split_identity(entry["identity_tex"])
    notation_text, notation_tex = _notation_sentence(rng)
    text = (
        text_t.format(identity=entry["identity_read"], given=entry["given_read"], start=start_read, target=target_read)
        + " "
        + notation_text
    )
    latex_str = (
        _fill_tex(latex_t, identity=entry["identity_tex"], given=entry["given_tex"], start=start_tex, target=target_tex)
        + " "
        + notation_tex
    )
    return text, latex_str


def _fmt_apply(entry: dict, rng: random.Random, templates=_APPLY_TEMPLATES) -> tuple[str, str]:
    idx = rng.randrange(len(templates))
    text_t, latex_t = templates[idx]
    notation_text, notation_tex = _notation_sentence(rng)
    text = (
        text_t.format(start=entry["start_read"], given=entry["given_read"], target=entry["target_read"])
        + " "
        + notation_text
    )
    latex_str = (
        _fill_tex(
            latex_t,
            start=_read_to_tex(entry["start_read"]),
            given=entry["given_tex"],
            target=_read_to_tex(entry["target_read"]),
        )
        + " "
        + notation_tex
    )
    return text, latex_str


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


def _prove_question(entry: dict, rng: random.Random) -> dict:
    text, latex_str = _fmt_prove(entry, rng)
    marking = _marking(
        entry["start"], entry["target"], entry["symbols"], entry["reference_lines"], entry["key_steps"], entry["traps"]
    )
    return dict(text=text, latex=latex_str, answer=marking, answer_text=entry["identity_read"], distractors=[])


def _apply_question(entry: dict, rng: random.Random, templates=_APPLY_TEMPLATES) -> dict:
    text, latex_str = _fmt_apply(entry, rng, templates)
    marking = _marking(
        entry["start"], entry["target"], entry["symbols"], entry["reference_lines"], entry["key_steps"], entry["traps"]
    )
    answer_text = f"{entry['start_read']} = {entry['target_read']}"
    return dict(text=text, latex=latex_str, answer=marking, answer_text=answer_text, distractors=[])


@generator(
    "pr1a_product_to_sum_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def pr1a_product_to_sum_proof(rng: random.Random, variant: str = "easy") -> dict:
    """ACMSM047: prove one of the four product-to-sum identities from the (cited, never re-proved)
    angle-sum/difference formulas (easy), apply one to a numeric-multiple product (medium), prove and apply
    its sum-to-product sibling (hard), or evaluate a special-angle product exactly / prove a composite tan
    identity (exam). No distractors: the answer is a rubric marking dict, never a plain value."""
    if variant == "easy":
        entry = rng.choice(_EASY_POOL)
        return _prove_question(entry, rng)

    if variant == "medium":
        entry = rng.choice(_MEDIUM_POOL)
        return _apply_question(entry, rng)

    if variant == "hard":
        if rng.random() < 0.5:
            entry = rng.choice(_HARD_PROVE_POOL)
            return _prove_question(entry, rng)
        entry = rng.choice(_HARD_NUMERIC_POOL)
        return _apply_question(entry, rng)

    if variant == "exam":
        if rng.random() < 0.5:
            entry = rng.choice(_EXAM_APPLY_POOL)
            return _apply_question(entry, rng, templates=_EXAM_APPLY_TEMPLATES)
        entry = rng.choice(_EXAM_COMPOSITE_POOL)
        return _prove_question(entry, rng)

    raise ValueError(f"unknown variant: {variant}")
