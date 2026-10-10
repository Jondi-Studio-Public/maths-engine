"""One topic, ``pr1a_angle_sum_difference_proof``, four presets
(easy/medium/hard/exam). Every item is either a PROOF of one of
sin(A +/- B), cos(A +/- B), tan(A +/- B) as a chain from stated givens, or
-- easy tier only -- an APPLY item: an exact value of sin/cos of a special
angle found by writing it as a sum/difference of known reference angles.

This skill NEVER asks for a double-angle identity (ACMSM044's territory) and
never a product-to-sum one (ACMSM047).

Cheap in-generator verification only: every reference chain here
is a FIXED pool entry (finite draw space), so each is checked once, at
import time, by evaluating every line with the `math` module at a small
set of fixed-but-irregular real points (symbolic entries) or once directly
(apply entries, which carry no free symbol). A full CAS check of each chain
is not done here: it costs 0.1-0.4s per chain and would blow the generator
speed budget.

1) fixed three defects in the original build:

1. Pool too thin -- every proof identity is now expanded over 5 Latin
   letter pairs (the safe list: (A,B), (P,R), (x,t), (u,v), (x,u)),
   and easy gained a second one-substitution proof (sin(A - B) from the
   given sin(A + B)), via `_expand_family`/`_LETTER_PAIRS` below.
2. Notation -- every learner-facing identity/given/start/end line now
   brackets every function argument (`cos(A)cos(B)`, never `cos A cos B`),
   and every prompt carries a short notation sentence.
3. Domain -- the tan family's given clause now states its domain (cos A,
   cos B and cos(A +/- B) nonzero) instead of ever implying "for all real".
"""

from __future__ import annotations

import math
import random
import re

from ..core import check, generator, phrase

# ---------------------------------------------------------------------------
# Cheap numeric verification helpers -- `math`-only, no sympy here.
# ---------------------------------------------------------------------------

# Three fixed-but-irregular real-radian points avoiding every singularity this pool's chains divide
# by (cos of either letter, cos of their sum, cos of their difference, 1 -/+ tan*tan) -- confirmed
# clear of all six by direct computation before being fixed here. The letter NAMES a chain uses vary
# (the 5 pairs); the two numeric slots below are bound to whichever pair a given entry names.
_PROBE_AB = ((0.4, 0.7), (1.0, -0.5), (2.3, 0.9))

_EVAL_NS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "sqrt": math.sqrt}


def _eval_line(line: str, values: dict[str, float] | None = None) -> float:
    ns = dict(_EVAL_NS)
    if values:
        ns.update(values)
    return eval(line, {"__builtins__": {}}, ns)  # noqa: S307 -- fixed internal strings only


def _verify_symbolic_chain(reference_lines: list[str], label: str, letters: tuple[str, str] = ("A", "B")) -> None:
    """Every consecutive pair of `reference_lines` agrees at all 3 probe
    points -- a cheap numeric stand-in for a full CAS chain check.
    `letters` names which two free symbols the
    chain actually uses (varies per `_expand_family` letter-pair variant)."""
    l1, l2 = letters
    for a, b in _PROBE_AB:
        values = [_eval_line(line, {l1: a, l2: b}) for line in reference_lines]
        for i in range(len(values) - 1):
            check(
                math.isclose(values[i], values[i + 1], rel_tol=1e-9, abs_tol=1e-9),
                f"{label}: reference line {i + 1} does not numerically match line {i}",
            )


def _verify_numeric_chain(reference_lines: list[str], label: str) -> None:
    values = [_eval_line(line) for line in reference_lines]
    for i in range(len(values) - 1):
        check(
            math.isclose(values[i], values[i + 1], rel_tol=1e-9, abs_tol=1e-9),
            f"{label}: reference line {i + 1} does not numerically match line {i}",
        )


# ---------------------------------------------------------------------------
# Shared marking-dict / key-step / trap builders
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


# Letter-pair variants. Every proof identity below is written once, in canonical A/B form, then
# expanded over 5 Latin letter pairs (the safe list) so the pool has real variety instead of
# one fixed pair repeated at every seed. The `cos_diff_derivation` entry's unit-circle points are
# always named M/N (never A/B/P/R/x/t/u/v), so a pair such as (P, R) never collides with a point
# label.

_LETTER_PAIRS = [("A", "B", "ab"), ("P", "R", "pr"), ("x", "t", "xt"), ("u", "v", "uv"), ("x", "u", "xu")]

_PLACEHOLDER_A = ""
_PLACEHOLDER_B = ""
_TOKEN_RE_A = re.compile(r"(?<![A-Za-z0-9_])A(?![A-Za-z0-9_])")
_TOKEN_RE_B = re.compile(r"(?<![A-Za-z0-9_])B(?![A-Za-z0-9_])")


def _sub_letters(text: str, l1: str, l2: str) -> str:
    text = _TOKEN_RE_A.sub(_PLACEHOLDER_A, text)
    text = _TOKEN_RE_B.sub(_PLACEHOLDER_B, text)
    return text.replace(_PLACEHOLDER_A, l1).replace(_PLACEHOLDER_B, l2)


def _subst_deep(obj, l1: str, l2: str):
    if isinstance(obj, str):
        return _sub_letters(obj, l1, l2)
    if isinstance(obj, list):
        return [_subst_deep(item, l1, l2) for item in obj]
    if isinstance(obj, dict):
        return {k: _subst_deep(v, l1, l2) for k, v in obj.items()}
    return obj


def _expand_family(entry: dict) -> list[dict]:
    variants = []
    for l1, l2, tag in _LETTER_PAIRS:
        variant = _subst_deep(entry, l1, l2)
        variant["id"] = f"{entry['id']}_{tag}"
        variant["symbols"] = {l1: "real", l2: "real"}
        variant["letters"] = (l1, l2)
        variants.append(variant)
    return variants


# PROVE pool -- one entry per identity family, canonical A/B form.

_COS_SUM = dict(
    id="cos_sum",
    tier="easy",
    start="cos(A + B)",
    target="cos(A)*cos(B) - sin(A)*sin(B)",
    identity_read="cos(A + B) = cos(A)cos(B) - sin(A)sin(B)",
    identity_tex=r"\cos(A + B) = \cos(A)\cos(B) - \sin(A)\sin(B)",
    given_read=("cos(A - B) = cos(A)cos(B) + sin(A)sin(B), and cos(-angle) = cos(angle), sin(-angle) = -sin(angle)"),
    given_tex=(
        r"\cos(A - B) = \cos(A)\cos(B) + \sin(A)\sin(B)\text{, and }"
        r"\cos(-\text{angle}) = \cos(\text{angle}),\ \sin(-\text{angle}) = -\sin(\text{angle})"
    ),
    reference_lines=[
        "cos(A + B)",
        "cos(A - (-B))",
        "cos(A)*cos(-B) + sin(A)*sin(-B)",
        "cos(A)*cos(B) - sin(A)*sin(B)",
    ],
    key_steps=_key_steps(
        "the given cos(A - B) formula and the even/odd facts", "replacing B by -B in the given cos(A - B) formula"
    ),
    traps=[
        _trap(
            "sign_slip",
            "Does the chain ever assert cos(A + B) = cos A cos B + sin A sin B (the cos(A - B) sign) instead?",
            ["cos(A + B)", "cos(A)*cos(B) + sin(A)*sin(B)", "cos(A)*cos(B) - sin(A)*sin(B)"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_rhs",
            "Does the proof actually start from the left-hand side cos(A + B), not the target?",
            ["cos(A)*cos(B) - sin(A)*sin(B)", "cos(A)*cos(-B) + sin(A)*sin(-B)", "cos(A - (-B))", "cos(A + B)"],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response show the derivation, or just assert the target as the very first step after the start?",
            ["cos(A + B)", "cos(A)*cos(B) - sin(A)*sin(B)"],
            "passes_chain",
        ),
    ],
)

_SIN_DIFF_EASY = dict(
    id="sin_diff_easy",
    tier="easy",
    start="sin(A - B)",
    target="sin(A)*cos(B) - cos(A)*sin(B)",
    identity_read="sin(A - B) = sin(A)cos(B) - cos(A)sin(B)",
    identity_tex=r"\sin(A - B) = \sin(A)\cos(B) - \cos(A)\sin(B)",
    given_read=("sin(A + B) = sin(A)cos(B) + cos(A)sin(B), and cos(-angle) = cos(angle), sin(-angle) = -sin(angle)"),
    given_tex=(
        r"\sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B)\text{, and }"
        r"\cos(-\text{angle}) = \cos(\text{angle}),\ \sin(-\text{angle}) = -\sin(\text{angle})"
    ),
    reference_lines=[
        "sin(A - B)",
        "sin(A + (-B))",
        "sin(A)*cos(-B) + cos(A)*sin(-B)",
        "sin(A)*cos(B) - cos(A)*sin(B)",
    ],
    key_steps=_key_steps(
        "the given sin(A + B) formula and the even/odd facts", "replacing B by -B in the given sin(A + B) formula"
    ),
    traps=[
        _trap(
            "sign_slip",
            "Does the chain ever assert sin(A - B) = sin A cos B + cos A sin B (the sin(A + B) sign) instead?",
            ["sin(A - B)", "sin(A)*cos(B) + cos(A)*sin(B)", "sin(A)*cos(B) - cos(A)*sin(B)"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_rhs",
            "Does the proof actually start from the left-hand side sin(A - B), not the target?",
            ["sin(A)*cos(B) - cos(A)*sin(B)", "sin(A)*cos(-B) + cos(A)*sin(-B)", "sin(A + (-B))", "sin(A - B)"],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response show the derivation, or just assert the target as the very first step after the start?",
            ["sin(A - B)", "sin(A)*cos(B) - cos(A)*sin(B)"],
            "passes_chain",
        ),
    ],
)

_SIN_SUM = dict(
    id="sin_sum",
    tier="medium",
    start="sin(A + B)",
    target="sin(A)*cos(B) + cos(A)*sin(B)",
    identity_read="sin(A + B) = sin(A)cos(B) + cos(A)sin(B)",
    identity_tex=r"\sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B)",
    given_read=(
        "cos(A - B) = cos(A)cos(B) + sin(A)sin(B), and the co-function identities "
        "sin(angle) = cos(pi/2 - angle), cos(angle) = sin(pi/2 - angle)"
    ),
    given_tex=(
        r"\cos(A - B) = \cos(A)\cos(B) + \sin(A)\sin(B)\text{, and the co-function identities }"
        r"\sin(\text{angle}) = \cos(\tfrac{\pi}{2} - \text{angle}),\ "
        r"\cos(\text{angle}) = \sin(\tfrac{\pi}{2} - \text{angle})"
    ),
    reference_lines=[
        "sin(A + B)",
        "cos(pi/2 - (A + B))",
        "cos((pi/2 - A) - B)",
        "cos(pi/2 - A)*cos(B) + sin(pi/2 - A)*sin(B)",
        "sin(A)*cos(B) + cos(A)*sin(B)",
    ],
    key_steps=_key_steps(
        "the co-function identity and the given cos(A - B) formula",
        "rewriting sin(A + B) as cos(pi/2 - (A + B)) before applying cos(A - B)",
    ),
    traps=[
        _trap(
            "sum_distributes",
            "Does the chain ever assert sin(A + B) = sin A + sin B?",
            ["sin(A + B)", "sin(A) + sin(B)", "sin(A)*cos(B) + cos(A)*sin(B)"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_rhs",
            "Does the proof start from sin(A + B) itself, not the target?",
            [
                "sin(A)*cos(B) + cos(A)*sin(B)",
                "cos(pi/2 - A)*cos(B) + sin(pi/2 - A)*sin(B)",
                "cos((pi/2 - A) - B)",
                "cos(pi/2 - (A + B))",
                "sin(A + B)",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response derive the result, or quote it as the first step after the start?",
            ["sin(A + B)", "sin(A)*cos(B) + cos(A)*sin(B)"],
            "passes_chain",
        ),
    ],
)

_SIN_DIFF = dict(
    id="sin_diff",
    tier="medium",
    start="sin(A - B)",
    target="sin(A)*cos(B) - cos(A)*sin(B)",
    identity_read="sin(A - B) = sin(A)cos(B) - cos(A)sin(B)",
    identity_tex=r"\sin(A - B) = \sin(A)\cos(B) - \cos(A)\sin(B)",
    given_read=(
        "cos(A - B) = cos(A)cos(B) + sin(A)sin(B), and the co-function identities "
        "sin(angle) = cos(pi/2 - angle), cos(angle) = sin(pi/2 - angle)"
    ),
    given_tex=(
        r"\cos(A - B) = \cos(A)\cos(B) + \sin(A)\sin(B)\text{, and the co-function identities }"
        r"\sin(\text{angle}) = \cos(\tfrac{\pi}{2} - \text{angle}),\ "
        r"\cos(\text{angle}) = \sin(\tfrac{\pi}{2} - \text{angle})"
    ),
    reference_lines=[
        "sin(A - B)",
        "cos(pi/2 - (A - B))",
        "cos((pi/2 - A) + B)",
        "cos((pi/2 - A) - (-B))",
        "cos(pi/2 - A)*cos(-B) + sin(pi/2 - A)*sin(-B)",
        "sin(A)*cos(B) - cos(A)*sin(B)",
    ],
    key_steps=_key_steps(
        "the co-function identity and the given cos(A - B) formula",
        "rewriting sin(A - B) as cos(pi/2 - (A - B)) before applying cos(A - B) with B replaced by -B",
    ),
    traps=[
        _trap(
            "sign_slip",
            "Does the chain ever assert sin(A - B) = sin A cos B + cos A sin B (the sin(A + B) sign)?",
            ["sin(A - B)", "sin(A)*cos(B) + cos(A)*sin(B)", "sin(A)*cos(B) - cos(A)*sin(B)"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_rhs",
            "Does the proof start from sin(A - B) itself, not the target?",
            [
                "sin(A)*cos(B) - cos(A)*sin(B)",
                "cos(pi/2 - A)*cos(-B) + sin(pi/2 - A)*sin(-B)",
                "cos((pi/2 - A) - (-B))",
                "cos((pi/2 - A) + B)",
                "cos(pi/2 - (A - B))",
                "sin(A - B)",
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response derive the result, or quote it as the first step after the start?",
            ["sin(A - B)", "sin(A)*cos(B) - cos(A)*sin(B)"],
            "passes_chain",
        ),
    ],
)

_TAN_SUM = dict(
    id="tan_sum",
    tier="hard",
    start="tan(A + B)",
    target="(tan(A) + tan(B))/(1 - tan(A)*tan(B))",
    identity_read="tan(A + B) = (tan(A) + tan(B))/(1 - tan(A)tan(B))",
    identity_tex=r"\tan(A + B) = \frac{\tan(A) + \tan(B)}{1 - \tan(A)\tan(B)}",
    given_read=(
        "sin(A + B) = sin(A)cos(B) + cos(A)sin(B) and cos(A + B) = cos(A)cos(B) - sin(A)sin(B), "
        "valid where cos(A), cos(B) and cos(A + B) are all nonzero"
    ),
    given_tex=(
        r"\sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B)\text{ and }"
        r"\cos(A + B) = \cos(A)\cos(B) - \sin(A)\sin(B)\text{, valid where }\cos(A)"
        r"\text{, }\cos(B)\text{ and }\cos(A + B)\text{ are all nonzero}"
    ),
    reference_lines=[
        "tan(A + B)",
        "sin(A + B)/cos(A + B)",
        "(sin(A)*cos(B) + cos(A)*sin(B))/(cos(A)*cos(B) - sin(A)*sin(B))",
        "(sin(A)*cos(B)/(cos(A)*cos(B)) + cos(A)*sin(B)/(cos(A)*cos(B)))/(1 - sin(A)*sin(B)/(cos(A)*cos(B)))",
        "(tan(A) + sin(B)/cos(B))/(1 - tan(A)*sin(B)/cos(B))",
        "(tan(A) + tan(B))/(1 - tan(A)*tan(B))",
    ],
    key_steps=_key_steps(
        "the given sin(A + B) and cos(A + B) sum formulas",
        "dividing numerator and denominator by cos A cos B (valid since cos A cos B != 0)",
    ),
    traps=[
        _trap(
            "sum_distributes",
            "Does the chain ever assert tan(A + B) = tan A + tan B?",
            ["tan(A + B)", "tan(A) + tan(B)", "(tan(A) + tan(B))/(1 - tan(A)*tan(B))"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_rhs",
            "Does the proof start from tan(A + B) itself, not the target?",
            list(
                reversed(
                    [
                        "tan(A + B)",
                        "sin(A + B)/cos(A + B)",
                        "(sin(A)*cos(B) + cos(A)*sin(B))/(cos(A)*cos(B) - sin(A)*sin(B))",
                        "(sin(A)*cos(B)/(cos(A)*cos(B)) + cos(A)*sin(B)/(cos(A)*cos(B)))/(1 - sin(A)*sin(B)/(cos(A)*cos(B)))",
                        "(tan(A) + sin(B)/cos(B))/(1 - tan(A)*sin(B)/cos(B))",
                        "(tan(A) + tan(B))/(1 - tan(A)*tan(B))",
                    ]
                )
            ),
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response derive the result, or quote it as the first step after the start?",
            ["tan(A + B)", "(tan(A) + tan(B))/(1 - tan(A)*tan(B))"],
            "passes_chain",
        ),
        _trap(
            "divides_without_domain_check",
            "Does the response state cos A cos B != 0 (and 1 - tan A tan B != 0) before dividing by them?",
            [
                "tan(A + B)",
                "sin(A + B)/cos(A + B)",
                "(sin(A)*cos(B) + cos(A)*sin(B))/(cos(A)*cos(B) - sin(A)*sin(B))",
                "(sin(A)*cos(B)/(cos(A)*cos(B)) + cos(A)*sin(B)/(cos(A)*cos(B)))/(1 - sin(A)*sin(B)/(cos(A)*cos(B)))",
                "(tan(A) + sin(B)/cos(B))/(1 - tan(A)*sin(B)/cos(B))",
                "(tan(A) + tan(B))/(1 - tan(A)*tan(B))",
            ],
            "passes_chain",
        ),
    ],
)

_TAN_DIFF = dict(
    id="tan_diff",
    tier="hard",
    start="tan(A - B)",
    target="(tan(A) - tan(B))/(1 + tan(A)*tan(B))",
    identity_read="tan(A - B) = (tan(A) - tan(B))/(1 + tan(A)tan(B))",
    identity_tex=r"\tan(A - B) = \frac{\tan(A) - \tan(B)}{1 + \tan(A)\tan(B)}",
    given_read=(
        "sin(A - B) = sin(A)cos(B) - cos(A)sin(B) and cos(A - B) = cos(A)cos(B) + sin(A)sin(B), "
        "valid where cos(A), cos(B) and cos(A - B) are all nonzero"
    ),
    given_tex=(
        r"\sin(A - B) = \sin(A)\cos(B) - \cos(A)\sin(B)\text{ and }"
        r"\cos(A - B) = \cos(A)\cos(B) + \sin(A)\sin(B)\text{, valid where }\cos(A)"
        r"\text{, }\cos(B)\text{ and }\cos(A - B)\text{ are all nonzero}"
    ),
    reference_lines=[
        "tan(A - B)",
        "sin(A - B)/cos(A - B)",
        "(sin(A)*cos(B) - cos(A)*sin(B))/(cos(A)*cos(B) + sin(A)*sin(B))",
        "(sin(A)*cos(B)/(cos(A)*cos(B)) - cos(A)*sin(B)/(cos(A)*cos(B)))/(1 + sin(A)*sin(B)/(cos(A)*cos(B)))",
        "(tan(A) - sin(B)/cos(B))/(1 + tan(A)*sin(B)/cos(B))",
        "(tan(A) - tan(B))/(1 + tan(A)*tan(B))",
    ],
    key_steps=_key_steps(
        "the given sin(A - B) and cos(A - B) difference formulas",
        "dividing numerator and denominator by cos A cos B (valid since cos A cos B != 0)",
    ),
    traps=[
        _trap(
            "sign_slip",
            "Does the chain ever assert tan(A - B) = (tan A + tan B)/(1 - tan A tan B) (the tan(A + B) form)?",
            ["tan(A - B)", "(tan(A) + tan(B))/(1 - tan(A)*tan(B))", "(tan(A) - tan(B))/(1 + tan(A)*tan(B))"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_rhs",
            "Does the proof start from tan(A - B) itself, not the target?",
            list(
                reversed(
                    [
                        "tan(A - B)",
                        "sin(A - B)/cos(A - B)",
                        "(sin(A)*cos(B) - cos(A)*sin(B))/(cos(A)*cos(B) + sin(A)*sin(B))",
                        "(sin(A)*cos(B)/(cos(A)*cos(B)) - cos(A)*sin(B)/(cos(A)*cos(B)))/(1 + sin(A)*sin(B)/(cos(A)*cos(B)))",
                        "(tan(A) - sin(B)/cos(B))/(1 + tan(A)*sin(B)/cos(B))",
                        "(tan(A) - tan(B))/(1 + tan(A)*tan(B))",
                    ]
                )
            ),
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response derive the result, or quote it as the first step after the start?",
            ["tan(A - B)", "(tan(A) - tan(B))/(1 + tan(A)*tan(B))"],
            "passes_chain",
        ),
        _trap(
            "divides_without_domain_check",
            "Does the response state cos A cos B != 0 (and 1 + tan A tan B != 0) before dividing by them?",
            [
                "tan(A - B)",
                "sin(A - B)/cos(A - B)",
                "(sin(A)*cos(B) - cos(A)*sin(B))/(cos(A)*cos(B) + sin(A)*sin(B))",
                "(sin(A)*cos(B)/(cos(A)*cos(B)) - cos(A)*sin(B)/(cos(A)*cos(B)))/(1 + sin(A)*sin(B)/(cos(A)*cos(B)))",
                "(tan(A) - sin(B)/cos(B))/(1 + tan(A)*sin(B)/cos(B))",
                "(tan(A) - tan(B))/(1 + tan(A)*tan(B))",
            ],
            "passes_chain",
        ),
    ],
)

_COS_DIFF_DERIVATION = dict(
    id="cos_diff_derivation",
    tier="exam",
    start="cos(A - B)",
    target="cos(A)*cos(B) + sin(A)*sin(B)",
    identity_read="cos(A - B) = cos(A)cos(B) + sin(A)sin(B)",
    identity_tex=r"\cos(A - B) = \cos(A)\cos(B) + \sin(A)\sin(B)",
    given_read=(
        "the fact that, for M(cos A, sin A) and N(cos B, sin B) on the unit "
        "circle, MN^2 = 2 - 2cos(A - B) (the cosine rule at the origin), and "
        "sin(angle)^2 + cos(angle)^2 = 1"
    ),
    given_tex=(
        r"MN^2 = 2 - 2\cos(A - B)\text{ for }M(\cos A, \sin A), N(\cos B, \sin B)"
        r"\text{ on the unit circle, and }\sin(\text{angle})^2 + \cos(\text{angle})^2 = 1"
    ),
    reference_lines=[
        "cos(A - B)",
        "1 - ((cos(A) - cos(B))**2 + (sin(A) - sin(B))**2)/2",
        "1 - (cos(A)**2 - 2*cos(A)*cos(B) + cos(B)**2 + sin(A)**2 - 2*sin(A)*sin(B) + sin(B)**2)/2",
        "1 - ((cos(A)**2 + sin(A)**2) + (cos(B)**2 + sin(B)**2) - 2*cos(A)*cos(B) - 2*sin(A)*sin(B))/2",
        "1 - (1 + 1 - 2*cos(A)*cos(B) - 2*sin(A)*sin(B))/2",
        "cos(A)*cos(B) + sin(A)*sin(B)",
    ],
    key_steps=_key_steps(
        "the MN^2 = 2 - 2cos(A - B) given and the Pythagorean identity",
        "expanding MN^2 with the distance formula and substituting sin^2 t + cos^2 t = 1 for both M and N",
    ),
    traps=[
        _trap(
            "sign_slip",
            "Does the chain ever assert cos(A - B) = 1 + MN^2/2 (sign error rearranging the given)?",
            ["cos(A - B)", "1 + ((cos(A) - cos(B))**2 + (sin(A) - sin(B))**2)/2", "cos(A)*cos(B) + sin(A)*sin(B)"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_rhs",
            "Does the proof start from cos(A - B) itself, not the target?",
            list(
                reversed(
                    [
                        "cos(A - B)",
                        "1 - ((cos(A) - cos(B))**2 + (sin(A) - sin(B))**2)/2",
                        "1 - (cos(A)**2 - 2*cos(A)*cos(B) + cos(B)**2 + sin(A)**2 - 2*sin(A)*sin(B) + sin(B)**2)/2",
                        "1 - ((cos(A)**2 + sin(A)**2) + (cos(B)**2 + sin(B)**2) - 2*cos(A)*cos(B) - 2*sin(A)*sin(B))/2",
                        "1 - (1 + 1 - 2*cos(A)*cos(B) - 2*sin(A)*sin(B))/2",
                        "cos(A)*cos(B) + sin(A)*sin(B)",
                    ]
                )
            ),
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response derive cos(A - B) from the unit-circle distance formula, or quote the target as the first step?",
            ["cos(A - B)", "cos(A)*cos(B) + sin(A)*sin(B)"],
            "passes_chain",
        ),
    ],
)

_COMPOSITE_SIN = dict(
    id="composite_sin",
    tier="exam",
    start="sin(A + B)*sin(A - B)",
    target="sin(A)**2 - sin(B)**2",
    identity_read="sin(A + B) sin(A - B) = sin(A)^2 - sin(B)^2",
    identity_tex=r"\sin(A + B)\sin(A - B) = \sin(A)^2 - \sin(B)^2",
    given_read=(
        "sin(A + B) = sin(A)cos(B) + cos(A)sin(B), sin(A - B) = sin(A)cos(B) - cos(A)sin(B), "
        "and sin(angle)^2 + cos(angle)^2 = 1"
    ),
    given_tex=(
        r"\sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B),\ "
        r"\sin(A - B) = \sin(A)\cos(B) - \cos(A)\sin(B)\text{, and }"
        r"\sin(\text{angle})^2 + \cos(\text{angle})^2 = 1"
    ),
    reference_lines=[
        "sin(A + B)*sin(A - B)",
        "(sin(A)*cos(B) + cos(A)*sin(B))*(sin(A)*cos(B) - cos(A)*sin(B))",
        "(sin(A)*cos(B))**2 - (cos(A)*sin(B))**2",
        "sin(A)**2*cos(B)**2 - cos(A)**2*sin(B)**2",
        "sin(A)**2*(1 - sin(B)**2) - (1 - sin(A)**2)*sin(B)**2",
        "sin(A)**2 - sin(A)**2*sin(B)**2 - sin(B)**2 + sin(A)**2*sin(B)**2",
        "sin(A)**2 - sin(B)**2",
    ],
    key_steps=_key_steps(
        "the given sin(A + B), sin(A - B) formulas and sin^2 t + cos^2 t = 1",
        "expanding the product as a difference of squares, then substituting cos^2 t = 1 - sin^2 t for both A and B",
    ),
    traps=[
        _trap(
            "sign_slip",
            "Does the chain expand the product as a SUM of squares (sin A cos B)^2 + (cos A sin B)^2 instead of a difference?",
            ["sin(A + B)*sin(A - B)", "(sin(A)*cos(B))**2 + (cos(A)*sin(B))**2", "sin(A)**2 - sin(B)**2"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_rhs",
            "Does the proof start from sin(A + B) sin(A - B) itself, not the target?",
            list(
                reversed(
                    [
                        "sin(A + B)*sin(A - B)",
                        "(sin(A)*cos(B) + cos(A)*sin(B))*(sin(A)*cos(B) - cos(A)*sin(B))",
                        "(sin(A)*cos(B))**2 - (cos(A)*sin(B))**2",
                        "sin(A)**2*cos(B)**2 - cos(A)**2*sin(B)**2",
                        "sin(A)**2*(1 - sin(B)**2) - (1 - sin(A)**2)*sin(B)**2",
                        "sin(A)**2 - sin(A)**2*sin(B)**2 - sin(B)**2 + sin(A)**2*sin(B)**2",
                        "sin(A)**2 - sin(B)**2",
                    ]
                )
            ),
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response derive the result, or quote it as the first step after the start?",
            ["sin(A + B)*sin(A - B)", "sin(A)**2 - sin(B)**2"],
            "passes_chain",
        ),
    ],
)

_COMPOSITE_COS = dict(
    id="composite_cos",
    tier="exam",
    start="cos(A + B)*cos(A - B)",
    target="cos(A)**2 - sin(B)**2",
    identity_read="cos(A + B) cos(A - B) = cos(A)^2 - sin(B)^2",
    identity_tex=r"\cos(A + B)\cos(A - B) = \cos(A)^2 - \sin(B)^2",
    given_read=(
        "cos(A + B) = cos(A)cos(B) - sin(A)sin(B), cos(A - B) = cos(A)cos(B) + sin(A)sin(B), "
        "and sin(angle)^2 + cos(angle)^2 = 1"
    ),
    given_tex=(
        r"\cos(A + B) = \cos(A)\cos(B) - \sin(A)\sin(B),\ "
        r"\cos(A - B) = \cos(A)\cos(B) + \sin(A)\sin(B)\text{, and }"
        r"\sin(\text{angle})^2 + \cos(\text{angle})^2 = 1"
    ),
    reference_lines=[
        "cos(A + B)*cos(A - B)",
        "(cos(A)*cos(B) - sin(A)*sin(B))*(cos(A)*cos(B) + sin(A)*sin(B))",
        "(cos(A)*cos(B))**2 - (sin(A)*sin(B))**2",
        "cos(A)**2*cos(B)**2 - sin(A)**2*sin(B)**2",
        "cos(A)**2*(1 - sin(B)**2) - (1 - cos(A)**2)*sin(B)**2",
        "cos(A)**2 - cos(A)**2*sin(B)**2 - sin(B)**2 + cos(A)**2*sin(B)**2",
        "cos(A)**2 - sin(B)**2",
    ],
    key_steps=_key_steps(
        "the given cos(A + B), cos(A - B) formulas and sin^2 t + cos^2 t = 1",
        "expanding the product as a difference of squares, then substituting sin^2 t = 1 - cos^2 t for B and leaving A as cos^2 A",
    ),
    traps=[
        _trap(
            "sign_slip",
            "Does the chain expand the product as a SUM of squares (cos A cos B)^2 + (sin A sin B)^2 instead of a difference?",
            ["cos(A + B)*cos(A - B)", "(cos(A)*cos(B))**2 + (sin(A)*sin(B))**2", "cos(A)**2 - sin(B)**2"],
            "authoritative_failure",
        ),
        _trap(
            "starts_from_rhs",
            "Does the proof start from cos(A + B) cos(A - B) itself, not the target?",
            list(
                reversed(
                    [
                        "cos(A + B)*cos(A - B)",
                        "(cos(A)*cos(B) - sin(A)*sin(B))*(cos(A)*cos(B) + sin(A)*sin(B))",
                        "(cos(A)*cos(B))**2 - (sin(A)*sin(B))**2",
                        "cos(A)**2*cos(B)**2 - sin(A)**2*sin(B)**2",
                        "cos(A)**2*(1 - sin(B)**2) - (1 - cos(A)**2)*sin(B)**2",
                        "cos(A)**2 - cos(A)**2*sin(B)**2 - sin(B)**2 + cos(A)**2*sin(B)**2",
                        "cos(A)**2 - sin(B)**2",
                    ]
                )
            ),
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response derive the result, or quote it as the first step after the start?",
            ["cos(A + B)*cos(A - B)", "cos(A)**2 - sin(B)**2"],
            "passes_chain",
        ),
    ],
)

_PROVE_BY_TIER = {
    "easy": _expand_family(_COS_SUM) + _expand_family(_SIN_DIFF_EASY),
    "medium": _expand_family(_SIN_SUM) + _expand_family(_SIN_DIFF),
    "hard": _expand_family(_TAN_SUM) + _expand_family(_TAN_DIFF),
    "exam": _expand_family(_COS_DIFF_DERIVATION) + _expand_family(_COMPOSITE_SIN) + _expand_family(_COMPOSITE_COS),
}

for _entries in _PROVE_BY_TIER.values():
    for _entry in _entries:
        _verify_symbolic_chain(_entry["reference_lines"], _entry["id"], letters=_entry["letters"])


# ---------------------------------------------------------------------------
# APPLY pool -- easy tier only. Exact value of sin/cos of pi/12, 5pi/12,
# 7pi/12 or 11pi/12, written as a sum/difference of the standard reference
# angles (pi/6, pi/4, pi/3, 2pi/3, 3pi/4 -- the usual first/second-quadrant
# table, a normal prior given alongside the sum/difference formula itself).
# ---------------------------------------------------------------------------

_BASE_ANGLE_NAME = {2: "pi/6", 3: "pi/4", 4: "pi/3", 8: "2*pi/3", 9: "3*pi/4"}
_BASE_ANGLE_SIN = {2: "1/2", 3: "sqrt(2)/2", 4: "sqrt(3)/2", 8: "sqrt(3)/2", 9: "sqrt(2)/2"}
_BASE_ANGLE_COS = {2: "sqrt(3)/2", 3: "sqrt(2)/2", 4: "1/2", 8: "-1/2", 9: "-sqrt(2)/2"}
_BASE_ANGLE_READ = {2: "pi/6", 3: "pi/4", 4: "pi/3", 8: "2pi/3", 9: "3pi/4"}
_BASE_ANGLE_TEX = {
    2: r"\tfrac{\pi}{6}",
    3: r"\tfrac{\pi}{4}",
    4: r"\tfrac{\pi}{3}",
    8: r"\tfrac{2\pi}{3}",
    9: r"\tfrac{3\pi}{4}",
}

_TARGET_ANGLE_READ = {1: "pi/12", 5: "5pi/12", 7: "7pi/12", 11: "11pi/12"}
_TARGET_ANGLE_TEX = {
    1: r"\tfrac{\pi}{12}",
    5: r"\tfrac{5\pi}{12}",
    7: r"\tfrac{7\pi}{12}",
    11: r"\tfrac{11\pi}{12}",
}
_TARGET_ANGLE_DEG = {1: 15, 5: 75, 7: 105, 11: 165}

# (A_twelfths, B_twelfths, op) decompositions per target-angle twelfths.
_DECOMPS = {
    1: [(4, 3, "-"), (3, 2, "-")],
    5: [(3, 2, "+"), (9, 4, "-")],
    7: [(4, 3, "+"), (9, 2, "-")],
    11: [(9, 2, "+"), (8, 3, "+")],
}

# Final exact value, confirmed symbolically for every (func, r) pair below.
_FINAL_VALUE = {
    ("sin", 1): "(sqrt(6) - sqrt(2))/4",
    ("cos", 1): "(sqrt(6) + sqrt(2))/4",
    ("sin", 5): "(sqrt(6) + sqrt(2))/4",
    ("cos", 5): "(sqrt(6) - sqrt(2))/4",
    ("sin", 7): "(sqrt(6) + sqrt(2))/4",
    ("cos", 7): "(sqrt(2) - sqrt(6))/4",
    ("sin", 11): "(sqrt(6) - sqrt(2))/4",
    ("cos", 11): "-(sqrt(6) + sqrt(2))/4",
}
_FINAL_VALUE_READ = {
    ("sin", 1): "(sqrt6 - sqrt2)/4",
    ("cos", 1): "(sqrt6 + sqrt2)/4",
    ("sin", 5): "(sqrt6 + sqrt2)/4",
    ("cos", 5): "(sqrt6 - sqrt2)/4",
    ("sin", 7): "(sqrt6 + sqrt2)/4",
    ("cos", 7): "(sqrt2 - sqrt6)/4",
    ("sin", 11): "(sqrt6 - sqrt2)/4",
    ("cos", 11): "-(sqrt6 + sqrt2)/4",
}
_FINAL_VALUE_TEX = {
    ("sin", 1): r"\frac{\sqrt{6}-\sqrt{2}}{4}",
    ("cos", 1): r"\frac{\sqrt{6}+\sqrt{2}}{4}",
    ("sin", 5): r"\frac{\sqrt{6}+\sqrt{2}}{4}",
    ("cos", 5): r"\frac{\sqrt{6}-\sqrt{2}}{4}",
    ("sin", 7): r"\frac{\sqrt{6}+\sqrt{2}}{4}",
    ("cos", 7): r"\frac{\sqrt{2}-\sqrt{6}}{4}",
    ("sin", 11): r"\frac{\sqrt{6}-\sqrt{2}}{4}",
    ("cos", 11): r"-\frac{\sqrt{6}+\sqrt{2}}{4}",
}

# The concrete sum/difference formula an apply item actually uses, stated
# as a formula (not just named), in typeable (bracketed) notation, so the
# oracle's "every given is a formula, not only a name" check has something
# to find and a learner who copies it never mis-parses it.
_APPLY_FORMULA_READ = {
    ("sin", "+"): "sin(A + B) = sin(A)cos(B) + cos(A)sin(B)",
    ("sin", "-"): "sin(A - B) = sin(A)cos(B) - cos(A)sin(B)",
    ("cos", "+"): "cos(A + B) = cos(A)cos(B) - sin(A)sin(B)",
    ("cos", "-"): "cos(A - B) = cos(A)cos(B) + sin(A)sin(B)",
}
_APPLY_FORMULA_TEX = {
    ("sin", "+"): r"\sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B)",
    ("sin", "-"): r"\sin(A - B) = \sin(A)\cos(B) - \cos(A)\sin(B)",
    ("cos", "+"): r"\cos(A + B) = \cos(A)\cos(B) - \sin(A)\sin(B)",
    ("cos", "-"): r"\cos(A - B) = \cos(A)\cos(B) + \sin(A)\sin(B)",
}


def _build_apply_lines(func: str, a_twelfths: int, b_twelfths: int, op: str) -> list[str]:
    a_name, b_name = _BASE_ANGLE_NAME[a_twelfths], _BASE_ANGLE_NAME[b_twelfths]
    a_sin, a_cos = _BASE_ANGLE_SIN[a_twelfths], _BASE_ANGLE_COS[a_twelfths]
    b_sin, b_cos = _BASE_ANGLE_SIN[b_twelfths], _BASE_ANGLE_COS[b_twelfths]
    if func == "sin":
        step2 = f"sin({a_name} {op} {b_name})"
        inner_op = "+" if op == "+" else "-"
        step3 = f"sin({a_name})*cos({b_name}) {inner_op} cos({a_name})*sin({b_name})"
    else:
        step2 = f"cos({a_name} {op} {b_name})"
        inner_op = "-" if op == "+" else "+"
        step3 = f"cos({a_name})*cos({b_name}) {inner_op} sin({a_name})*sin({b_name})"
    step4 = (
        step3.replace(f"sin({a_name})", a_sin)
        .replace(f"cos({a_name})", a_cos)
        .replace(f"sin({b_name})", b_sin)
        .replace(f"cos({b_name})", b_cos)
    )
    return [step2, step3, step4]


_APPLY_ENTRIES: list[dict] = []
for _r, _decomps in _DECOMPS.items():
    for _func in ("sin", "cos"):
        _start = f"{_func}(pi*{_r}/12)"
        _final = _FINAL_VALUE[(_func, _r)]
        for _a, _b, _op in _decomps:
            _mid = _build_apply_lines(_func, _a, _b, _op)
            _APPLY_ENTRIES.append(
                dict(
                    func=_func,
                    r=_r,
                    a=_a,
                    b=_b,
                    op=_op,
                    start=_start,
                    target=_final,
                    reference_lines=[_start] + _mid + [_final],
                )
            )

for _apply_entry in _APPLY_ENTRIES:
    _verify_numeric_chain(_apply_entry["reference_lines"], f"apply {_apply_entry['func']} {_apply_entry['r']}")


def _apply_marking(entry: dict) -> dict:
    func, r, a, b, op = entry["func"], entry["r"], entry["a"], entry["b"], entry["op"]
    a_read, b_read = _BASE_ANGLE_READ[a], _BASE_ANGLE_READ[b]
    decomp_read = f"{_TARGET_ANGLE_READ[r]} = {a_read} {op} {b_read}"
    formula_read = _APPLY_FORMULA_READ[(func, op)]
    given_read = (
        f"{formula_read}, and the standard values {a_read}, {b_read} (sin and cos of pi/6, pi/4, pi/3, 2pi/3 and 3pi/4)"
    )
    final_read = _FINAL_VALUE_READ[(func, r)]
    identity_read = f"{func}({_TARGET_ANGLE_READ[r]}) = {final_read}, using {decomp_read}"
    key_steps = _key_steps(
        f"the {func} sum/difference formula and the standard angle values",
        f"writing {_TARGET_ANGLE_READ[r]} as {decomp_read} before applying the formula",
    )
    traps = [
        _trap(
            "wrong_decomposition_sign",
            f"Does the response use the correct sign in {decomp_read}, not the other one?",
            [
                entry["start"],
                f"{func}({a_read} {'+' if op == '-' else '-'} {b_read})",
                *_build_apply_lines(func, a, b, "+" if op == "-" else "-")[1:],
                entry["target"],
            ],
            "non_authoritative_failure",
        ),
        _trap(
            "cites_the_result_being_proved",
            "Does the response show the decomposition and formula, or just assert the exact value directly?",
            [entry["start"], entry["target"]],
            "passes_chain",
        ),
    ]
    return (
        _marking(entry["start"], entry["target"], {}, entry["reference_lines"], key_steps, traps),
        identity_read,
        given_read,
    )


# ---------------------------------------------------------------------------
# Prompt wording -- each (text_template, latex_template) pair shares
# {identity}/{given}/{start}/{target} slots (`.format()` for the text side;
# the latex side uses `%TOKEN%` markers and plain `.replace()` instead of
# `.format()`, because the latex fragments themselves contain literal `{`/`}`
# from `\text{...}`/`\frac{...}{...}`, which `.format()` would misread as
# fields). Every rendered prompt also carries a short notation sentence
# appended after the main template.
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
    (
        "Establish that {identity} by proving it from {given}. The chain must run from {start} to {target}.",
        r"\text{Establish that }%IDENTITY%\text{ by proving it from }%GIVEN%"
        r".\text{ The chain must run from }%START%\text{ to }%TARGET%.",
    ),
    (
        "Demonstrate that {identity}, reasoning from {given}, with your working beginning at "
        "{start} and concluding at {target}.",
        r"\text{Demonstrate that }%IDENTITY%,\text{ reasoning from }%GIVEN%"
        r",\text{ with your working beginning at }%START%\text{ and concluding at }%TARGET%.",
    ),
]

_APPLY_TEMPLATES = [
    (
        "Find the exact value of {start} ({deg} degrees). Write the angle as a sum or "
        "difference of standard angles and use the given sum/difference formula: {given}.",
        r"\text{Find the exact value of }%START%\ (%DEG%^\circ)"
        r".\text{ Write the angle as a sum or difference of standard angles and use the "
        r"given sum/difference formula: }%GIVEN%.",
    ),
    (
        "Evaluate {start} exactly by writing the angle as a sum or difference of standard angles ({given}).",
        r"\text{Evaluate }%START%\text{ exactly by writing the angle as a sum or "
        r"difference of standard angles (}%GIVEN%).",
    ),
    (
        "Give the exact value of {start}, using {given}.",
        r"\text{Give the exact value of }%START%,\text{ using }%GIVEN%.",
    ),
]

# The notation sentence -- at least two shapes, chosen per item, and
# always appended (both `text` and `latex`) after the main template. Never
# part of any regex's captured group: every extraction pattern in the test
# files anchors up to the first "." that ends the target/given clause.
_NOTATION_SENTENCES = [
    (
        "Write one expression per line; put brackets round every angle, as in sin(x), and "
        "use ^ for powers, as in cos(x)^2.",
        r"\text{Write one expression per line; put brackets round every angle, as in sin(x), "
        r"and use \textasciicircum{} for powers, as in cos(x)\textasciicircum{}2.}",
    ),
    (
        "Type one line per step; bracket every angle (for example sin(x)) and use ^ for powers (for example cos(x)^2).",
        r"\text{Type one line per step; bracket every angle (for example sin(x)) and use "
        r"\textasciicircum{} for powers (for example cos(x)\textasciicircum{}2).}",
    ),
]


def _notation_sentence(rng: random.Random) -> tuple[str, str]:
    return _NOTATION_SENTENCES[rng.randrange(len(_NOTATION_SENTENCES))]


def _fill_tex(template: str, *, identity=None, given=None, start=None, target=None, deg=None) -> str:
    out = template
    if identity is not None:
        out = out.replace("%IDENTITY%", identity)
    if given is not None:
        out = out.replace("%GIVEN%", given)
    if start is not None:
        out = out.replace("%START%", start)
    if target is not None:
        out = out.replace("%TARGET%", target)
    if deg is not None:
        out = out.replace("%DEG%", str(deg))
    return out


def _split_identity(identity_read: str) -> tuple[str, str]:
    """`"cos(A + B) = cos(A)cos(B) - sin(A)sin(B)"` -> the two readable
    sides, for a prompt's "start at ... / end at ..." phrasing -- never the
    sympy `start`/`target` strings themselves, which carry `*`/`**` (rule:
    no sympy syntax in learner-facing `text`)."""
    left, right = identity_read.split(" = ", 1)
    return left, right


def _text_verbatim(s: str) -> str:
    r"""Wrap a typeable-notation string (this module's own readable start/ target halves, already free of
    `**`/`*`/unicode superscripts) in `\text{...}`, escaping the one special character it can contain (`^`) as
    `\textasciicircum{}` -- the KaTeX-verified pattern. This is what makes the quoted "start from.../finish
    at..." clause exactly the same typeable line the learner sees in `text`, never a second,
    independently-built real-math rendering."""
    return r"\text{" + s.replace("^", r"\textasciicircum{}") + "}"


def _fmt_prove(entry: dict, rng: random.Random) -> tuple[str, str]:
    idx = rng.randrange(len(_PROVE_TEMPLATES))
    text_t, latex_t = _PROVE_TEMPLATES[idx]
    start_read, target_read = _split_identity(entry["identity_read"])
    notation_text, notation_tex = _notation_sentence(rng)
    text = (
        text_t.format(identity=entry["identity_read"], given=entry["given_read"], start=start_read, target=target_read)
        + " "
        + notation_text
    )
    latex_str = (
        _fill_tex(
            latex_t,
            identity=entry["identity_tex"],
            given=entry["given_tex"],
            start=_text_verbatim(start_read),
            target=_text_verbatim(target_read),
        )
        + " "
        + notation_tex
    )
    return text, latex_str


def _fmt_apply(entry: dict, given_read: str, rng: random.Random) -> tuple[str, str]:
    r, func, op = entry["r"], entry["func"], entry["op"]
    idx = rng.randrange(len(_APPLY_TEMPLATES))
    text_t, latex_t = _APPLY_TEMPLATES[idx]
    start_read = f"{func}({_TARGET_ANGLE_READ[r]})"
    notation_text, notation_tex = _notation_sentence(rng)
    text = text_t.format(start=start_read, given=given_read, deg=_TARGET_ANGLE_DEG[r]) + " " + notation_text
    latex_str = _fill_tex(
        latex_t, start=_text_verbatim(start_read), given=_apply_given_tex(func, op), deg=_TARGET_ANGLE_DEG[r]
    )
    latex_str = latex_str + " " + notation_tex
    return text, latex_str


def _apply_given_tex(func: str, op: str) -> str:
    return _APPLY_FORMULA_TEX[(func, op)] + r",\text{ and the standard trig values}"


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


@generator(
    "pr1a_angle_sum_difference_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def pr1a_angle_sum_difference_proof(rng: random.Random, variant: str = "easy") -> dict:
    """ACMMM041: prove one of sin/cos/tan(A +/- B) as a chain from stated
    givens (medium chains two givens; hard divides by cos A cos B; exam
    derives cos(A - B) itself from the unit circle, or proves a two-formula
    composite identity) or -- easy tier only -- find an exact value by
    writing a special angle as a sum/difference of standard ones. No
    distractors: the answer is a rubric marking dict."""
    if variant == "easy":
        if rng.random() < 0.5:
            entry = rng.choice(_PROVE_BY_TIER["easy"])
            text, latex_str = _fmt_prove(entry, rng)
            marking = _marking(
                entry["start"],
                entry["target"],
                entry["symbols"],
                entry["reference_lines"],
                entry["key_steps"],
                entry["traps"],
            )
            answer_text = entry["identity_read"]
        else:
            apply_entry = rng.choice(_APPLY_ENTRIES)
            marking, identity_read, given_read = _apply_marking(apply_entry)
            text, latex_str = _fmt_apply(apply_entry, given_read, rng)
            answer_text = identity_read
        return dict(text=text, latex=latex_str, answer=marking, answer_text=answer_text, distractors=[])

    if variant in ("medium", "hard", "exam"):
        entry = rng.choice(_PROVE_BY_TIER[variant])
    else:
        raise ValueError(f"unknown variant: {variant}")

    text, latex_str = _fmt_prove(entry, rng)
    marking = _marking(
        entry["start"], entry["target"], entry["symbols"], entry["reference_lines"], entry["key_steps"], entry["traps"]
    )
    return dict(text=text, latex=latex_str, answer=marking, answer_text=entry["identity_read"], distractors=[])
