"""Generates `float_error_budget` items: the first-order worst-case rounding error of a fixed sequence of
floating-point operations.

The model is stated in every question: each operation returns the exact result of its inputs multiplied by
(1 + δ), with its own d and |d| <= u, the inputs are stored exactly, and terms of order u^2 and smaller are
ignored. The worst-case absolute error of the computed result is then k u, where k is the sum over the operations
of |dr/dv_j| |v_j|, v_j the exact value that operation produced and r the final result. The learner is asked for k.

The first-order coefficients are computed with exact fractions by forward propagation of the rounding of each
operation; the build test recomputes k from the text with a symbolic derivative (SymPy) instead.

Ladder:

- easy: two or three additions and multiplications of positive whole numbers.
- medium: four operations with subtraction, and a division by a small constant.
- hard: five or six operations in which an intermediate result is used twice, so its rounding error is counted
  once on each path, with subtraction and a division.
- exam: comparing two evaluation orders of a sum of four numbers, a polynomial evaluated by Horner's rule, and
  longer mixed traces with a reused intermediate.

Every answer is a terminating decimal, so each is matched exactly. Each distractor is the answer of a named slip
and clears the answer by more than the grader's tolerance.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import GRADER_TOLERANCE, check, generator, pick_distractors
from ._free_proof_kit import Fm, Prompt, attempts

_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}

_DIVISORS = (2, 4, 5, 8)
_SYMBOL = {"+": "+", "-": "-", "*": "×", "/": "÷"}
_TEX = {"+": "+", "-": "-", "*": r"\times ", "/": r"\div "}


def _val(env: dict, x):
    return x if isinstance(x, Fraction) else env[x]


def _run(inputs: dict, ops: list[tuple]):
    """Exact values of each step and, for each step, its first-order sensitivities to every rounding.

    `ops` is a list of (dest, op, a, b) with a and b the name of an input or an earlier step, or a Fraction
    constant (exact, carrying no rounding of its own). Returns (values, coeffs): coeffs[name][j] is the derivative
    of that step's value with respect to the rounding multiplier d_j of step j, at d = 0.
    """
    env: dict = {k: Fraction(v) for k, v in inputs.items()}
    coef: dict = {k: {} for k in inputs}
    for dest, op, a, b in ops:
        va, vb = _val(env, a), _val(env, b)
        ca = coef[a] if not isinstance(a, Fraction) else {}
        cb = coef[b] if not isinstance(b, Fraction) else {}
        cd: dict = {}
        keys = set(ca) | set(cb)
        if op == "+":
            val = va + vb
            for j in keys:
                cd[j] = ca.get(j, 0) + cb.get(j, 0)
        elif op == "-":
            val = va - vb
            for j in keys:
                cd[j] = ca.get(j, 0) - cb.get(j, 0)
        elif op == "*":
            val = va * vb
            for j in keys:
                cd[j] = ca.get(j, 0) * vb + va * cb.get(j, 0)
        elif op == "/":
            check(vb != 0, "division by zero")
            val = va / vb
            for j in keys:
                cd[j] = ca.get(j, 0) / vb - va * cb.get(j, 0) / (vb * vb)
        else:  # pragma: no cover
            raise ValueError(op)
        cd[dest] = val  # the rounding of this step itself: val * d_dest
        env[dest] = val
        coef[dest] = cd
    return env, coef


def _k(inputs: dict, ops: list[tuple]) -> Fraction:
    env, coef = _run(inputs, ops)
    final = ops[-1][0]
    return sum((abs(c) for c in coef[final].values()), Fraction(0))


class _TooFewWrongValues(Exception):
    """The trace has fewer than two wrong values that read like the answer; the caller draws another trace."""


def _terminating(x: Fraction, places: int = 3) -> bool:
    scaled = x * 10**places
    return scaled.denominator == 1


def _number_text(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    s = f"{float(x):.6f}".rstrip("0")
    check(Fraction(s) == x, "the answer is a terminating decimal")
    return s


def _like(x: Fraction, answer_text: str) -> str | None:
    """`x` written with as many decimal places as the answer, or None when it needs more (a wrong value must read
    like the answer: a whole number beside whole numbers, one decimal place beside one)."""
    places = len(answer_text.split(".")[1]) if "." in answer_text else 0
    if (x * 10**places).denominator != 1:
        return None
    return str(x.numerator) if places == 0 else f"{float(x):.{places}f}"


def _value(x: Fraction):
    return int(x) if x.denominator == 1 else float(x)


def _inputs_fm(inputs: dict) -> Fm:
    items = [f"{k} = {v}" for k, v in inputs.items()]
    plain = ", ".join(items[:-1]) + " and " + items[-1]
    tex = ",\\;".join(f"{k}={v}" for k, v in inputs.items())
    return Fm(tex, plain)


def _op_fm(dest: str, op: str, a, b) -> Fm:
    def nm(x):
        return str(x.numerator) if isinstance(x, Fraction) else x

    return Fm(f"{dest}={nm(a)}{_TEX[op]}{nm(b)}", f"{dest} = {nm(a)} {_SYMBOL[op]} {nm(b)}")


def _wrong_values(inputs: dict, ops: list[tuple]) -> list[tuple[str, Fraction]]:
    """Slips: the number of operations, the magnitudes of the steps without carrying their error forward, the
    final result alone, and everything but the last rounding."""
    env, coef = _run(inputs, ops)
    final = ops[-1][0]
    steps = [d for d, *_ in ops]
    k = sum((abs(c) for c in coef[final].values()), Fraction(0))
    return [
        ("operation_count", Fraction(len(ops))),
        ("magnitudes_not_carried", sum((abs(env[d]) for d in steps), Fraction(0))),
        ("final_rounding_only", abs(env[final])),
        ("misses_last_rounding", k - abs(env[final])),
        ("each_step_once", sum((abs(coef[final][d]) for d in steps[:-1]), Fraction(0)) + abs(env[final])),
    ]


def _build(rng, inputs: dict, ops: list[tuple], *, story: str) -> dict:
    k = _k(inputs, ops)
    check(k > 0 and _terminating(k), "k is a positive terminating decimal")
    # independent re-derivation: the same sum from the partial derivatives of the rounded expression, by finite
    # differences on the rounding multipliers computed with exact fractions
    _check_by_perturbation(inputs, ops, k)
    p = Prompt()
    p.t("In a floating-point model, every operation returns the exact result of its two inputs multiplied by ").m(
        Fm("(1+\\delta)", "(1 + δ)")
    ).t(", where ").m(Fm("\\delta", "δ")).t(" is a different number for each operation and ").m(
        Fm("|\\delta|\\le u", "|δ| ≤ u")
    ).t(". The inputs ").m(_inputs_fm(inputs)).t(f" are stored exactly. {story}")
    first = True
    for dest, op, a, b in ops:
        p.t("" if first else ", then ")
        p.m(_op_fm(dest, op, a, b))
        first = False
    p.t(". Ignore terms of order ").m(Fm("u^{2}", "u^2")).t(
        " and smaller. The worst-case absolute error of the computed "
    )
    p.m(Fm(ops[-1][0])).t(" is ").m(Fm("k\\,u", "k u")).t(". Find ").m(Fm("k")).t(".")
    ans = _value(k)
    atxt = _number_text(k)
    wrongs = _wrong_values(inputs, ops)
    cands = [t for _, w in wrongs if w > 0 for t in [_like(w, atxt)] if t is not None]
    distractors = pick_distractors(atxt, cands, n=3, tolerance=GRADER_TOLERANCE)
    if len(distractors) < 2:
        raise _TooFewWrongValues

    env, coef = _run(inputs, ops)
    final = ops[-1][0]
    working = [
        f"{d} = {_number_text(env[d])}; its rounding changes the result by {_number_text(abs(coef[final][d]))} u"
        for d, *_ in ops
    ]
    working.append(f"k = {atxt}")
    return dict(
        text=p.text,
        latex=p.latex,
        answer=ans,
        answer_text=atxt,
        working=working,
        distractors=distractors,
    )


def _check_by_perturbation(inputs: dict, ops: list[tuple], k: Fraction) -> None:
    """Recompute k by perturbing each rounding multiplier a little (exact fractions) and measuring the change in
    the result, then taking the limit as the perturbation tends to zero by Richardson-style differencing on the
    exactly polynomial or rational dependence. For these operations the dependence on one multiplier is a
    rational function, so a symmetric difference at tiny eps matches the derivative to well within one part in a
    million; the check is on the sum, which must agree with k to that tolerance."""
    eps = Fraction(1, 10**9)
    steps = [d for d, *_ in ops]

    def run_with(j: str, delta: Fraction) -> Fraction:
        env: dict = {kk: Fraction(v) for kk, v in inputs.items()}
        for dest, op, a, b in ops:
            va, vb = _val(env, a), _val(env, b)
            if op == "+":
                v = va + vb
            elif op == "-":
                v = va - vb
            elif op == "*":
                v = va * vb
            else:
                v = va / vb
            if dest == j:
                v = v * (1 + delta)
            env[dest] = v
        return env[ops[-1][0]]

    total = Fraction(0)
    for j in steps:
        slope = (run_with(j, eps) - run_with(j, -eps)) / (2 * eps)
        total += abs(slope)
    check(abs(total - k) < Fraction(1, 10**4), "perturbation check of the error coefficient sum")


# ---------------------------------------------------------------------------
# Easy
# ---------------------------------------------------------------------------
def _easy(rng) -> dict:
    n_in = rng.choice((3, 3, 4))
    names = ["a", "b", "c", "d"][:n_in]
    for _ in attempts(500, "an easy trace"):
        inputs = {n: rng.randint(2, 9) for n in names}
        ops = []
        prev = names[0]
        for i, n in enumerate(names[1:]):
            op = rng.choice(("+", "+", "*"))
            dest = f"s{i + 1}"
            ops.append((dest, op, prev, n))
            prev = dest
        if len(ops) >= 2:
            break
    return _build(rng, inputs, ops, story="Compute ")


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------
def _medium(rng) -> dict:
    for _ in attempts(2000, "a medium trace"):
        inputs = {"a": rng.randint(2, 12), "b": rng.randint(2, 12), "c": rng.randint(2, 9), "d": rng.randint(2, 9)}
        o1 = rng.choice(("+", "*"))
        o2 = rng.choice(("+", "-", "*"))
        o3 = rng.choice(("-", "+"))
        ops = [("s", o1, "a", "b"), ("t", o2, "s", "c")]
        if rng.random() < 0.5:
            ops.append(("q", "/", "t", Fraction(rng.choice(_DIVISORS))))
            ops.append(("r", o3, "q", "d"))
        else:
            ops.append(("q", "*", "t", "d"))
            ops.append(("r", o3, "q", "a"))
        env, _ = _run(inputs, ops)
        if any(v <= 0 for v in env.values()):
            continue
        if _terminating(_k(inputs, ops)):
            break
    return _build(rng, inputs, ops, story="Compute ")


# ---------------------------------------------------------------------------
# Hard
# ---------------------------------------------------------------------------
def _hard(rng) -> dict:
    for _ in attempts(4000, "a hard trace with a reused intermediate"):
        inputs = {"a": rng.randint(2, 9), "b": rng.randint(2, 9), "c": rng.randint(2, 7)}
        shape = rng.choice(("square_minus", "ratio", "double_use"))
        if shape == "square_minus":
            ops = [("s", "+", "a", "b"), ("t", "*", "s", "s"), ("w", "-", "t", "s"), ("r", "*", "w", "c")]
        elif shape == "ratio":
            ops = [
                ("s", "*", "a", "b"),
                ("t", "+", "s", "c"),
                ("w", "*", "t", "s"),
                ("q", "/", "w", Fraction(rng.choice(_DIVISORS))),
                ("r", "-", "q", "b"),
            ]
        else:
            ops = [("s", "+", "a", "b"), ("t", "-", "s", "c"), ("w", "*", "s", "t"), ("r", "+", "w", "s")]
        env, _ = _run(inputs, ops)
        if any(v <= 0 for v in env.values()):
            continue
        if _terminating(_k(inputs, ops)):
            break
    return _build(rng, inputs, ops, story="Compute ")


# ---------------------------------------------------------------------------
# Exam
# ---------------------------------------------------------------------------
def _exam_orders(rng) -> dict:
    """k for the left-to-right sum minus k for the pairwise sum of four positive numbers: asks for the saving."""
    inputs = {n: rng.randint(1, 12) for n in "abcd"}
    left = [("s1", "+", "a", "b"), ("s2", "+", "s1", "c"), ("r", "+", "s2", "d")]
    pair = [("p1", "+", "a", "b"), ("p2", "+", "c", "d"), ("r", "+", "p1", "p2")]
    kl, kp = _k(inputs, left), _k(inputs, pair)
    diff = kl - kp
    check(
        diff
        == (inputs["a"] + inputs["b"])
        + (inputs["a"] + inputs["b"] + inputs["c"])
        - (inputs["a"] + inputs["b"])
        - (inputs["c"] + inputs["d"]),
        "difference of orders",
    )
    if diff == 0:
        inputs["d"] += 1
        kl, kp = _k(inputs, left), _k(inputs, pair)
        diff = kl - kp
    check(diff != 0, "the two orders differ")
    p = Prompt()
    p.t("In a floating-point model, every operation returns the exact result of its two inputs multiplied by ")
    p.m(Fm("(1+\\delta)", "(1 + δ)")).t(", where ").m(Fm("\\delta", "δ")).t(
        " is a different number for each operation and "
    ).m(Fm("|\\delta|\\le u", "|δ| ≤ u")).t(". The inputs are stored exactly, and terms of order ").m(
        Fm("u^{2}", "u^2")
    ).t(" and smaller are ignored. The numbers ").m(_inputs_fm(inputs)).t(" are added in two ways. ")
    p.t("Order one adds them left to right: ").m(Fm("((a+b)+c)+d")).t(". Order two adds them in pairs: ").m(
        Fm("(a+b)+(c+d)")
    )
    p.t(". Order one has worst-case absolute error ").m(Fm("k_{1}u", "k1 u")).t(" and order two has ").m(
        Fm("k_{2}u", "k2 u")
    ).t(". Find ").m(Fm("k_{1}-k_{2}", "k1 - k2")).t(".")
    ans = _value(diff)
    atxt = _number_text(diff)
    env_l, coef_l = _run(inputs, left)
    env_p, coef_p = _run(inputs, pair)
    wrong = [
        ("order_two_minus_order_one", -diff),
        ("counts_operations", Fraction(0)),
        ("only_final_rounding_differs", abs(env_l["r"]) - abs(env_p["r"])),
        ("first_partial_sums_only", abs(env_l["s1"]) + abs(env_l["s2"]) - abs(env_p["p1"]) - abs(env_p["p2"])),
        ("sum_of_k", kl + kp),
    ]
    cands = [t for _, w in wrong for t in [_like(w, atxt)] if t is not None]
    distractors = pick_distractors(atxt, cands, n=3, tolerance=GRADER_TOLERANCE)
    if len(distractors) < 2:
        raise _TooFewWrongValues

    return dict(
        text=p.text,
        latex=p.latex,
        answer=ans,
        answer_text=atxt,
        working=[f"k1 = {_number_text(kl)}", f"k2 = {_number_text(kp)}", f"k1 - k2 = {atxt}"],
        distractors=distractors,
    )


def _exam_horner(rng) -> dict:
    for _ in attempts(500, "a Horner trace"):
        inputs = {"a": rng.randint(1, 6), "b": rng.randint(1, 9), "c": rng.randint(1, 9), "x": rng.randint(2, 8)}
        ops = [("t1", "*", "a", "x"), ("t2", "+", "t1", "b"), ("t3", "*", "t2", "x"), ("r", "+", "t3", "c")]
        if _terminating(_k(inputs, ops)):
            break
    return _build(
        rng,
        inputs,
        ops,
        story="This evaluates a polynomial by Horner's rule. Compute ",
    )


def _exam_long(rng) -> dict:
    for _ in attempts(5000, "a long trace"):
        inputs = {"a": rng.randint(2, 9), "b": rng.randint(2, 9), "c": rng.randint(2, 9)}
        ops = [
            ("s", "+", "a", "b"),
            ("t", "*", "s", "c"),
            ("w", "-", "t", "s"),
            ("q", "/", "w", Fraction(rng.choice(_DIVISORS))),
            ("z", "*", "q", "s"),
            ("r", "+", "z", "t"),
        ]
        env, _ = _run(inputs, ops)
        if any(v <= 0 for v in env.values()):
            continue
        if _terminating(_k(inputs, ops)):
            break
    return _build(rng, inputs, ops, story="Compute ")


FAMILIES = {
    "easy": [_easy],
    "medium": [_medium],
    "hard": [_hard],
    "exam": [_exam_orders, _exam_horner, _exam_long],
}


@generator("float_error_budget", level=16, presets=_PRESETS)
def float_error_budget(rng, variant="easy"):
    """`ucv1-floating-point-error-budget`. The answer is a number, graded as a number."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    for _ in attempts(200, "a trace with usable wrong values"):
        try:
            return rng.choice(FAMILIES[variant])(rng)
        except _TooFewWrongValues:
            continue
    raise AssertionError("unreachable: attempts raises when it runs out")
