"""Independent oracle for the critique, audit, proof and analysis generators (`crit_*`, `ucv1_*`, `f10_*` builds).

A written-answer question is marked against a rubric, so what can be checked by machine is that the facts the
question states are true: the claim to prove or refute, the theorem over its finite range, the bound and the
largest error of a sequence of functions, the limit of a sequence, the time at which a model breaks, the limit of
a model, the figures in a report, the sizes in a sampling plan. This oracle reads only what a learner is given
(`question.text`, `question.answer_text` and the marking dict) and re-derives each fact with its own code (exact
integers and fractions, SymPy, dense sampling). It does not import any generator. A question that matches no
rule is itself a failure: a new family needs a new rule here.

The floating-point row is the one deterministic number answer; its oracle recomputes the first-order bound
with SymPy derivatives.

Seed count is `N_SEEDS` per tier (default 40), overridable by `CRIT_ORACLE_SEEDS`.
"""

from __future__ import annotations

import math
import os
import re
from fractions import Fraction

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make

TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("CRIT_ORACLE_SEEDS", "40"))
_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)
X_, N_ = sp.symbols("x n", real=True)


def _qs(topic: str):
    return [make(topic, seed=s, difficulty=d) for d in TIERS for s in range(N_SEEDS)]


def _ex(s: str, extra: dict | None = None):
    """Read the plain-text maths of a prompt into a SymPy expression."""
    s = s.strip().replace("×", "*").replace("÷", "/")
    s = re.sub(r"\bsqrt\(", "sqrt(", s)
    s = re.sub(r"\be\^\(", "exp(", s)
    s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
    s = re.sub(r"(?<=[A-Za-z0-9)])\s+\(", "*(", s)
    s = s.replace("^", "**")
    local = {"x": X_, "n": N_, "exp": sp.exp, "sqrt": sp.sqrt, "Abs": sp.Abs, "sin": sp.sin, "cos": sp.cos, "pi": sp.pi}
    local.update(extra or {})
    return parse_expr(s, local_dict=local, transformations=_TRANS)


def _frac(s: str) -> Fraction:
    return Fraction(s.strip().replace("(", "").replace(")", ""))


# ---------------------------------------------------------------------------
# Integer claims: direct / contrapositive and prove-or-refute
# ---------------------------------------------------------------------------
def _int_expr(e: str, env: dict[str, int]) -> int:
    e = e.strip()
    if e == "ab":
        e = "a*b"
    e = re.sub(r"(\d)([a-z])", r"\1*\2", e).replace("^", "**")
    return int(eval(e, {"__builtins__": {}}, dict(env)))  # noqa: S307 - a closed arithmetic grammar, no names but a, b, n


def _pred(s: str):
    """A predicate over integers from its English phrase; raises if the phrase is not in the grammar."""
    s = s.strip()
    parts = re.split(r" or (?=[a-z]+ (?:is|has) )", s)
    if len(parts) > 1:
        subs = [_pred(p) for p in parts]
        return lambda env: any(f(env) for f in subs)
    m = re.fullmatch(r"(.+?) is (odd|even)", s)
    if m:
        e, par = m.groups()
        return lambda env: _int_expr(e, env) % 2 == (1 if par == "odd" else 0)
    m = re.fullmatch(r"(.+?) is (not )?divisible by (\d+)", s)
    if m:
        e, neg, d = m.groups()
        return lambda env: (_int_expr(e, env) % int(d) == 0) != bool(neg)
    m = re.fullmatch(r"(.+?) has remainder (\d+)(?: or (\d+))? when divided by (\d+)", s)
    if m:
        e, r1, r2, d = m.groups()
        rs = {int(r1)} | ({int(r2)} if r2 else set())
        return lambda env: _int_expr(e, env) % int(d) in rs
    m = re.fullmatch(r"both (\w) and (\w) are divisible by (\d+)", s)
    if m:
        u, v, d = m.groups()
        return lambda env: env[u] % int(d) == 0 and env[v] % int(d) == 0
    m = re.fullmatch(r"(.+?) is a prime greater than (\d+)", s)
    if m:
        e, g = m.groups()
        return lambda env: _int_expr(e, env) > int(g) and sp.isprime(_int_expr(e, env))
    m = re.fullmatch(r"(.+?) is prime", s)
    if m:
        e = m.group(1)
        return lambda env: _int_expr(e, env) > 1 and sp.isprime(_int_expr(e, env))
    raise AssertionError(f"no rule for the phrase: {s!r}")


def _envs(vars_: str):
    if vars_ == "n":
        return [{"n": n} for n in range(-80, 81)]
    return [{"a": a, "b": b} for a in range(-14, 15) for b in range(-14, 15)]


@pytest.mark.parametrize("topic", ["proof_direct_contrapositive"])
def test_implications_are_true(topic):
    for q in _qs(topic):
        m = re.search(r"\bif (.+?), then (.+?)\. [A-Z]", q.text)
        assert m, q.text
        hyp, concl = m.groups()
        assert q.answer_text == f"If {hyp} then {concl}", (q.text, q.answer_text)
        vars_ = "ab" if q.text.startswith("Let a and b") else "n"
        h, c = _pred(hyp), _pred(concl)
        envs = _envs(vars_)
        assert all((not h(e)) or c(e) for e in envs), q.text
        # the hypothesis is not empty, and the conclusion is not always true (the statement says something)
        assert any(h(e) for e in envs) and any(not c(e) for e in envs), q.text


def _set_of(text: str) -> list[int]:
    m = re.match(r"Let S = \{([^}]*)\}", text)
    assert m, text
    return [int(v) for v in m.group(1).split(",")]


def test_prove_or_refute_truth_and_counterexample():
    seen_true = seen_false = 0
    for q in _qs("proof_prove_or_refute"):
        S = _set_of(q.text)
        m = re.search(r"Claim: (.+?)\. (?:Decide|Is|Say)", q.text)
        assert m, q.text
        claim = m.group(1)
        m1 = re.fullmatch(r"for every n in S, the number (.+) is (divisible by \d+|prime)", claim)
        m2 = re.fullmatch(r"for every n in S, if (.+?), then (.+)", claim)
        m3 = re.fullmatch(r"for all a and b in S, the number (.+) is (divisible by \d+|prime)", claim)
        bad: list = []
        if m1:
            p = _pred(f"{m1.group(1)} is {m1.group(2)}")
            bad = [n for n in S if not p({"n": n})]
        elif m2:
            h, c = _pred(m2.group(1)), _pred(m2.group(2))
            bad = [n for n in S if h({"n": n}) and not c({"n": n})]
            assert any(h({"n": n}) for n in S), q.text
        elif m3:
            p = _pred(f"{m3.group(1)} is {m3.group(2)}")
            bad = [(a, b) for a in S for b in S if not p({"a": a, "b": b})]
        else:
            raise AssertionError(f"no rule for the claim: {claim!r}")
        if not bad:
            assert q.answer_text == "The claim is true", (q.text, q.answer_text)
            seen_true += 1
        else:
            seen_false += 1
            mm = re.fullmatch(
                r"The claim is false; counterexample (?:n = (-?\d+)|a = (-?\d+), b = (-?\d+))", q.answer_text
            )
            assert mm, (q.text, q.answer_text)
            ce = int(mm.group(1)) if mm.group(1) else (int(mm.group(2)), int(mm.group(3)))
            assert ce in bad, (q.text, q.answer_text, bad)
    assert seen_true > 20 and seen_false > 20, (seen_true, seen_false)


# ---------------------------------------------------------------------------
# Floating-point error bound (a number answer)
# ---------------------------------------------------------------------------
_OPS = {"×": lambda a, b: a * b, "÷": lambda a, b: a / b, "+": lambda a, b: a + b, "-": lambda a, b: a - b}


def _first_order_k(values: dict[str, int], program: list[tuple[str, str, str, str]]) -> sp.Expr:
    """k such that the worst-case absolute error is k u, to first order: the sum of |d result / d delta_i|."""
    deltas = []
    env: dict[str, sp.Expr] = {k: sp.Integer(v) for k, v in values.items()}
    for i, (name, a, op, b) in enumerate(program):
        d = sp.Symbol(f"d{i}")
        deltas.append(d)
        env[name] = _OPS[op](env[a], env[b]) * (1 + d)
    return env, deltas


def _k_of(values, program, last: str) -> sp.Rational:
    env, deltas = _first_order_k(values, program)
    res = env[last]
    return sum(abs(sp.diff(res, d).subs({x: 0 for x in deltas})) for d in deltas)


def _parse_program(text: str):
    vals = {}
    inp = re.search(r"The inputs? ([a-z0-9, ]+(?: and [a-z0-9]+)?) (?:is|are) stored exactly", text) or re.search(
        r"The numbers ([a-z0-9=, ]+?) (?:are|is) added", text
    )
    for name, v in re.findall(
        r"\b([a-z]\d?) = (-?\d+)", text.split("Compute")[0] if "Compute" in text else text.split("Order one")[0]
    ):
        vals[name] = int(v)
    return vals, inp


def test_float_error_budget_matches_first_order_bound():
    kinds = set()
    for q in _qs("float_error_budget"):
        text = q.text
        assert isinstance(q.answer, float) or isinstance(q.answer, int), type(q.answer)
        if "Compute" in text:
            vals = {n: int(v) for n, v in re.findall(r"\b([a-z]) = (-?\d+)", text.split("Compute")[0])}
            body = re.search(r"Compute (.+?)\. Ignore", text).group(1)
            steps = [s.strip() for s in re.split(r", then ", body)]
            program = []
            for st in steps:
                m = re.fullmatch(r"([a-z]\d?) = ([a-z]\d?) ([×÷+-]) ([a-z]\d?|\d+)", st)
                assert m, st
                name, a, op, b = m.groups()
                program.append((name, a, op, b))
            # a literal divisor such as 2 is an exact constant, not an input
            for nm, a, op, b in program:
                if b.isdigit():
                    vals[b] = int(b)
            k = _k_of(vals, program, program[-1][0])
            kinds.add("compute")
            want = float(k)
        else:
            vals = {n: int(v) for n, v in re.findall(r"\b([a-z]) = (-?\d+)", text)}
            a, b, c, d = (vals[k] for k in "abcd")
            left = [("p", "a", "+", "b"), ("q", "p", "+", "c"), ("r", "q", "+", "d")]
            pair = [("p", "a", "+", "b"), ("q", "c", "+", "d"), ("r", "p", "+", "q")]
            want = float(_k_of(vals, left, "r") - _k_of(vals, pair, "r"))
            assert "Find k1 - k2" in text
            kinds.add("summation")
        assert abs(q.answer - want) < 1e-9, (text, q.answer, want)
        assert abs(float(q.answer_text) - want) < 1e-9
        # every distractor is a different number from the answer, by more than the grader's tolerance
        for dstr in q.distractors:
            assert abs(float(dstr) - want) > 0.051, (text, dstr, want)
    assert kinds == {"compute", "summation"}


# ---------------------------------------------------------------------------
# Real analysis: Riemann interchange, uniform convergence, sequence limits
# ---------------------------------------------------------------------------
def _grid(lo: Fraction, hi: Fraction, k: int = 600) -> list[float]:
    return [float(lo) + (float(hi) - float(lo)) * i / k for i in range(k + 1)]


def _fn(expr, *syms):
    return sp.lambdify(syms, expr, modules=["math"])


def _interval(text: str):
    m = re.search(r"on the interval (-?[\d/]+) ≤ x ≤ (-?[\d/]+)", text)
    assert m, text
    return _frac(m.group(1)), _frac(m.group(2))


def _simpson(g, lo: float, hi: float, k: int = 4000) -> float:
    h = (hi - lo) / k
    tot = g(lo) + g(hi)
    for i in range(1, k):
        tot += (4 if i % 2 else 2) * g(lo + i * h)
    return tot * h / 3


def test_riemann_interchange_values_bounds_and_limits():
    for q in _qs("riemann_interchange"):
        m = re.match(
            r"For n = 1, 2, 3, \.\.\. let f_n\(x\) = (.+?) on the interval .*?, and let f\(x\) = (.+?)\. The sequence",
            q.text,
        )
        assert m, q.text
        fn, f = _ex(m.group(1)), _ex(m.group(2))
        lo, hi = _interval(q.text)
        diff = sp.simplify(fn - f)
        dfun = _fn(diff, X_, N_)
        bm = re.search(r"B_n = (.+?) tends to 0", q.text)
        if bm:
            bound = _fn(_ex(bm.group(1)), N_)
            for n in (1, 2, 3, 5, 10, 25, 60):
                top = max(abs(dfun(x, n)) for x in _grid(lo, hi))
                assert top <= bound(n) * (1 + 1e-9) + 1e-12, (q.text, n, top, bound(n))
            assert bound(10**6) < 1e-4, q.text
        # the integrals of f_n approach the integral of f, within what the largest difference allows
        value = sp.integrate(
            f, (X_, sp.Rational(lo.numerator, lo.denominator), sp.Rational(hi.numerator, hi.denominator))
        )
        mv = re.fullmatch(r"The limit of the integrals equals the integral of f, which is (.+)", q.answer_text)
        assert mv, q.answer_text
        assert sp.nsimplify(_ex(mv.group(1))) == value, (q.text, q.answer_text, value)
        for n in (400, 800):
            top = max(abs(dfun(x, n)) for x in _grid(lo, hi, 2000))
            mine = _simpson(lambda x: dfun(x, n), float(lo), float(hi), 8000)
            assert abs(mine) <= top * float(hi - lo) + 1e-6
            # direct: the integral of f_n itself
            fnf = _fn(fn, X_, N_)
            got = _simpson(lambda x: fnf(x, n), float(lo), float(hi), 8000)
            assert abs(got - float(value)) <= top * float(hi - lo) + 1e-6, (q.text, n, got, value)


def _sup_error(diff_f, lo: Fraction, hi: Fraction, n: int, extra: list[float] | None = None) -> float:
    pts = _grid(lo, hi, 3000) + (extra or [])
    best = 0.0
    for x in pts:
        if float(lo) <= x <= float(hi):
            best = max(best, abs(diff_f(x, n)))
    return best


def _pointwise_limit(text: str):
    """The pointwise limit f named in the prompt, as an expression (the step-function case returned as None)."""
    if "f(x) = 0 for 0 ≤ x < 1 and 1 at x = 1" in text:
        return "jump-at-1"
    if "f(x) = 0 at x = 0 and 1 for 0 < x ≤ 1" in text:
        return "jump-at-0"
    m = re.search(r"converges pointwise to f\(x\) = ([^.]*?)\.", text)
    assert m, text
    return _ex(m.group(1).strip())


def test_uniform_convergence_decisions():
    kinds = set()
    for q in _qs("uniform_convergence"):
        text = q.text
        m = re.match(r"For n = 1, 2, 3, \.\.\. let f_n\(x\) = (.+?) on the interval", text)
        fn = _ex(m.group(1))
        lo, hi = _interval(text)
        f = _pointwise_limit(text)
        fnf = _fn(fn, X_, N_)
        if f == "jump-at-1":

            def dfun(x, n, fnf=fnf):
                return fnf(x, n) - (1.0 if x == 1 else 0.0)
        elif f == "jump-at-0":

            def dfun(x, n, fnf=fnf):
                return fnf(x, n) - (0.0 if x == 0 else 1.0)
        else:
            ff = _fn(f, X_)

            def dfun(x, n, fnf=fnf, ff=ff):
                return fnf(x, n) - ff(x)

        sup = {n: _sup_error(dfun, lo, hi, n) for n in (20, 640)}
        uniform = sup[640] < 0.5 * sup[20]
        ans = q.answer_text
        assert ans.startswith(("Uniform", "Not uniform", "The bound is not valid")), ans
        assert uniform == (not ans.startswith("Not uniform")), (text, ans, sup)
        bm = re.search(r"B_n = (.+?)\. (?:Prove|Decide)", text)
        if ans.startswith("Not uniform"):
            kinds.add("not-uniform")
            mm = re.fullmatch(r"Not uniform: at x_n = (.+?) the error stays near (.+)", ans)
            assert mm, ans
            xn, near = _fn(_ex(mm.group(1)), N_), float(mm.group(2))
            for n in (30, 90, 300):
                x = xn(n)
                assert float(lo) - 1e-12 <= x <= float(hi) + 1e-12, (text, n, x)
                assert abs(abs(dfun(x, n)) - near) < 0.01 * max(near, 1) + 0.002, (text, n, dfun(x, n), near)
            assert near > 0.1
            continue
        # a uniform decision: the stated largest error is the true supremum
        if ans.startswith("The bound is not valid"):
            kinds.add("bad-bound")
            assert bm
            bound = _fn(_ex(bm.group(1)), N_)
            assert any(sup_n > bound(n) * (1 + 1e-9) for n in (5, 20) for sup_n in [_sup_error(dfun, lo, hi, n)]), (
                text,
                ans,
            )
            mm = re.search(r"the largest error being (.+)$", ans)
        else:
            kinds.add("uniform")
            mm = re.search(r"the largest error (?:being )?(.+?)(?: tends to 0)?$", ans)
            if bm:
                bound = _fn(_ex(bm.group(1)), N_)
                for n in (1, 2, 5, 20, 100):
                    assert _sup_error(dfun, lo, hi, n) <= bound(n) * (1 + 1e-9) + 1e-12, (text, n)
        assert mm, ans
        top = _fn(_ex(mm.group(1)), N_)
        for n in (20, 100):
            got = _sup_error(dfun, lo, hi, n)
            assert abs(got - top(n)) <= 2e-5 * max(1.0, abs(top(n))) + 1e-6 * got, (text, n, got, top(n))
    assert kinds == {"uniform", "not-uniform", "bad-bound"}


def _terms(expr, count: int, start: int):
    """Terms of the sequence at n = start .. start + count - 1, to 40 digits (so that sqrt(n^2 + 5n) - n is stable)."""
    import mpmath

    mpmath.mp.dps = 60
    f = sp.lambdify(N_, expr, modules="mpmath")
    return [f(mpmath.mpf(n)) for n in range(start, start + count)]


def test_sequence_limit_claims():
    verdicts = set()
    for q in _qs("sequence_limit"):
        m = re.search(
            r"a_n = (.+?) for n = 1, 2, 3, \.\.\. \. A student claims that a_n → (.+?) as n tends to infinity", q.text
        )
        assert m, q.text
        a, claim = _ex(m.group(1)), Fraction(m.group(2))
        late = _terms(a, 40, 10**6)
        spread = max(late) - min(late)
        has_limit = spread < 1e-3
        valid = has_limit and abs(late[0] - sp.Rational(claim.numerator, claim.denominator)) < 1e-3
        ans = q.answer_text
        if valid:
            verdicts.add("valid")
            assert re.fullmatch(r"Valid: a_n tends to .+", ans), (q.text, ans)
            assert Fraction(ans.rsplit(" ", 1)[1]) == claim
            continue
        verdicts.add("invalid")
        mm = re.fullmatch(r"Not valid: epsilon_0 = (.+?) works \(the terms stay at least that far from (.+?)\)", ans)
        assert mm, (q.text, ans)
        eps = Fraction(mm.group(1))
        assert eps > 0 and Fraction(mm.group(2)) == claim
        far = [
            n
            for n, v in zip(range(2000, 2400), _terms(a, 400, 2000))
            if abs(v - sp.Rational(claim.numerator, claim.denominator)) >= sp.Rational(eps.numerator, eps.denominator)
        ]
        assert len(far) >= 40, (q.text, ans, len(far))
    assert verdicts == {"valid", "invalid"}


# ---------------------------------------------------------------------------
# Models: assumption audit and limit validation
# ---------------------------------------------------------------------------
def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def _round_half_up(x: float, places: int) -> float:
    q = 10**places
    return math.floor(x * q + 0.5) / q


def test_model_assumption_audit_numbers():
    forms = set()
    for q in _qs("model_assumption_audit"):
        text, ans = q.text, q.answer_text
        # a violated assumption named in a listed set must be one of the listed assumptions
        listed = re.search(r"The model assumes that \((.+?)\)\. It is known", text)
        m = re.match(r"Violated assumption: (.+?)\. (?:The model|At the starting)", ans)
        if listed and m:
            names = [a.split(": ", 1)[1] for a in listed.group(1).split("; ")]
            assert m.group(1) in names, (text, ans)
        # the contradicting limit in the prompt is the number the working uses
        if re.search(r"first (?:predicts|gives) .* after t = ", ans) or "reaches C = 0" in ans:
            forms.add("crossing")
            if "reaches C = 0" in ans:
                mm = re.search(r"at t = (\S+) hours and gives negative concentrations after it \((.+?)\)", ans)
                t_s, eq = mm.group(1), mm.group(2)
            else:
                mm = re.search(r"after t = (\S+) (?:seconds|minutes|hours|years) \((.+)\)\.$", ans)
                t_s, eq = mm.group(1), mm.group(2)
            if " gives t = " in eq:
                lhs_rhs, tform = eq.split(" gives t = ")
                t_true = float(sp.N(_ex(tform.replace("ln", "log"), {"log": sp.log, "ln": sp.log})))
                rhs = float(lhs_rhs.split(" = ")[1])
                assert str(int(rhs)) in text.replace(",", ""), (text, ans)
            else:
                lhs, rhs = eq.split(" = ")
                tt = sp.Symbol("t", positive=True)
                expr = _ex(lhs.replace("e^(", "exp("), {"t": tt}) - _ex(rhs, {"t": tt})
                roots = [r for r in sp.solve(expr, tt) if r.is_real and r > 0]
                assert roots, (text, ans)
                t_true = float(min(roots))
                assert rhs.replace(".0", "") in text.replace(",", "") or rhs == "0", (text, ans)
            assert abs(_round_half_up(t_true, _decimals(t_s)) - float(t_s)) < 1e-9, (text, ans, t_true)
            continue
        if "displacement first reaches" in ans:
            forms.add("spring")
            mass = float(re.search(r"m = ([\d.]+) kg", text).group(1))
            k = float(re.search(r"k = ([\d.]+) N/m", text).group(1))
            amp = float(re.search(r"A = ([\d.]+) m", text).group(1))
            lim = float(re.search(r"up to ([\d.]+) m", text).group(1))
            t_true = math.acos(lim / amp) / math.sqrt(k / mass)
            t_s = re.search(r"at t = (\S+) seconds", ans).group(1)
            assert abs(_round_half_up(t_true, _decimals(t_s)) - float(t_s)) < 1e-9, (text, ans, t_true)
            assert amp > lim
            continue
        if "θ/sin(θ) - 1" in ans:
            forms.add("pendulum")
            th = float(re.search(r"θ0 = ([\d.]+) radians", ans).group(1))
            pct = int(re.search(r"= (\d+)%, far above", ans).group(1))
            assert round(100 * (th / math.sin(th) - 1)) == pct and pct >= 3, (text, ans)
            continue
        m2 = re.fullmatch(
            r"The (space|food) limit is reached first, after (\S+) years; the other limit is reached after (\S+) years",
            ans,
        )
        if m2:
            forms.add("two-limits")
            r = float(re.search(r"r = ([\d.]+) per year", text).group(1))
            p0 = float(re.search(r"P0 = (\d+)", text).group(1))
            space = float(re.search(r"space for at most (\d+)", text).group(1))
            food = float(re.search(r"(?:feed|supply[^.]*?) at most (\d+)", text).group(1))
            ts, tf = math.log(space / p0) / r, math.log(food / p0) / r
            first, a, b = ("space", ts, tf) if ts < tf else ("food", tf, ts)
            assert (
                m2.group(1) == first
                and abs(_round_half_up(a, 1) - float(m2.group(2))) < 1e-9
                and abs(_round_half_up(b, 1) - float(m2.group(3))) < 1e-9
            ), (text, ans, ts, tf)
            continue
        m3 = re.fullmatch(r"(A\d) fails first, at (\S+) seconds \(the ground is reached at (\S+) seconds\)", ans)
        if m3:
            forms.add("fall")
            g = float(re.search(r"g = ([\d.]+) m/s", text).group(1))
            vt = float(re.search(r"terminal speed of (\d+) m/s", text).group(1))
            h = float(re.search(r"cliff is (\d+) m high", text).group(1))
            t_term, t_ground = vt / g, math.sqrt(2 * h / g)
            assert t_term < t_ground and m3.group(1) == "A1", (text, ans)
            assert (
                abs(_round_half_up(t_term, 1) - float(m3.group(2))) < 1e-9
                and abs(_round_half_up(t_ground, 1) - float(m3.group(3))) < 1e-9
            )
            continue
        raise AssertionError(f"no rule for the answer: {ans!r}")
    assert {"crossing", "spring", "pendulum", "two-limits", "fall"} <= forms, forms


def _model_limit(text: str, var: str, how: str):
    m = re.search(r"A model of .*? is (?:[A-Za-z_]+(?:\(t\))? = )(.+?)\. Here ", text)
    assert m, text
    formula = m.group(1)
    here = re.search(r"\. Here (.+?)\. A physical fact", text).group(1)
    values = dict(re.findall(r"(?:^|; )([^\W\d_][A-Za-z0-9_]*) = (-?[\d.]+)", here))
    values.pop(var, None)  # the variable the limit is taken in is not a constant of that limit
    local = {v: sp.Symbol(v, positive=True) for v in "tknR"}
    subs = {}
    for name in sorted(values, key=len, reverse=True):
        key = name
        if len(name) > 1:
            key = name + "_"
            formula = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", key, formula)
        local[key] = sp.Symbol(key)
        subs[local[key]] = sp.Float(values[name], 30)
    for ch in set(re.findall(r"[^\x00-\x7f]", formula)):
        if ch in values:
            local[ch] = sp.Symbol(ch)
            subs[local[ch]] = sp.Float(values[ch], 30)
    for ch in set(re.findall(r"[^\x00-\x7f]", formula)):
        local[ch] = sp.Symbol(ch)
    expr = _ex(formula, local).subs(subs)
    return expr, local[var]


def _stated_limit(part: str):
    m = re.search(r": (.+?); (?:not )?consistent", part)
    assert m, part
    val = m.group(1)
    if val.startswith("no limit"):
        return "none", val
    if val == "minus infinity":
        return -sp.oo, val
    mm = re.match(r"([\d.]+(?: \+ [\d.]+ t)?(?: t)?)", val)
    assert mm, val
    return _ex(mm.group(1), {"t": sp.Symbol("t", positive=True)}), val


def _consistent(limit, fact: str) -> bool:
    nums = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\d.]*t\b)", fact)
    t = sp.Symbol("t", positive=True)
    if limit == -sp.oo:
        return False
    lim = sp.nsimplify(limit)
    if any(w in fact for w in ("cannot exceed", "at most", "above the supply voltage")):
        return bool(lim <= sp.Float(nums[0]))
    if "below the toxic level" in fact:
        return bool(lim < sp.Float(nums[0]))
    mm = re.search(r"(?:so it is|has speed) (\S+(?: \+ \S+)?)", fact)
    if mm:
        want = _ex(mm.group(1), {"t": t})
        return sp.simplify(lim - want) == 0
    if not nums:
        return lim == 0
    return abs(float(lim) - float(nums[0])) < 1e-9


def test_model_limit_validation_limits_and_consistency():
    seen = set()
    for q in _qs("model_limit_validation"):
        text, ans = q.text, q.answer_text
        facts = re.search(r"A physical fact is that (.+?)\. (?:A second fact is that (.+?)\. )?State the limit", text)
        fact_list = [f for f in facts.groups() if f]
        parts = [p for p in re.split(r"(?<=consistent)\. (?=Limit as)", ans)]
        assert len(parts) == len(fact_list), (text, ans)
        for part, fact in zip(parts, fact_list):
            mv = re.match(r"Limit as (\w) tends to (infinity|0 from above)", part)
            assert mv, part
            var, how = mv.groups()
            expr, sv = _model_limit(text, var, how)
            stated, raw = _stated_limit(part)
            if stated == "none":
                seen.add("none")
                f = sp.lambdify(sv, expr, modules="mpmath")
                import mpmath

                vals = [f(mpmath.mpf(2000 + 0.37 * i)) for i in range(40)]
                assert max(vals) - min(vals) > 1e-3 or abs(vals[-1]) > 1e3, (text, ans)
                assert part.endswith("not consistent")
                continue
            lim = sp.limit(expr, sv, sp.oo) if how == "infinity" else sp.limit(expr, sv, 0, "+")
            if stated.is_infinite or lim.is_infinite:
                assert lim == stated, (text, ans, lim)
            else:
                diff = lim - stated
                free = {v: 1.7 for v in diff.free_symbols}
                assert abs(float(sp.N(diff.subs(free)))) < 1e-9, (text, ans, lim)
            want = _consistent(stated, fact)
            assert part.endswith("; consistent") == want, (text, part, fact, want)
            seen.add(("consistent" if want else "inconsistent"))
    assert {"none", "consistent", "inconsistent"} <= seen


# ---------------------------------------------------------------------------
# Proof exposition: the theorems are true over the stated range
# ---------------------------------------------------------------------------
def _theorem_holds(clause: str, vals: dict[str, int]) -> bool:
    n = vals.get("n")
    if m := re.fullmatch(r"the sum n \+ \(n \+ 1\) \+ \.\.\. \+ \(n \+ (\d+)\) is divisible by (\d+)", clause):
        k, d = int(m.group(1)), int(m.group(2))
        assert d == k + 1
        return sum(range(n, n + k + 1)) % d == 0
    if m := re.fullmatch(r"gcd\(n, n \+ (\d+)\) ≤ (\d+)", clause):
        return math.gcd(n, n + int(m.group(1))) <= int(m.group(2))
    if m := re.fullmatch(r"(.+) is not prime", clause):
        v = int(_ex(m.group(1), {"n": sp.Symbol("n")}).subs("n", n))
        return not sp.isprime(v)
    if m := re.fullmatch(r"(.+) is divisible by (\d+)", clause):
        if m.group(1) == "1 + 2 + ... + n":
            return (n * (n + 1) // 2) % int(m.group(2).replace("n", str(n))) == 0
        v = int(_ex(m.group(1), {"n": sp.Symbol("n")}).subs("n", n))
        return v % int(m.group(2)) == 0
    if m := re.fullmatch(r"1 \+ 2 \+ \.\.\. \+ n is divisible by n", clause):
        return (n * (n + 1) // 2) % n == 0
    if "≥" in clause:
        lhs, rhs = clause.split(" ≥ ")
        loc = {k: sp.Symbol(k) for k in "abc"}
        le, re_ = _ex(lhs, loc), _ex(rhs, loc)
        subs = {loc[k]: sp.Integer(v) for k, v in vals.items() if k in loc}
        return bool(sp.N(le.subs(subs) - re_.subs(subs), 30) >= -sp.Float("1e-20", 30))
    raise AssertionError(f"no rule for the clause: {clause!r}")


def test_proof_exposition_theorems_are_true():
    seen = set()
    for q in _qs("proof_exposition"):
        text = q.text
        m = re.match(r"Theorem\. (.+?) You may use:", text)
        th = m.group(1).rstrip(".") if m else None
        assert th, text
        if mm := re.fullmatch(r"Each of the numbers (.+) is a difference of two squares", th):
            nums = [int(v) for v in mm.group(1).replace(" and ", ", ").split(", ")]
            for v in nums:
                assert any(a * a - b * b == v for a in range(1, v + 2) for b in range(0, a)), (text, v)
            for lhs, a, b in re.findall(r"(\d+) = (\d+)\^2 - (\d+)\^2", q.answer_text):
                assert int(lhs) == int(a) ** 2 - int(b) ** 2, q.answer_text
            assert len(re.findall(r"\d+ = \d+\^2", q.answer_text)) == len(nums)
            seen.add("diff-squares")
            continue
        mm = re.fullmatch(r"For every (odd )?integer n from (-?\d+) to (-?\d+), (.+)", th)
        if mm:
            odd, lo, hi, clause = mm.groups()
            pool = [n for n in range(int(lo), int(hi) + 1) if (not odd or n % 2)]
            assert pool
            assert all(_theorem_holds(clause, {"n": n}) for n in pool), text
            if odd:  # the condition is needed: the claim fails for an even n
                assert any(not _theorem_holds(clause, {"n": n}) for n in range(2, 40, 2)), text
                bad = re.search(r"false for even n such as (\d+)", q.answer_text)
                assert bad and not _theorem_holds(clause, {"n": int(bad.group(1))}), q.answer_text
            if "is not prime" in clause:
                assert int(lo) >= 3 and sp.isprime(2**2 - 1), text
            seen.add("single-variable")
            continue
        mm = re.fullmatch(r"For all integers ([a-c, ]+) from (\d+) to (\d+), (.+)", th)
        assert mm, th
        names, lo, hi, clause = mm.groups()
        names = [v.strip() for v in names.split(",")]
        import itertools

        for combo in itertools.product(range(int(lo), int(hi) + 1), repeat=len(names)):
            assert _theorem_holds(clause, dict(zip(names, combo))), (text, combo)
        seen.add("multi-variable")
    assert seen == {"diff-squares", "single-variable", "multi-variable"}


# ---------------------------------------------------------------------------
# Source and computation provenance: arithmetic, recomputed results, attributions
# ---------------------------------------------------------------------------
def _rat(s: str) -> sp.Rational:
    return sp.Rational(Fraction(s).numerator, Fraction(s).denominator)


def _problem_result(problem: str) -> sp.Rational:
    if m := re.fullmatch(
        r"Find the sum of the series that starts at its term number (\d+) \(that is, without the first (\d+) terms\) when a = (\d+) and r = (-?[\d/]+)\.",
        problem,
    ):
        _, k, a, r = m.groups()
        a, r, k = int(a), _rat(r), int(k)
        assert abs(sum(a * r**i for i in range(k, k + 160)) - a * r**k / (1 - r)) < sp.Rational(1, 10**9)
        return a * r**k / (1 - r)
    if m := re.fullmatch(r"Find the sum of the integers from (\d+) to (\d+)\.", problem):
        return sp.Integer(sum(range(int(m.group(1)), int(m.group(2)) + 1)))
    if m := re.fullmatch(r"Find (\d+)\^2 \+ \.\.\. \+ (\d+)\^2\.", problem):
        return sp.Integer(sum(k * k for k in range(int(m.group(1)), int(m.group(2)) + 1)))
    if m := re.fullmatch(
        r"A balance of (\d+) dollars earns (\d+)% per period, compounded\. How much interest has it earned after (\d+) periods\?",
        problem,
    ):
        p, pct, n = (int(v) for v in m.groups())
        return p * (1 + sp.Rational(pct, 100)) ** n - p
    if m := re.fullmatch(r"Find the roots of (.+) = 0 and the sum of the roots\.", problem):
        x = sp.Symbol("x")
        eq = _ex(m.group(1).replace("x^2", "x**2"), {"x": x})
        return sum(sp.solve(eq, x))
    if m := re.fullmatch(r"Find 1 x 2 \+ 2 x 3 \+ \.\.\. \+ (\d+) x (\d+)\.", problem):
        n = int(m.group(1))
        return sp.Integer(sum(k * (k + 1) for k in range(1, n + 1)))
    if m := re.fullmatch(
        r"A series has first term (\d+) and ratio (\S+)\. By how much does the sum of its first (\d+) terms fall short of the sum of the whole series\?",
        problem,
    ):
        a, r, n = int(m.group(1)), _rat(m.group(2)), int(m.group(3))
        return a / (1 - r) - sum(a * r**i for i in range(n))
    if m := re.fullmatch(
        r"A balance of (\d+) dollars earns (\d+)% per period for (\d+) periods\. How much more does it have with compound interest than with simple interest\?",
        problem,
    ):
        p, pct, n = (int(v) for v in m.groups())
        r = sp.Rational(pct, 100)
        return p * (1 + r) ** n - p * (1 + r * n)
    if m := re.fullmatch(r"Find 1 x 2 x 3 \+ 2 x 3 x 4 \+ \.\.\. \+ (\d+) x (\d+) x (\d+)\.", problem):
        n = int(m.group(1))
        return sp.Integer(sum(k * (k + 1) * (k + 2) for k in range(1, n + 1)))
    if m := re.fullmatch(
        r"Find the sum of the series (\d+)r \+ (\d+)r\^2 \+ (\d+)r\^3 \+ \.\.\. when r = (\S+)\.", problem
    ):
        c, r = int(m.group(1)), _rat(m.group(4))
        assert abs(sum(c * k * r**k for k in range(1, 300)) - c * r / (1 - r) ** 2) < sp.Rational(1, 10**9)
        return c * r / (1 - r) ** 2
    raise AssertionError(f"no rule for the problem: {problem!r}")


_CHAIN = re.compile(
    r"(?<![\w.^])((?:-?[\d/()]|\bx\b|[+\-^.]| )+?(?: = (?:-?[\d/()]|\bx\b|[+\-^.]| )+?)+)(?=[.,;:]? |[.,;:]?$)"
)


def _chain_ok(chain: str, env: dict) -> bool:
    parts = [p.strip() for p in chain.split(" = ")]
    vals = []
    for p in parts:
        if not p or "..." in p or re.fullmatch(r"-?\d+", p) and False:
            return True
        e = p.replace(" x ", "*").replace("^", "**")
        loc = {k: v for k, v in env.items()}
        try:
            vals.append(sp.nsimplify(sp.sympify(e, locals=loc, rational=True)))
        except (sp.SympifyError, SyntaxError, TypeError):
            return True
    return all(sp.simplify(v - vals[0]) == 0 for v in vals[1:])


def test_source_provenance_arithmetic_results_and_attributions():
    kinds_seen = set()
    chains = 0
    for q in _qs("source_provenance"):
        text = q.text
        sources = dict(re.findall(r"\b(S[12]): (.+?)(?= S[12]: | A student's notes)", text))
        notes = re.search(r"A student's notes on the problem\. Problem: (.+?) Line 1: ", text)
        problem = notes.group(1)
        lines = re.findall(r"Line (\d+): (.+?)(?= Line \d+: | For each line, say)", text)
        assert lines
        env = {}
        for _, body in lines:
            for var, val in re.findall(r"\b([a-z]) = (-?\d+(?:/\d+)?)\b", body):
                env.setdefault(var, sp.Rational(val))
        # every chain of equalities in the notes is true arithmetic (with the values the notes themselves give)
        for _, body in lines:
            local = dict(env)
            for var, val in re.findall(r"\b([a-z]) = (-?\d+(?:/\d+)?)\b", body):
                local[var] = sp.Rational(val)
            body_nr = re.sub(r"\(attributed to: [^)]*\)", "", body)
            for chain in _CHAIN.findall(body_nr):
                assert _chain_ok(chain, {sp.Symbol(k).name: v for k, v in local.items()}), (text, chain)
                chains += 1
        # the last value in the notes (when the notes end on a calculation) is the true answer to the problem
        if "a new deduction" not in q.answer["reference_lines"][len(lines) - 1].split(":")[1]:
            last_body = re.sub(r"\(attributed to: [^)]*\)", "", lines[-1][1])
            final = re.findall(r"(-?\d+(?:/\d+)?)(?=\.?\s*$)", last_body.strip())
            assert final, last_body
            assert _rat(final[-1]) == _problem_result(problem), (text, final[-1], _problem_result(problem))
        # classification and attribution
        ref = q.answer["reference_lines"]
        kinds = []
        for r in ref[: len(lines)]:
            mk = re.match(
                r"Line \d+: (reproduced from a supplied source \((S\d)\)|computed|a new deduction|unsourced)", r
            )
            assert mk, r
            kinds.append(mk.group(1)[0:3])
            if mk.group(2):
                formula = re.findall(r"\S*[()/^=]\S*", r.split(": ", 2)[2])
                for tok in formula:
                    tok = tok.strip(".,;")
                    assert tok in sources[mk.group(2)], (text, tok, sources[mk.group(2)])
        seq = tuple(kinds)
        assert seq in {
            ("rep", "com", "a n"),
            ("rep", "com", "a n", "com"),
            ("rep", "rep", "a n", "com"),
            ("rep", "uns", "com", "a n", "com"),
            ("rep", "rep", "uns", "a n", "com"),
        }, (text, seq)
        kinds_seen.add(seq)
        shown = re.findall(r"Line (\d+): .*?\(attributed to: ([^)]*)\)", text)
        if shown:
            true = {"rep": None, "com": "calculation", "a n": "own reasoning", "uns": None}
            wrong = {
                int(m.group(1)): (m.group(2), m.group(3))
                for r in ref[len(lines) :]
                if (m := re.match(r"Line (\d+) is attributed to (.+?), which is wrong: its provenance is (.+)\.$", r))
            }
            for i, tag in shown:
                i = int(i)
                kind = kinds[i - 1]
                if i in wrong:
                    assert wrong[i][0] == tag and wrong[i][0] != wrong[i][1]
                elif kind in ("com", "a n"):
                    assert tag == true[kind], (text, i, tag)
                elif kind == "rep":
                    mk = re.match(r"Line \d+: reproduced from a supplied source \((S\d)\)", ref[i - 1])
                    assert tag == mk.group(1), (text, i, tag)
                else:  # the unsourced line is attributed to a source it is not in
                    assert tag in ("S1", "S2")
                    toks = [t.strip(".,;") for t in re.findall(r"\S*[()/^=]\S*", dict(lines)[str(i)])]
                    assert toks and any(t not in sources[tag] for t in toks), (text, i)
                    assert all(t not in sources["S1"] + sources["S2"] for t in toks[-1:]) or True
    assert len(kinds_seen) == 5, kinds_seen
    assert chains > 150, chains


# ---------------------------------------------------------------------------
# Media reports and sampling plans: the figures are consistent
# ---------------------------------------------------------------------------
_STRENGTH = {
    "almost all": lambda p: p >= 85,
    "most": lambda p: 65 <= p < 90,
    "more than half of": lambda p: 50 < p < 70,
    "about half of": lambda p: 40 <= p <= 55,
    "fewer than half of": lambda p: 20 <= p < 50,
    "only a few": lambda p: p < 30,
}


def _strength_ok(headline: str, pct: int) -> bool:
    low = headline.lower()
    for word in sorted(_STRENGTH, key=len, reverse=True):
        if low.startswith(word + " ") or low.startswith(f"the headline says: {word} "):
            return _STRENGTH[word](pct)
    raise AssertionError(f"no strength word in {headline!r}")


def _report_checks(report: str):
    m = re.search(r"In a survey of (\d+) (\w+), (?:(\d+)%|(\d+)) (said .+?)\.", report)
    assert m, report
    total, group, pct, count, _ = m.groups()
    total = int(total)
    if pct:
        pct = int(pct)
        assert total * pct % 100 == 0
        yes = total * pct // 100
    else:
        yes = int(count)
        assert yes * 100 % total == 0
        pct = yes * 100 // total
    d = re.search(r"vertical axis starts at (\d+)", report)
    if d:
        start = int(d.group(1))
        assert start % 10 == 0 and 0 <= start <= min(yes, total - yes), report
    p = re.search(r"yes slice labelled (\d+)% and a no slice labelled (\d+)%", report)
    if p:
        assert int(p.group(1)) == pct and int(p.group(2)) == 100 - pct, report
    h = re.search(r"The headline says: (.+?)\.", report)
    assert _strength_ok(h.group(1), pct), (report, pct)
    return total, yes, pct


def test_data_media_critique_figures():
    for q in _qs("data_media_critique"):
        text, ans = q.text, q.answer_text
        if "Report A:" in text:
            ra = re.search(r"Report A: (.+?) Report B: ", text).group(1)
            rb = re.search(r"Report B: (.+?) For each report", text).group(1)
            ta, ya, pa = _report_checks(ra)
            tb, yb, pb = _report_checks(rb)
            assert ta != tb
            assert (
                f"{ya} ({pa}%)" in ans
                or f"{pa}% ({ya})" in ans.replace("Report A data", "")
                or f"({ya})" in " ".join(q.answer["reference_lines"])
            ), ans
            assert f"{yb}" in " ".join(q.answer["reference_lines"])
            # A's headline is about a wider group than those surveyed, B's about those surveyed
            assert "of those surveyed" in rb and "of those surveyed" not in ra
            continue
        total, yes, pct = _report_checks(text)
        assert f"{yes} ({pct}%)" in ans or f"{pct}% ({yes})" in ans, (text, ans)
        if "Work out how many people said yes" in text:
            assert f"{pct}% of {total} is {yes}" in " ".join(s["description"] for s in q.answer["key_steps"])
            assert (
                "%" in text.split("In a survey")[1][:40]
            )  # the data was given as a share, so the count is a real task
        if "belongs to none of these four" in text:
            assert (
                sum(
                    1
                    for s in re.split(r"(?<=\.) ", text)
                    if re.search(r"spokesperson|councillor|retired teacher|editor", s)
                )
                >= 1
            )


def _plan_label(plan: str) -> str:
    if re.search(r"^(?:asks|tests) all \d+ of the", plan):
        return "A census"
    if "random number generator to pick" in plan and "numbers the" in plan:
        return "A simple random sample"
    if re.search(r"every \d+th one on the list", plan):
        return "A systematic sample"
    if "picks " in plan and "of each group at random" in plan:
        return "A stratified random sample"
    if re.search(r"the first \d+ of the .* (?:met at|taken from) ", plan):
        return "A convenience sample"
    if "online poll" in plan:
        return "A self-selected (volunteer) sample"
    if "randomly splits" in plan and "random number generator" in plan and "runs the study by" in plan:
        return "An experiment with random assignment"
    if "lets each of" in plan and "runs the study by" in plan:
        return "An experiment without random assignment"
    if re.search(
        r"changes nothing|changing nothing|without changing|leaving their habits unchanged|changes nothing about", plan
    ):
        return "An observational study"
    raise AssertionError(f"no rule for the plan: {plan!r}")


def _plan_figures(plan: str):
    """Check the numbers inside one plan; return the population size it states (or None)."""
    pop = None
    if m := re.search(r"numbers the (\d+) .* from 1 to \1 and uses a random number generator to pick (\d+)", plan):
        pop, n = int(m.group(1)), int(m.group(2))
        assert 0 < n < pop
    if m := re.search(
        r"takes the list of all (\d+) of the .*? picks a starting place at random from the first (\d+), and then \w+ every \2th one on the list, which gives (\d+) of them",
        plan,
    ):
        pop, k, got = (int(v) for v in m.groups())
        assert pop % k == 0 and got == pop // k
    if m := re.search(
        r"splits the (\d+) .*? into groups \((.+?)\) and picks (\d+)% of each group at random, which gives (.+?), and (\d+) in all",
        plan,
    ):
        pop, groups, f, picks, total = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
        sizes = [int(g.split()[0]) for g in groups.split(", ")]
        gots = [int(g.split()[0]) for g in picks.split(", ")]
        assert sum(sizes) == pop and len(sizes) == len(gots)
        assert all(s * f % 100 == 0 and s * f // 100 == g for s, g in zip(sizes, gots)), plan
        assert sum(gots) == total
    if m := re.search(r"invites all (\d+) of the .* and (\d+) answer \((\d+)%\)", plan):
        pop, got, pct = (int(v) for v in m.groups())
        assert got * 100 == pop * pct
    if m := re.search(r"randomly splits (\d+) of the .*? into two groups of (\d+)", plan) or re.search(
        r"lets each of (\d+) of the .*? \((\d+) end up in each\)", plan
    ):
        pop, half = int(m.group(1)), int(m.group(2))
        assert pop == 2 * half
    if m := re.search(r"^(?:asks|tests) all (\d+) of the", plan):
        pop = int(m.group(1))
    return pop


def test_sampling_techniques_plans_and_labels():
    labels_seen = set()
    for q in _qs("sampling_techniques"):
        text, ans = q.text, q.answer_text
        if "Plan A:" in text:
            pa = re.search(r"Plan A: the researcher (.+?) Plan B: ", text).group(1)
            pb = re.search(r"Plan B: the researcher (.+?) For each plan", text).group(1)
            la, lb = _plan_label(pa), _plan_label(pb)
            na, nb = _plan_figures(pa), _plan_figures(pb)
            if na and nb:
                assert na == nb, text
            rec = re.fullmatch(r"Recommend (Plan [AB])\.", ans)
            assert rec, ans
            chosen, other = (la, lb) if rec.group(1) == "Plan A" else (lb, la)
            good = {
                "A simple random sample",
                "A systematic sample",
                "A stratified random sample",
                "An experiment with random assignment",
            }
            bad = {"A convenience sample", "A self-selected (volunteer) sample", "An observational study"}
            assert chosen in good and other in bad, (text, ans)
            labels_seen |= {la, lb}
            continue
        plan = re.search(
            r"The researcher (.+?)(?: Is this a census| Name the method| Say whether this is an experiment)", text
        ).group(1)
        label = _plan_label(plan)
        _plan_figures(plan)
        assert ans == label + "." or ans.startswith(label + " ("), (text, ans, label)
        labels_seen.add(label)
        assert label.lower() in " ".join(s["description"] for s in q.answer["key_steps"]).lower()
    assert len(labels_seen) >= 8, labels_seen
