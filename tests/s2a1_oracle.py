"""Shared oracle and checks for the second-year integration skills (`tests/test_s2a1_*.py`).

The oracle reads only the question text. `region_constraints` pulls the inequalities out of a stem, `integrand_of`
the integrand or density, and `mc_integral` estimates the integral in plain Cartesian coordinates by Monte Carlo
with a fixed seed. That is far too coarse to be a key, but it is independent of every limit, Jacobian and
coordinate change in the generators: a wrong region or a missing factor of r is off by tens of per cent.
"""

from __future__ import annotations

import re

import numpy as np
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills

x, y, z = sympy.symbols("x y z", real=True)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_RUN = re.compile(r"[0-9xyz²³√+\-*/(). ≤≥]+")


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def read(text: str, symbols=None):
    text = text.strip().rstrip(".").replace("²", "**2").replace("³", "**3").replace("^", "**")
    text = re.sub(r"(?<=[0-9xyz)])√", "*sqrt", text).replace("√", "sqrt")
    local = {"x": x, "y": y, "z": z, "pi": sympy.pi}
    local.update(symbols or {})
    return parse_expr(text, transformations=_TRANSFORMS, local_dict=local)


def region_constraints(text: str):
    """Boolean-valued NumPy functions of (x, y, z), one per inequality in the text."""
    out = []
    for word, op in (("below", "≤"), ("above", "≥")):
        for found in re.finditer(rf"{word} the (?:surface|plane) z = ([^,]+?)(?:,| and|\.)", text):
            out.append((sympy.lambdify((x, y, z), z - read(found.group(1)), "numpy"), op))
    for run in _RUN.findall(text):
        if "≤" not in run and "≥" not in run:
            continue
        pieces = re.split(r"(≤|≥)", run.strip().rstrip("."))
        exprs = [read(p) for p in pieces[0::2]]
        for left, op, right in zip(exprs, pieces[1::2], exprs[1:]):
            rel = left - right  # left op right  <=>  rel op 0
            fn = sympy.lambdify((x, y, z), rel, "numpy")
            out.append((fn, op))
    return out


def inside(constraints, px, py, pz):
    ok = np.ones(px.shape, dtype=bool)
    for fn, op in constraints:
        v = fn(px, py, pz) + 0 * px
        ok &= (v <= 1e-12) if op == "≤" else (v >= -1e-12)
    return ok


def integrand_of(text: str):
    found = re.search(r"∭ (.+?) dV", text)
    if found:
        return read(found.group(1))
    found = re.search(r"[ρδ] = ([^ ]+(?: [+\-] [^ ]+)*?) kilograms", text)
    if found:
        return read(found.group(1))
    return sympy.Integer(1)


def mc_integral(constraints, integrand, seed: int = 12345, count: int = 1_200_000) -> float:
    """Monte Carlo estimate of the integral of `integrand` over the region the constraints define."""
    rng = np.random.default_rng(seed)
    f = sympy.lambdify((x, y, z), integrand, "numpy")
    half = 1.0
    for _ in range(14):
        pts = rng.uniform(-half, half, size=(3, 200_000))
        hit = inside(constraints, *pts)
        if hit.any() and np.abs(pts[:, hit]).max() < 0.55 * half:
            break
        half *= 2
    else:
        raise AssertionError("no bounding box found")
    pts = rng.uniform(-half, half, size=(3, 400_000))
    hit = inside(constraints, *pts)
    lo = pts[:, hit].min(axis=1) - 0.05 * half
    hi = pts[:, hit].max(axis=1) + 0.05 * half
    pts = rng.uniform(lo[:, None], hi[:, None], size=(3, count))
    hit = inside(constraints, *pts)
    values = np.where(hit, f(*pts) + 0 * pts[0], 0.0)
    return float(np.prod(hi - lo) * values.mean())


def mc_matches(text: str, key: float, rel: float = 0.04) -> bool:
    estimate = mc_integral(region_constraints(text), integrand_of(text))
    return abs(estimate - key) <= rel * max(abs(key), 1e-9)


def parse_typed(text: str):
    """An answer typed the way the site shows it (2(-1 + e^4), 3ln(3)/2, 8√2/9, 5π/3) as an exact SymPy value."""
    text = text.strip().replace("^", "**").replace("π", "pi").replace("·", "*").replace("ln(", "log(")
    text = re.sub(r"√(\d+)", r"sqrt(\1)", text)
    text = re.sub(r"\be\b", "E", text)
    text = re.sub(r"(?<=[\d)E])(?=[a-zA-Z(])", "*", text)
    text = re.sub(r"(?<=\))(?=\d)", "*", text)
    return sympy.sympify(text, locals={"pi": sympy.pi, "E": sympy.E})


def typed(value) -> str:
    """The typed spelling of an exact value, as `parse_typed` reads it back."""
    from mathgen.generators._ucv1_calc_markup import value_text

    return value_text(value)


def key_value(answer_text: str) -> float:
    return float(parse_typed(answer_text).evalf())


# ---------------------------------------------------------------------------------------------------------------
# checks shared by every file


def question(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def check_key_and_distractors(sk, topic: str, tier: str, seeds, minimum: int = 3) -> None:
    for seed in seeds:
        q = question(topic, tier, seed)
        assert grade(sk, q, q.answer_text).correct, (seed, q.text, q.answer_text)
        assert len(q.distractors) >= minimum, (seed, q.text)
        assert len(set(q.distractors)) == len(q.distractors) and q.answer_text not in q.distractors
        for text in q.distractors:
            assert not grade(sk, q, text).correct, (seed, q.text, text)
        assert len(q.distractor_values) == len(q.distractors)


def check_exam_shape(sk, topic: str, seeds, marks: list[int], kinds: list[set]) -> None:
    for seed in seeds:
        q = question(topic, "exam", seed)
        assert [p.label for p in q.parts] == list("abcd"[: len(marks)])
        assert [p.marks for p in q.parts] == marks, (seed, [p.marks for p in q.parts])
        assert [p.follow_through is not None for p in q.parts][0] is False
        assert any(p.follow_through for p in q.parts[1:])
        for p, allowed in zip(q.parts, kinds):
            assert p.answer_kind in allowed, (seed, p.label, p.answer_kind)
            assert len(p.distractors) >= 2
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, full)


def check_exam_distractors(sk, topic: str, seeds) -> None:
    for seed in seeds:
        q = question(topic, "exam", seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for text in p.distractors:
                result = grade(sk, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)


def check_follow_through(sk, topic: str, seeds, minimum: int = 20) -> None:
    """A wrong earlier answer, carried correctly into the next part, earns that part's marks as follow-through."""
    checked = 0
    for seed in seeds:
        q = question(topic, "exam", seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for index, part in enumerate(q.parts[:-1]):
            nxt = q.parts[index + 1]
            if not nxt.follow_through or f"part_{part.label}" not in nxt.follow_through:
                continue
            others = [s for s in re.findall(r"part_[a-d]", nxt.follow_through) if s != f"part_{part.label}"]
            if others:
                continue
            for wrong in part.distractors:
                value = parse_typed(wrong)
                carried = sympy.sympify(nxt.follow_through, locals={"pi": sympy.pi}).subs(
                    sympy.Symbol(f"part_{part.label}"), value
                )
                carried = sympy.simplify(carried)
                if not carried.is_real or abs(float(carried) - float(parse_typed(nxt.answer_text))) < 1e-9:
                    continue
                if nxt.answer_kind == "fraction" and not carried.is_Rational:
                    continue
                carried_text = typed(carried)
                if (
                    carried_text in nxt.distractors
                    or carried_text == wrong
                    or abs(float(carried) - float(value)) < 1e-9
                ):
                    continue
                reply = {**canon, part.label: wrong, nxt.label: carried_text}
                result = grade(sk, q, reply)
                assert result.parts[part.label].marks == 0
                assert result.parts[nxt.label].marks == nxt.marks, (seed, part.label, wrong, carried, result)
                assert result.parts[nxt.label].follow_through is True
                checked += 1
    assert checked >= minimum, checked


def check_tags(sk, topic: str, tier: str, seeds) -> None:
    """Every option names an error the skill declares."""
    declared = {m.id for m in sk.misconceptions}
    for seed in seeds:
        q = question(topic, tier, seed)
        owners = q.parts if q.parts else [q]
        for owner in owners:
            for text in owner.distractors:
                assert owner.option_notes.get(text) in declared, (seed, text, owner.option_notes.get(text))
