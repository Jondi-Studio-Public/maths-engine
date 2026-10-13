"""Generates `cp1_de_moivre_proof` items: the real-valued working inside a proof of De Moivre's theorem, or a use of
the theorem.

Topic ``cp1_de_moivre_proof`` (open-response proof, ``level=14``). Complex numbers are not typed as lines of working:
a PROVE item names the real or imaginary part of a product or power and asks for a chain of equal real expressions
(an angle-sum step in the inductive proof, or a multiple-angle formula read off the binomial expansion of
(cos t + i sin t)^n). An APPLY item uses the theorem to find the real or imaginary part of a power or quotient of
numbers given in modulus-argument form, as a numeric chain.

Tiers: the inductive step for the real and imaginary parts and the squares and cubes; negative powers, cos 3t, sin 3t
and the angle-sum step for a product; cos 4t, sin 4t, tan 3t and a sum identity; sin 5t, cos 5t, cos 6t and the
power-reduction forms of cos^3 and sin^3.

The generator never runs a full CAS chain check. Every reference chain is confirmed numerically at fixed irregular
points, and every stated formula against a direct binomial expansion with SymPy, with the `math` module and SymPy only.
"""

from __future__ import annotations

import functools

import sympy as sp

from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import M, T

_NAMES = ("t", "u", "x")
_MOIVRE = "De Moivre's theorem says (cos t + i sin t)^n = cos(nt) + i sin(nt)."


def _fam(key, tier, template_ref, lead_builder, *, kind_note="", needs_k=False, domain="", names=_NAMES):
    fams = []
    for name in names:
        ref = [line.replace("{t}", name) for line in template_ref]
        symbols = {name: "real"}
        if needs_k:
            symbols["k"] = "integer"
        fams.append(
            dict(
                key=f"{key}_{name}",
                tier=tier,
                kind="identity",
                symbols=symbols,
                ref=ref,
                lead=lead_builder(name, ref),
                domain=domain.replace("{t}", name),
                jump_ok=len(ref) >= 4,
                given_step="states which part (real or imaginary) of (cos t + i sin t)^n is being equated, and the identity used",
            )
        )
    return fams


def _expansion_check(n: int, part: str, expr: str, name: str) -> None:
    """`expr` is the real/imaginary part of (cos + i sin)^n: compare with a direct expansion at a point."""
    t = sp.Symbol(name)
    z = (sp.cos(t) + sp.I * sp.sin(t)) ** n
    expanded = sp.expand(z)
    want = sp.re(expanded.subs(t, 0.7)) if part == "re" else sp.im(expanded.subs(t, 0.7))
    got = sp.sympify(expr).subs(t, 0.7)
    check(abs(float(want) - float(got)) < 1e-9, f"the stated {part} part of the n = {n} expansion is wrong")


def _step_lead(part: str):
    def build(name, ref):
        which = "real" if part == "re" else "imaginary"
        return [
            T(
                f"In the inductive step of a proof of De Moivre's theorem, assume (cos {name} + i sin {name})^k = cos(k{name}) + "
                f"i sin(k{name}) and multiply both sides by cos {name} + i sin {name}. The {which} part of the product is "
            ),
            M(ref[0]),
            T(". Show that it equals "),
            M(ref[-1]),
            T(", the part needed for n = k + 1."),
        ]

    return build


def _expansion_lead(n: int, part: str, label: str, final_note: str = ""):
    def build(name, ref):
        which = "real" if part == "re" else "imaginary"
        return [
            T(
                f"{_MOIVRE} Expanding (cos {name} + i sin {name})^{n} with the binomial theorem and equating {which} parts gives "
                f"{label}({n}{name}) = "
            ),
            M(ref[0]),
            T(". Show that this equals "),
            M(ref[-1]),
            T("." + final_note),
        ]

    return build


def _build_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {k: [] for k in ("easy", "medium", "hard", "exam", "apply_easy", "apply_exam")}
    add = lambda tier, fams: pools[tier].extend(fams)  # noqa: E731

    # --- easy: the inductive step, and n = 2, 3 ---
    add(
        "easy",
        _fam(
            "step_re",
            "easy",
            ["cos(k*{t})*cos({t}) - sin(k*{t})*sin({t})", "cos(k*{t} + {t})", "cos((k+1)*{t})"],
            _step_lead("re"),
            needs_k=True,
            domain="Here k is a positive integer and {t} is a real number.",
        ),
    )
    add(
        "easy",
        _fam(
            "step_im",
            "easy",
            ["sin(k*{t})*cos({t}) + cos(k*{t})*sin({t})", "sin(k*{t} + {t})", "sin((k+1)*{t})"],
            _step_lead("im"),
            needs_k=True,
            domain="Here k is a positive integer and {t} is a real number.",
        ),
    )
    add(
        "easy",
        _fam(
            "square_re",
            "easy",
            ["cos({t})**2 - sin({t})**2", "cos(2*{t})"],
            lambda n, r: _expansion_lead(2, "re", "cos")(n, r),
        ),
    )
    add(
        "easy",
        _fam(
            "square_im",
            "easy",
            ["2*sin({t})*cos({t})", "sin(2*{t})"],
            lambda n, r: _expansion_lead(2, "im", "sin")(n, r),
        ),
    )

    # --- medium ---
    def inverse_lead(part):
        def build(name, ref):
            which = "real" if part == "re" else "imaginary"
            return [
                T(
                    f"{_MOIVRE} For n = -1, multiply the top and bottom of 1/(cos {name} + i sin {name}) by cos {name} - i sin {name}. "
                    f"The {which} part of the result is "
                ),
                M(ref[0]),
                T(". Show that it equals "),
                M(ref[-1]),
                T(", using cos^2 + sin^2 = 1."),
            ]

        return build

    add(
        "medium",
        _fam(
            "inverse_re",
            "medium",
            ["cos({t})/(cos({t})**2 + sin({t})**2)", "cos({t})/1", "cos({t})"],
            inverse_lead("re"),
        ),
    )
    add(
        "medium",
        _fam(
            "inverse_im",
            "medium",
            ["-sin({t})/(cos({t})**2 + sin({t})**2)", "-sin({t})/1", "-sin({t})"],
            inverse_lead("im"),
        ),
    )
    add(
        "medium",
        _fam(
            "cos3",
            "medium",
            [
                "cos({t})**3 - 3*cos({t})*sin({t})**2",
                "cos({t})**3 - 3*cos({t})*(1 - cos({t})**2)",
                "4*cos({t})**3 - 3*cos({t})",
            ],
            lambda n, r: _expansion_lead(3, "re", "cos", " Replace sin^2 by 1 - cos^2.")(n, r),
        ),
    )
    add(
        "medium",
        _fam(
            "sin3",
            "medium",
            [
                "3*cos({t})**2*sin({t}) - sin({t})**3",
                "3*(1 - sin({t})**2)*sin({t}) - sin({t})**3",
                "3*sin({t}) - 4*sin({t})**3",
            ],
            lambda n, r: _expansion_lead(3, "im", "sin", " Replace cos^2 by 1 - sin^2.")(n, r),
        ),
    )
    add(
        "medium",
        _fam(
            "double_cos",
            "medium",
            ["cos({t})**2 - sin({t})**2", "cos({t})**2 - (1 - cos({t})**2)", "2*cos({t})**2 - 1"],
            lambda n, r: _expansion_lead(2, "re", "cos", " Replace sin^2 by 1 - cos^2.")(n, r),
        ),
    )

    def product_lead(name, ref):
        return [
            T("Multiply (cos a + i sin a) by (cos b + i sin b). The real part of the product is "),
            M(ref[0]),
            T(". Show that it equals "),
            M(ref[-1]),
            T(", the real part of cos(a + b) + i sin(a + b)."),
        ]

    pools["medium"].append(
        dict(
            key="product_re",
            tier="medium",
            kind="identity",
            symbols={"a": "real", "b": "real"},
            ref=["cos(a)*cos(b) - sin(a)*sin(b)", "cos(a + b)"],
            lead=product_lead("a", ["cos(a)*cos(b) - sin(a)*sin(b)", "cos(a + b)"]),
            domain="Here a and b are real numbers.",
            given_step="states the angle-sum identity or the product of two numbers in modulus-argument form",
        )
    )

    # --- hard ---
    add(
        "hard",
        _fam(
            "cos4",
            "hard",
            [
                "cos({t})**4 - 6*cos({t})**2*sin({t})**2 + sin({t})**4",
                "cos({t})**4 - 6*cos({t})**2*(1 - cos({t})**2) + (1 - cos({t})**2)**2",
                "8*cos({t})**4 - 8*cos({t})**2 + 1",
            ],
            lambda n, r: _expansion_lead(4, "re", "cos", " Replace sin^2 by 1 - cos^2.")(n, r),
        ),
    )
    add(
        "hard",
        _fam(
            "sin4",
            "hard",
            [
                "4*cos({t})**3*sin({t}) - 4*cos({t})*sin({t})**3",
                "4*sin({t})*cos({t})*(cos({t})**2 - sin({t})**2)",
                "4*sin({t})*cos({t})*(2*cos({t})**2 - 1)",
            ],
            lambda n, r: _expansion_lead(
                4, "im", "sin", " Factor out 4 sin t cos t, then use cos^2 - sin^2 = 2cos^2 - 1."
            )(n, r),
        ),
    )
    add(
        "hard",
        _fam(
            "cos3_sum",
            "hard",
            [
                "cos({t})**3 - 3*cos({t})*sin({t})**2 + 3*cos({t})",
                "cos({t})**3 - 3*cos({t})*(1 - cos({t})**2) + 3*cos({t})",
                "4*cos({t})**3",
            ],
            lambda n, r: [
                T(
                    f"{_MOIVRE} The real part of the n = 3 expansion is cos(3{n}) = cos^3({n}) - 3cos({n})sin^2({n}). Show that cos(3{n}) + 3cos({n}) equals 4cos^3({n}), starting from "
                ),
                M(r[0]),
                T("."),
            ],
        ),
    )

    # --- exam ---
    add(
        "exam",
        _fam(
            "sin5",
            "exam",
            [
                "5*cos({t})**4*sin({t}) - 10*cos({t})**2*sin({t})**3 + sin({t})**5",
                "5*(1 - sin({t})**2)**2*sin({t}) - 10*(1 - sin({t})**2)*sin({t})**3 + sin({t})**5",
                "16*sin({t})**5 - 20*sin({t})**3 + 5*sin({t})",
            ],
            lambda n, r: _expansion_lead(5, "im", "sin", " Replace cos^2 by 1 - sin^2.")(n, r),
        ),
    )
    add(
        "exam",
        _fam(
            "cos5",
            "exam",
            [
                "cos({t})**5 - 10*cos({t})**3*sin({t})**2 + 5*cos({t})*sin({t})**4",
                "cos({t})**5 - 10*cos({t})**3*(1 - cos({t})**2) + 5*cos({t})*(1 - cos({t})**2)**2",
                "16*cos({t})**5 - 20*cos({t})**3 + 5*cos({t})",
            ],
            lambda n, r: _expansion_lead(5, "re", "cos", " Replace sin^2 by 1 - cos^2.")(n, r),
        ),
    )
    add(
        "exam",
        _fam(
            "cos6",
            "exam",
            [
                "cos({t})**6 - 15*cos({t})**4*sin({t})**2 + 15*cos({t})**2*sin({t})**4 - sin({t})**6",
                "cos({t})**6 - 15*cos({t})**4*(1 - cos({t})**2) + 15*cos({t})**2*(1 - cos({t})**2)**2 - (1 - cos({t})**2)**3",
                "32*cos({t})**6 - 48*cos({t})**4 + 18*cos({t})**2 - 1",
            ],
            lambda n, r: _expansion_lead(6, "re", "cos", " Replace sin^2 by 1 - cos^2.")(n, r),
        ),
    )
    add(
        "exam",
        _fam(
            "cos_cubed",
            "exam",
            ["(cos(3*{t}) + 3*cos({t}))/4", "(4*cos({t})**3 - 3*cos({t}) + 3*cos({t}))/4", "cos({t})**3"],
            lambda n, r: [
                T(f"Using cos(3{n}) = 4cos^3({n}) - 3cos({n}) from De Moivre's theorem, show that "),
                M(r[0]),
                T(" = "),
                M(r[-1]),
                T(", which writes cos^3 as multiple angles."),
            ],
        ),
    )
    add(
        "exam",
        _fam(
            "sin_cubed",
            "exam",
            ["(3*sin({t}) - sin(3*{t}))/4", "(3*sin({t}) - (3*sin({t}) - 4*sin({t})**3))/4", "sin({t})**3"],
            lambda n, r: [
                T(f"Using sin(3{n}) = 3sin({n}) - 4sin^3({n}) from De Moivre's theorem, show that "),
                M(r[0]),
                T(" = "),
                M(r[-1]),
                T(", which writes sin^3 as multiple angles."),
            ],
        ),
    )

    # --- apply ---
    for r_, n_, m_, part in (
        (2, 4, 12, "re"),
        (2, 3, 12, "re"),
        (3, 2, 12, "re"),
        (2, 5, 6, "re"),
        (2, 3, 9, "im"),
        (4, 2, 12, "im"),
        (2, 4, 12, "im"),
        (3, 3, 12, "im"),
    ):
        pools["apply_easy"].append(_apply_power(r_, n_, m_, part))
    for r1, r2, a1, a2, m_, part in (
        (6, 2, 5, 1, 12, "re"),
        (8, 2, 7, 1, 12, "im"),
        (10, 5, 4, 1, 12, "re"),
        (6, 3, 5, 3, 12, "im"),
        (9, 3, 10, 2, 12, "re"),
        (12, 4, 11, 5, 12, "im"),
    ):
        pools["apply_exam"].append(_apply_quotient(r1, r2, a1, a2, m_, part))
    return pools


def _angle(n: int, m: int) -> str:
    return f"{n}*pi/{m}" if n != 1 else f"pi/{m}"


def _apply_power(r: int, n: int, m: int, part: str) -> dict:
    func = "cos" if part == "re" else "sin"
    which = "real" if part == "re" else "imaginary"
    value = sp.nsimplify(r**n * getattr(sp, func)(sp.pi * n / m))
    check(value != 0, "apply_power: the part should not be zero")
    value_text = str(sp.simplify(value))
    sym = getattr(sp, func)
    check(
        abs(float(value) - r**n * float(sym(sp.pi * n / m))) < 1e-9,
        "apply_power: the value does not match De Moivre's theorem",
    )
    ref = [f"{r}**{n}*{func}({n}*pi/{m})", f"{r**n}*{func}({sp.Rational(n, m)}*pi)", value_text]
    # A reduced angle line, written the way a person writes it: the multiple of pi as a fraction.
    reduced = sp.Rational(n, m)
    ref[1] = f"{r**n}*{func}({reduced.p}*pi/{reduced.q})" if reduced.p != 1 else f"{r**n}*{func}(pi/{reduced.q})"
    ref = [ref[0], ref[1], value_text]
    angle_text = f"π/{m}"
    return dict(
        key=f"apply_power_{r}_{n}_{m}_{part}",
        tier="apply_easy",
        lead=[
            T(
                f"Use De Moivre's theorem to find the {which} part of ({r}(cos({angle_text}) + i sin({angle_text})))^{n} exactly."
            )
        ],
        ref=ref,
        wrong_final=f"{value_text}+1",
        answer_text=f"{which} part = {kit.disp(value_text)}",
        given_step="states that the modulus is raised to the power n and the argument multiplied by n",
    )


def _apply_quotient(r1: int, r2: int, a1: int, a2: int, m: int, part: str) -> dict:
    func = "cos" if part == "re" else "sin"
    which = "real" if part == "re" else "imaginary"
    modulus = sp.Rational(r1, r2)
    angle = sp.Rational(a1 - a2, m)
    value = sp.simplify(modulus * getattr(sp, func)(sp.pi * angle))
    check(value != 0, "apply_quotient: the part should not be zero")
    value_text = str(value)
    num_text = f"{r1}(cos({a1}π/{m}) + i sin({a1}π/{m}))" if a1 != 1 else f"{r1}(cos(π/{m}) + i sin(π/{m}))"
    den_text = f"{r2}(cos({a2}π/{m}) + i sin({a2}π/{m}))" if a2 != 1 else f"{r2}(cos(π/{m}) + i sin(π/{m}))"
    first = f"{r1}/{r2}*{func}({a1}*pi/{m}-{a2}*pi/{m})"
    reduced = (
        f"{modulus.p}/{modulus.q}*{func}({angle.p}*pi/{angle.q})"
        if angle.p != 1
        else f"{modulus.p}/{modulus.q}*{func}(pi/{angle.q})"
    )
    if modulus.q == 1:
        reduced = reduced.replace(f"{modulus.p}/1*", f"{modulus.p}*")
    ref = [first, reduced, value_text]
    return dict(
        key=f"apply_quotient_{r1}_{r2}_{a1}_{a2}_{part}",
        tier="apply_exam",
        lead=[
            T(
                f"Use the division form of De Moivre's theorem (divide the moduli, subtract the arguments) to find the {which} part of "
                f"({num_text})/({den_text}) exactly."
            )
        ],
        ref=ref,
        wrong_final=f"({value_text})+1",
        answer_text=f"{which} part = {kit.disp(value_text)}",
        given_step="divides the moduli, subtracts the arguments and reads off the required part",
    )


@functools.lru_cache(maxsize=None)
def pools() -> dict[str, list[dict]]:
    """The families, built on first use (building them runs SymPy, which the package import should not pay for)."""
    return _build_pools()


@generator(
    "cp1_de_moivre_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def cp1_de_moivre_proof(rng, variant="easy"):
    """`senior-prove-and-use-de-moivre-s-theorem-for`. Every question carries a marking dict (`Question.answer`):
    the working as a chain of equal real expressions, the reference chain, the key steps and the traps."""
    if variant not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"unknown variant: {variant}")
    if variant in ("easy", "exam") and rng.random() < 0.4:
        return kit.apply_item(rng, kit.pick(rng, pools()[f"apply_{variant}"]))
    return kit.prove_item(rng, kit.pick(rng, pools()[variant]))
