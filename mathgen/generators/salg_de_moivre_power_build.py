"""Generates `salg_de_moivre_power` questions (easy, medium, hard) and the exam chain `salg_exam_de_moivre_power`.

Skill ``senior-use-de-moivre-s-theorem-for-integral-powers`` (``senior-mathematics``, answer ``pair``): a power of a
complex number by De Moivre's theorem, ``(r cis θ)^n = r^n cis(nθ)``, with whole-number moduli and arguments in
degrees. Products and quotients are the sibling skill
``senior-define-and-use-multiplication-and-division-of-complex-numbers-in-polar-form``; this skill is about the power.

* easy: ``z = r cis θ`` squared or cubed, with r of 1 to 3 and nθ below 360°.
* medium: powers 3 to 7 of a number of modulus 1 or 2 whose argument grows past 360°, so it has to be reduced; and
  negative powers of a number of modulus 1, where the argument goes negative before it is reduced.
* hard: z given in Cartesian form (1 ± i, -1 ± i, 1 ± √3i, √3 ± i and their negatives), so the modulus and the argument
  (in the right quadrant) have to be found first; the power is chosen so that the modulus stays a whole number.
* exam (``salg_exam_de_moivre_power``): z = ±1 ± √3 i or ±√3 ± i; (a) |z|; (b) Arg z in degrees from 0° to 360°;
  (c) |z^n|, using (a); (d) z^n as p + qi, using (a) and (b).

Each power is confirmed by multiplying the Cartesian number out with Python's ``cmath`` (and, for the exam chain, with
SymPy's exact ``expand``).
"""

from __future__ import annotations

import cmath
import math

import sympy as sp

from ..answers import Distractor
from ..core import check, draw_until, generator, latex_answer, phrase, pick_distractors
from ._salg_kit import int_dist, polar_dist, polar_pair

_PAIR_ASK = "Give the result as r cis θ by writing the pair (r, θ), with θ in degrees and 0 ≤ θ < 360."


def _norm(angle: int) -> int:
    return angle % 360


def _confirm(z: complex, n: int, r: int, theta: int) -> None:
    value = z**n
    check(abs(abs(value) - r) < 1e-6 * max(1, r), f"modulus {abs(value)} differs from {r}")
    arg = math.degrees(cmath.phase(value)) % 360
    check(min(abs(arg - theta), 360 - abs(arg - theta)) < 1e-6, f"argument {arg} differs from {theta}")


def _pair_question(rng, stems, slots, r, theta, cands, working):
    answer = polar_pair(r, theta)
    dist = pick_distractors(answer.answer_text(), [polar_dist(a, b, t) for a, b, t in cands if (a, b) != (r, theta)], 3)
    check(len(dist) >= 2, f"too few power distractors for {slots}: {cands}")
    return dict(
        text=phrase(rng, stems, **slots) + " " + _PAIR_ASK,
        answer=answer.answer_dict(),
        answer_text=answer.answer_text(),
        answer_latex=rf"r = {r}, \theta = {theta}°",
        answer_kind="pair",
        working=working,
        distractors=dist,
    )


def _polar_text(r: int, theta: int) -> str:
    """`z = 2 cis 40°`, or `z = cis 40°` for a modulus of 1."""
    return f"z = {'' if r == 1 else f'{r} '}cis {theta}°"


def _sup(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))


_POWER_STEMS = ("Let {z}. Find z{n}.", "Given {z}, find z{n}.", "For {z}, find z{n}.")


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    r0, n = rng.choice([2, 2, 3]), rng.choice([2, 3])
    t0 = draw_until(
        rng, lambda r: r.randrange(10, 180, 10), lambda t: n * t < 360, what="an argument with nθ below 360"
    )
    r, theta = r0**n, n * t0
    _confirm(cmath.rect(r0, math.radians(t0)), n, r, theta)
    cands = [
        (r0, theta, "demoivre-modulus-not-raised"),
        (r0 * n, theta, "demoivre-modulus-multiplied"),
        (r, t0, "demoivre-argument-not-multiplied"),
        (r0 * n, t0, "demoivre-modulus-multiplied"),
        (r0 * n, t0 * n, "demoivre-modulus-multiplied"),
    ]
    return _pair_question(
        rng,
        _POWER_STEMS,
        dict(z=_polar_text(r0, t0), n=_sup(n)),
        r,
        theta,
        cands,
        [
            f"Raise the modulus to the power {n}: {r0}^{n} = {r}.",
            f"Multiply the argument by {n}: {n} × {t0}° = {theta}°.",
        ],
    )


# -- medium ---------------------------------------------------------------------------------------------------------


def _medium_positive(rng):
    r0, n = rng.choice([1, 2]), rng.randint(3, 7)
    t0 = draw_until(
        rng, lambda r: r.randrange(20, 180, 10), lambda t: n * t >= 360, what="an argument with nθ of 360 or more"
    )
    r, raw = r0**n, n * t0
    theta = _norm(raw)
    _confirm(cmath.rect(r0, math.radians(t0)), n, r, theta)
    cands = [
        (r, raw, "polar-angle-not-reduced"),
        (r0 * n, theta, "demoivre-modulus-multiplied"),
        (r, t0, "demoivre-argument-not-multiplied"),
        (r, _norm(raw - 180), "polar-angle-reduced-by-half-turn"),
        (r0, theta, "demoivre-modulus-not-raised"),
    ]
    return _pair_question(
        rng,
        _POWER_STEMS,
        dict(z=_polar_text(r0, t0), n=_sup(n)),
        r,
        theta,
        cands,
        [
            f"Raise the modulus to the power {n}: {r0}^{n} = {r}.",
            f"Multiply the argument by {n}: {n} × {t0}° = {raw}°, which is {(raw - theta) // 360} full turn{'s' if raw - theta > 360 else ''} more than {theta}°.",
            f"So z{_sup(n)} = {'' if r == 1 else f'{r} '}cis {theta}°.",
        ],
    )


def _medium_negative(rng):
    n = rng.choice([-1, -2, -3, -4])
    t0 = rng.randrange(20, 180, 10)
    raw = n * t0
    theta = _norm(raw)
    _confirm(cmath.rect(1, math.radians(t0)), n, 1, theta)
    cands = [
        (1, -raw, "demoivre-negative-power-sign-lost"),
        (1, raw, "polar-angle-not-reduced"),
        (abs(n), theta, "demoivre-modulus-multiplied"),
        (1, t0, "demoivre-argument-not-multiplied"),
    ]
    return _pair_question(
        rng,
        _POWER_STEMS,
        dict(z=_polar_text(1, t0), n=_sup(n)),
        1,
        theta,
        cands,
        [
            f"The modulus is 1, and 1 to any power is 1.",
            f"Multiply the argument by {n}: {n} × {t0}° = {raw}°, and adding full turns to bring it into range gives {theta}°.",
        ],
    )


def _medium(rng):
    return rng.choice([_medium_positive, _medium_positive, _medium_negative])(rng)


# -- hard -----------------------------------------------------------------------------------------------------------

#: (real part, imaginary part, text, r as sqrt-text, modulus kind). Moduli are 2 (kind "two") or sqrt 2 (kind "root2").
_CARTESIAN = []
for _a, _b, _kind in ((1, 1, "root2"), (1, -1, "root2"), (-1, 1, "root2"), (-1, -1, "root2")):
    _CARTESIAN.append((_a, _b, _kind))
for _a, _b in (
    (1, sp.sqrt(3)),
    (-1, sp.sqrt(3)),
    (1, -sp.sqrt(3)),
    (-1, -sp.sqrt(3)),
    (sp.sqrt(3), 1),
    (-sp.sqrt(3), 1),
    (sp.sqrt(3), -1),
    (-sp.sqrt(3), -1),
):
    _CARTESIAN.append((_a, _b, "two"))


def _part_text(value) -> str:
    """The size of one part of a Cartesian number: 1 or √3, with its sign."""
    return ("-" if value < 0 else "") + ("1" if value in (1, -1) else "√3")


def cartesian_text(a, b) -> str:
    """`a + bi` with the signs written the way a student reads them: `-1 + i√3`, `√3 - i`."""
    imaginary = "i" if abs(b) == 1 else "i√3"
    return f"{_part_text(a)} {'+' if b > 0 else '-'} {imaginary}"


def _hard(rng):
    a, b, kind = rng.choice(_CARTESIAN)
    z = complex(float(a), float(b))
    theta0 = round(math.degrees(cmath.phase(z))) % 360
    n = draw_until(
        rng,
        lambda r: r.randint(3, 11),
        lambda k: (kind == "two" or k % 2 == 0) and (k * theta0) % 360 != theta0,
        what="a power that keeps the modulus whole",
    )
    mod = 2 if kind == "two" else None
    r = 2**n if kind == "two" else 2 ** (n // 2)
    raw = n * theta0
    theta = _norm(raw)
    _confirm(z, n, r, theta)
    ref = {1: theta0, 2: 180 - theta0, 3: theta0 - 180, 4: 360 - theta0}[theta0 // 90 + 1 if theta0 % 90 else 1]
    cands = [
        (r, raw, "polar-angle-not-reduced"),
        (r, theta0, "demoivre-argument-not-multiplied"),
        (r, _norm(n * ref), "demoivre-quadrant-ignored"),
    ]
    if kind == "two":
        cands += [(2, theta, "demoivre-modulus-not-raised"), (2 * n, theta, "demoivre-modulus-multiplied")]
    else:
        cands += [(2**n, theta, "demoivre-modulus-root-not-taken"), (n, theta, "demoivre-modulus-multiplied")]
    return _pair_question(
        rng,
        _POWER_STEMS,
        dict(z=f"z = {cartesian_text(a, b)}", n=_sup(n)),
        r,
        theta,
        cands,
        [
            f"|z| = {'2' if kind == 'two' else '√2'} and Arg z = {theta0}°, so z = {'2' if kind == 'two' else '√2'} cis {theta0}°.",
            f"z{_sup(n)} = {'2' if kind == 'two' else '(√2)'}^{n} cis ({n} × {theta0}°) = {r} cis {raw}°, so theta = {theta}°.",
        ],
    )


@generator(
    "salg_de_moivre_power",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def salg_de_moivre_power(rng, tier="easy"):
    """A power of a complex number by De Moivre's theorem; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _typed_complex(value: sp.Expr) -> str:
    """`p + qi` as a student types it, from an exact SymPy complex number: `-32 + 32sqrt(3)i`."""
    re_part, im_part = sp.re(value), sp.im(value)
    text = ""
    if re_part != 0:
        text = str(re_part).replace("*sqrt", "sqrt")
    if im_part != 0:
        mag = sp.Abs(im_part)
        mag_text = "" if mag == 1 else str(mag).replace("*sqrt", "sqrt")
        imag = f"{mag_text}i"
        if text:
            text += f" {'+' if im_part > 0 else '-'} {imag}"
        else:
            text = f"{'-' if im_part < 0 else ''}{imag}"
    return text or "0"


@generator("salg_exam_de_moivre_power", level=12, presets={"exam": dict(tier="exam")})
def salg_exam_de_moivre_power(rng, tier="exam"):
    """Exam chain: |z|, Arg z, then |z^n| and z^n in Cartesian form by following through."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a, b, _kind = rng.choice([c for c in _CARTESIAN if c[2] == "two"])
    n = rng.randint(3, 11)
    z = a + sp.I * b
    theta0 = int(round(math.degrees(cmath.phase(complex(float(a), float(b)))))) % 360
    power = sp.expand(z**n)
    check(
        abs(complex(power) - complex(2**n * cmath.exp(1j * math.radians(n * theta0)))) < 1e-6,
        "power differs from cmath",
    )
    power_text = _typed_complex(power)
    check(power_text.replace("i", "I") != "", "empty power text")
    ref = {1: theta0, 2: 180 - theta0, 3: theta0 - 180, 4: 360 - theta0}[theta0 // 90 + 1]
    two_n = 2**n

    def number_part(prompt, key, marks, cands, working, ft=None, bounds=None):
        part = dict(
            prompt=prompt,
            answer_kind="number",
            answer=key,
            answer_text=str(key),
            answer_latex=str(key),
            marks=marks,
            sensible_range=bounds,
            working=working,
            distractors=pick_distractors(str(key), [int_dist(v, t) for v, t in cands if v != key and v > 0], 3),
        )
        check(len(part["distractors"]) >= 2, f"too few distractors for {prompt!r}")
        if ft:
            part["follow_through"] = ft
        return part

    # Distractors for the Cartesian part: the slips of De Moivre's theorem applied to 2 cis theta0.
    def cart(modulus, angle, tag, conj=False, swap=False):
        value = modulus * (sp.cos(sp.pi * angle / 180) + sp.I * sp.sin(sp.pi * angle / 180))
        if conj:
            value = sp.conjugate(value)
        if swap:
            value = modulus * (sp.sin(sp.pi * angle / 180) + sp.I * sp.cos(sp.pi * angle / 180))
        value = sp.expand(value)
        text = _typed_complex(value)
        # Both sides are exact algebraic numbers, so a float comparison far above rounding error is decisive.
        return Distractor(text, tag, text), abs(complex(value) - complex(power)) > 1e-9

    d_cands = []
    for args in (
        (2, n * theta0, "demoivre-modulus-not-raised"),
        (two_n, theta0, "demoivre-argument-not-multiplied"),
        (two_n * n // n, n * theta0, "demoivre-conjugate-given", True),
        (two_n, n * theta0, "demoivre-sine-cosine-swapped", False, True),
        (2 * n, n * theta0, "demoivre-modulus-multiplied"),
    ):
        cand, different = cart(*args)
        if different:
            d_cands.append(cand)
    d_dist = pick_distractors(power_text, d_cands, 3)
    check(len(d_dist) >= 2, "too few Cartesian power distractors")

    return dict(
        text=f"Let z = {cartesian_text(a, b)}.",
        parts=[
            number_part(
                "Find the modulus of z.",
                2,
                1,
                [(4, "demoivre-modulus-root-not-taken"), (1, "demoivre-modulus-is-one-component")],
                [f"|z| = sqrt(({_part_text(a)})² + ({_part_text(b)})²) = sqrt(4) = 2."],
                bounds=(0, None),
            ),
            number_part(
                "Find the argument of z, in degrees, with 0 ≤ arg < 360.",
                theta0,
                2,
                [
                    (ref, "demoivre-quadrant-ignored"),
                    (90 - ref, "demoivre-argument-from-wrong-ratio"),
                    (_norm(theta0 + 180), "demoivre-quadrant-ignored"),
                    (360 - theta0, "demoivre-quadrant-ignored"),
                ],
                [
                    f"The point ({_part_text(a)}, {_part_text(b)}) is in quadrant {theta0 // 90 + 1}, with reference angle {ref}°, so Arg z = {theta0}°."
                ],
                bounds=(0, 360),
            ),
            number_part(
                f"Hence find |z{_sup(n)}|.",
                two_n,
                1,
                [
                    (2 * n, "demoivre-modulus-multiplied"),
                    (n * n, "demoivre-modulus-raised-the-wrong-way"),
                    (4**n, "demoivre-modulus-root-not-taken"),
                    (2 + n, "demoivre-modulus-multiplied"),
                ],
                [f"|z{_sup(n)}| = |z|^{n} = 2^{n} = {two_n}."],
                ft=f"part_a**{n}",
                bounds=(0, None),
            ),
            dict(
                prompt=f"Hence write z{_sup(n)} in the form p + qi.",
                answer_kind="expression",
                answer=power_text,
                answer_text=power_text,
                answer_latex=latex_answer(power),
                marks=3,
                follow_through=f"part_a**{n}*(cos({n}*part_b*pi/180) + I*sin({n}*part_b*pi/180))",
                working=[
                    f"z{_sup(n)} = 2^{n} cis ({n} × {theta0}°) = {two_n} (cos {n * theta0}° + i sin {n * theta0}°).",
                    f"So z{_sup(n)} = {power_text}.",
                ],
                distractors=d_dist,
            ),
        ],
    )
