"""Levels 19-20: engineering maths and structural mechanics word templates."""

from fractions import Fraction

from sympy import Rational

from ..core import generator, pick_distractors, round_half_up, stated_precision_string


def _frac(value):
    return Fraction(str(round(float(value), 9)))


def _places_for(value: Fraction, cap: int) -> int:
    """Fewest decimal places (up to ``cap``) that show ``value`` exactly, or
    ``cap`` when it does not terminate that finely."""
    for places in range(cap + 1):
        if (value * 10**places).denominator == 1:
            return places
    return cap


def _aligned_units(answer, candidates, fmt):
    """Show the answer and every distractor at ONE shared number of decimal
    places, so the options cannot be told apart by shape ("60 N" against
    "6.66667 N"). The answer shows its own exact decimals (two when it does
    not terminate); distractors are rounded to the shared grid, and a
    distractor that would read as the answer, or as a rounded-away zero, is
    dropped. ``fmt`` wraps the numeral in the label and unit. Returns
    ``(answer_numeral, answer_text, distractor_texts)``."""
    value = _frac(answer)
    places = _places_for(value, 6)
    if (value * 10**places).denominator != 1:
        places = 2
    values = [_frac(c) for c in candidates]
    # A candidate under half a hundredth would show as 0.00 (the "E = 6.7e-06"
    # inverted-ratio slip): not displayable at this precision, so not offered.
    values = [v for v in values if v == 0 or abs(v) >= Fraction(1, 200)]
    places = max([places] + [_places_for(v, 2) for v in values])

    def show(v):
        return stated_precision_string(round_half_up(v, places), places)

    numeral = show(value)
    shown = [show(v) for v in values if show(v) != numeral and (v == 0 or round_half_up(v, places) != 0)]
    return numeral, fmt(numeral), [fmt(t) for t in shown]


@generator("engineering", level=19)
def engineering(rng, kinds=("stress", "strain", "moment", "equilibrium")):
    kind = rng.choice(list(kinds))
    if kind == "hookes_law":
        # Backwards: pick a stiffness and extension, derive the force.
        k = rng.choice([50, 100, 150, 200, 250, 300, 400, 500])  # N/m
        ext = rng.choice([Rational(1, 20), Rational(1, 10), Rational(1, 5), Rational(3, 10), Rational(1, 2)])  # m
        F = k * ext
        num, ans_text, dist = _aligned_units(F, [Rational(k) / ext, k, ext], lambda n: f"F = {n} N")
        return dict(
            text=(
                f"A spring with stiffness k = {k} N/m is stretched by {float(ext):g} m. "
                f"Use Hooke's law to find the force required."
            ),
            latex=rf"F = kx = {k} \times {float(ext):g}",
            answer=float(F),
            answer_text=ans_text,
            answer_latex=rf"F = {num}\,\text{{N}}",
            working=[f"F = kx = {k} × {float(ext):g} = {float(F):g} N"],
            # Misconceptions: divided instead of multiplied; quoted k alone;
            # quoted the extension alone.
            distractors=pick_distractors(ans_text, dist),
        )
    if kind == "youngs_modulus":
        # Backwards: pick E (GPa) and a strain, derive the stress.
        E_GPa = rng.choice([70, 100, 120, 150, 200, 210])
        eps_micro = rng.choice([200, 250, 400, 500, 800, 1000])
        eps = Rational(eps_micro, 10**6)
        sigma = E_GPa * 1000 * eps  # MPa
        num, ans_text, dist = _aligned_units(E_GPa, [E_GPa * 1000, eps / sigma, sigma], lambda n: f"E = {n} GPa")
        return dict(
            text=(
                f"A material has strain {float(eps):g} under a stress of {float(sigma):g} MPa. "
                f"Calculate its Young's modulus in GPa."
            ),
            latex=rf"E = \sigma/\varepsilon = {float(sigma):g} / {float(eps):g}",
            answer=float(E_GPa),
            answer_text=ans_text,
            answer_latex=rf"E = {num}\,\text{{GPa}}",
            working=[f"E = σ/ε = {float(sigma):g} MPa / {float(eps):g} = {E_GPa * 1000:g} MPa = {E_GPa:g} GPa"],
            # Misconceptions: forgot the MPa->GPa conversion; inverted the
            # ratio; quoted the stress as if it were the modulus.
            distractors=pick_distractors(ans_text, dist),
        )
    if kind == "stress":
        # Backwards: pick nice stress (MPa) and area (mm^2), derive force.
        sigma = rng.choice([20, 25, 40, 50, 80, 100, 125, 160, 200])
        A = rng.choice([50, 100, 200, 250, 400, 500])
        F = sigma * A  # N
        num, ans_text, dist = _aligned_units(sigma, [sigma / 1000, sigma * 1000, F * A], lambda n: f"\u03c3 = {n} MPa")
        return dict(
            text=(
                f"A bar of cross-sectional area {A} mm\u00b2 carries an axial force of "
                f"{F / 1000:g} kN. Calculate the stress in MPa."
            ),
            latex=rf"\sigma = \frac{{F}}{{A}} = \frac{{{F / 1000:g}\,\text{{kN}}}}{{{A}\,\text{{mm}}^2}}",
            answer=sigma,
            answer_text=ans_text,
            answer_latex=rf"\sigma = {num}\,\text{{MPa}}",
            working=[f"\u03c3 = F/A = {F} N / {A} mm\u00b2 = {sigma} N/mm\u00b2 = {sigma} MPa"],
            # Misconceptions: kN/mm\u00b2 unit slip (\u00f71000); N with m\u00b2 (\u00d71000 the
            # other way); multiplied F by A.
            distractors=pick_distractors(ans_text, dist),
        )
    if kind == "strain":
        # Backwards: pick strain and original length, derive extension.
        L = rng.choice([200, 250, 400, 500, 800, 1000])  # mm
        eps_micro = rng.choice([250, 400, 500, 800, 1000, 1250, 2000])  # microstrain
        dL = Rational(eps_micro, 10**6) * L
        eps = Rational(eps_micro, 10**6)
        num, ans_text, dist = _aligned_units(
            eps, [L / float(dL), float(dL), float(eps) * 100], lambda n: f"\u03b5 = {n}"
        )
        return dict(
            text=(f"A bar of original length {L} mm extends by {float(dL):g} mm under load. Calculate the strain."),
            latex=rf"\varepsilon = \frac{{\delta L}}{{L}} = \frac{{{float(dL):g}}}{{{L}}}",
            answer=float(eps),
            answer_text=ans_text,
            answer_latex=rf"\varepsilon = {num}",
            working=[f"\u03b5 = \u03b4L/L = {float(dL):g}/{L} = {float(eps):g} ({eps_micro} \u03bc\u03b5)"],
            # Misconceptions: inverted the ratio; forgot to divide; quoted the
            # percentage value as the strain.
            distractors=pick_distractors(ans_text, dist),
        )
    if kind == "moment":
        F = rng.choice([5, 10, 15, 20, 25, 40, 50])  # N
        d = rng.choice([Rational(1, 2), 1, Rational(3, 2), 2, 3, 4])  # m
        M = F * d
        num, ans_text, dist = _aligned_units(M, [F + d, Rational(F) / d, F], lambda n: f"M = {n} N\u00b7m")
        return dict(
            text=(
                f"A force of {F} N acts perpendicular to a lever at {float(d):g} m from the pivot. "
                f"Calculate the moment about the pivot in N\u00b7m."
            ),
            latex=rf"M = F d = {F} \times {float(d):g}",
            answer=float(M),
            answer_text=ans_text,
            answer_latex=rf"M = {num}\,\text{{N}}\cdot\text{{m}}",
            working=[f"Moment = force × perpendicular distance = {F} × {float(d):g} = {float(M):g} N·m"],
            # Misconceptions: added force and distance; divided; ignored the
            # distance entirely.
            distractors=pick_distractors(ans_text, dist),
        )
    # Equilibrium: see-saw balance, backwards from equal moments.
    M_bal = rng.choice([60, 120, 180, 240, 360])
    F1 = rng.choice([d for d in [20, 30, 40, 60] if M_bal % d == 0])
    d1 = M_bal // F1
    d2 = rng.choice([d for d in [1, 2, 3, 4] if M_bal % d == 0])
    F2 = M_bal // d2
    num, ans_text, dist = _aligned_units(
        # The last three are fallbacks for the degenerate draw (equal weights at
        # equal distances) where every slip above equals the answer.
        F2,
        [F1, M_bal, Rational(F1 * d2, d1), 2 * F2, F2 + F1, F2 + d1],
        lambda n: f"F = {n} N",
    )
    return dict(
        text=(
            f"A beam balances on a pivot. A {F1} N weight sits {d1} m to the left. "
            f"What force placed {d2} m to the right keeps the beam in equilibrium?"
        ),
        latex=rf"{F1} \times {d1} = F \times {d2}",
        answer=F2,
        answer_text=ans_text,
        answer_latex=rf"F = {num}\,\text{{N}}",
        working=[f"Anticlockwise moment = {F1} \u00d7 {d1} = {M_bal} N\u00b7m", f"F = {M_bal} / {d2} = {F2} N"],
        # Misconceptions: assumed the same weight balances; answered with the
        # moment instead of the force; flipped the distance ratio.
        distractors=pick_distractors(ans_text, dist),
    )


@generator("structural", level=20)
def structural(rng, kinds=("reactions", "max_bm_udl", "max_shear_udl")):
    kind = rng.choice(list(kinds))
    if kind == "bending_moment_point_load":
        # Simply supported beam, point load W at distance a from A, span L:
        # bending moment under the load, M = R_A * a.
        L = rng.choice([4, 5, 6, 8, 10, 12])
        a = rng.randint(1, L - 1)
        W = rng.choice([10, 20, 30, 40, 50, 60])
        b = L - a
        Ra = Rational(W * b, L)
        M = Ra * a
        num, ans_text, dist = _aligned_units(M, [W * a, Ra * b, Ra], lambda n: f"M = {n} kN·m")
        return dict(
            text=(
                f"A simply supported beam of span {L} m carries a point load of {W} kN "
                f"at {a} m from support A. Find the bending moment under the load."
            ),
            latex=rf"M = R_A \times a = \frac{{{W} \times {b}}}{{{L}}} \times {a}",
            answer=float(M),
            answer_text=ans_text,
            answer_latex=rf"M = {num}\,\text{{kN}}\cdot\text{{m}}",
            working=[
                f"R_A = Wb/L = {W}×{b}/{L} = {float(Ra):g} kN",
                f"M = R_A × a = {float(Ra):g} × {a} = {float(M):g} kN·m",
            ],
            # Misconceptions: used the full load instead of the reaction;
            # used R_A with the wrong (far-side) distance; quoted R_A itself.
            distractors=pick_distractors(ans_text, dist),
        )
    if kind == "cantilever_udl":
        # Cantilever, fixed at one end, UDL over the full length.
        L = rng.choice([2, 3, 4, 5, 6])
        w = rng.choice([2, 4, 5, 6, 8, 10])
        R = w * L
        M = Rational(w * L**2, 2)
        return dict(
            text=(
                f"A cantilever beam of length {L} m, fixed at one end, carries a uniformly "
                f"distributed load of {w} kN/m over its full length. Find the fixed-end "
                f"reaction R and fixing moment M."
            ),
            latex=rf"R = wL,\ M = \frac{{wL^2}}{{2}}",
            answer={"R": float(R), "M": float(M)},
            answer_text=f"R = {float(R):g} kN, M = {float(M):g} kN·m",
            answer_latex=rf"R = {float(R):g}\,\text{{kN}},\ M = {float(M):g}\,\text{{kN}}\cdot\text{{m}}",
            working=[f"R = wL = {w} × {L} = {float(R):g} kN", f"M = wL²/2 = {w} × {L}² / 2 = {float(M):g} kN·m"],
            # Misconceptions: halved the reaction as if simply supported;
            # used the simply-supported UDL moment formula wL²/8; forgot to
            # square L in the moment.
            distractors=pick_distractors(
                f"R = {float(R):g} kN, M = {float(M):g} kN·m",
                [
                    f"R = {float(R) / 2:g} kN, M = {float(M):g} kN·m",
                    f"R = {float(R):g} kN, M = {float(Rational(w * L**2, 8)):g} kN·m",
                    f"R = {float(R):g} kN, M = {float(Rational(w * L, 2)):g} kN·m",
                ],
            ),
        )
    if kind == "reactions":
        # Simply supported beam, point load W at distance a from A, span L.
        L = rng.choice([4, 5, 6, 8, 10, 12])
        a = rng.randint(1, L - 1)
        W = rng.choice([10, 20, 30, 40, 50, 60])
        b = L - a
        Ra, Rb = Rational(W * b, L), Rational(W * a, L)
        return dict(
            text=(
                f"A simply supported beam of span {L} m carries a point load of {W} kN "
                f"at {a} m from support A. Find the reactions R_A and R_B."
            ),
            latex=rf"L = {L}\,\text{{m}},\ W = {W}\,\text{{kN at }} a = {a}\,\text{{m from A}}",
            answer={"R_A": float(Ra), "R_B": float(Rb)},
            answer_text=f"R_A = {float(Ra):g} kN, R_B = {float(Rb):g} kN",
            answer_latex=rf"R_A = {float(Ra):g}\,\text{{kN}},\ R_B = {float(Rb):g}\,\text{{kN}}",
            working=[
                f"Moments about A: R_B \u00d7 {L} = {W} \u00d7 {a}  \u2192  R_B = {float(Rb):g} kN",
                f"R_A = {W} - {float(Rb):g} = {float(Ra):g} kN",
            ],
            # Misconceptions: swapped the reactions; assumed W/2 each; forgot
            # to divide the moment by the span; gave each support the full load.
            distractors=pick_distractors(
                f"R_A = {float(Ra):g} kN, R_B = {float(Rb):g} kN",
                [
                    f"R_A = {float(Rb):g} kN, R_B = {float(Ra):g} kN",
                    f"R_A = {W / 2:g} kN, R_B = {W / 2:g} kN",
                    f"R_A = {W * b:g} kN, R_B = {W * a:g} kN",
                    f"R_A = {W:g} kN, R_B = {W:g} kN",
                ],
            ),
        )
    if kind == "max_bm_udl":
        # Backwards: choose w and L so wL^2/8 is clean.
        L = rng.choice([4, 8, 12])
        w = rng.choice([2, 4, 5, 6, 8, 10])
        M = Rational(w * L**2, 8)
        num, ans_text, dist = _aligned_units(
            M, [w * L**2 / 12, w * L**2 / 2, w * L / 8], lambda n: f"M_max = {n} kN\u00b7m"
        )
        return dict(
            text=(
                f"A simply supported beam of span {L} m carries a uniformly distributed load "
                f"of {w} kN/m over its full length. Find the maximum bending moment."
            ),
            latex=rf"M_{{max}} = \frac{{wL^2}}{{8}} = \frac{{{w} \times {L}^2}}{{8}}",
            answer=float(M),
            answer_text=ans_text,
            answer_latex=rf"M_{{max}} = {num}\,\text{{kN}}\cdot\text{{m}}",
            working=[f"M_max = wL\u00b2/8 = {w} \u00d7 {L}\u00b2 / 8 = {float(M):g} kN\u00b7m (at midspan)"],
            # Misconceptions: used the fixed-end formula wL\u00b2/12; the cantilever
            # formula wL\u00b2/2; forgot to square the span.
            distractors=pick_distractors(ans_text, dist),
        )
    # Max shear for UDL on simply supported span: wL/2 at the supports.
    L = rng.choice([4, 6, 8, 10, 12])
    w = rng.choice([2, 3, 4, 5, 6, 8, 10])
    V = Rational(w * L, 2)
    num, ans_text, dist = _aligned_units(V, [w * L, w * L**2 / 8, w * L / 4], lambda n: f"V_max = {n} kN")
    return dict(
        text=(
            f"A simply supported beam of span {L} m carries a uniformly distributed load of "
            f"{w} kN/m. Find the maximum shear force."
        ),
        latex=rf"V_{{max}} = \frac{{wL}}{{2}} = \frac{{{w} \times {L}}}{{2}}",
        answer=float(V),
        answer_text=ans_text,
        answer_latex=rf"V_{{max}} = {num}\,\text{{kN}}",
        working=[f"V_max = wL/2 = {w} \u00d7 {L} / 2 = {float(V):g} kN (at the supports)"],
        # Misconceptions: total load wL; confused with the moment wL\u00b2/8;
        # halved twice.
        distractors=pick_distractors(ans_text, dist),
    )
