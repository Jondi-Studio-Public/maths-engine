"""Generates `model_limit_validation` items: state the limiting behaviour of a supplied model and say whether it is
consistent with a stated physical assumption.

Each question gives a closed-form model with numbers and units (Newton cooling, an RC circuit, radioactive decay,
logistic growth, terminal speed, repeated doses, a rational yield) and one physical assumption (a ceiling, a
resting value, a measured speed). The generator computes the limit exactly (fractions, or the stated closed form)
and decides consistency by comparing it with the number in the assumption, so each question knows its limit with
units and its verdict; roughly half are consistent. The answer is marked against a rubric, so `answer` is a
marking dict (see `_crit_kit`), not a value.

Ladder:

- easy: the limit as t grows is read off an exponential approach (cooling, charging, decay).
- medium: the limit needs an expression to be simplified (logistic, terminal speed, repeated doses, a ratio of
  linear expressions) and is compared with a ceiling or a measured value.
- hard: a limit as a parameter shrinks (drag towards zero) that has to be found as a 0/0 limit, compared with the
  drag-free result.
- exam: both limits of one model are checked against two assumptions, and a damped oscillation is compared with a
  claim that friction brings it to rest.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator
from ._crit_kit import attempts, Fm, Prompt, SHORT_TAILS, finish, frac_plain, short_marking

_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}


def _n(x) -> str:
    """A number as plain text: whole numbers bare, otherwise a short decimal."""
    f = Fraction(x).limit_denominator(10**6)
    if f.denominator == 1:
        return str(f.numerator)
    s = f"{float(f):.6f}".rstrip("0")
    check(Fraction(s) == f, "a terminating decimal")
    return s


def _model(context, fm, params, limit_phrase, limit_text, assumption, consistent, working, verdict_words) -> dict:
    return dict(
        context=context,
        model=fm,
        params=params,
        limit_phrase=limit_phrase,
        limit_text=limit_text,
        assumption=assumption,
        consistent=consistent,
        working=working,
        verdict_words=verdict_words,
    )


# ---------------------------------------------------------------------------
# Easy
# ---------------------------------------------------------------------------
def _cooling(rng) -> dict:
    ta = rng.choice((18, 20, 21, 22, 24))
    t0 = rng.choice((70, 75, 80, 85, 90)) if rng.random() < 0.7 else rng.choice((2, 4, 5))
    k = rng.choice((0.03, 0.05, 0.08, 0.1, 0.12))
    ok = rng.random() < 0.5
    room = ta if ok else ta + rng.choice((-6, -4, 5, 8))
    fm = Fm("T(t)=T_{a}+(T_{0}-T_{a})e^{-kt}", "T(t) = Ta + (T0 - Ta) e^(-kt)")
    return _model(
        "the temperature of a drink left on a bench",
        fm,
        [
            ("T", "temperature of the drink", None, "°C"),
            ("t", "time", None, "minutes"),
            ("Ta", "temperature used in the model for the surroundings", _n(ta), "°C"),
            ("T0", "starting temperature", _n(t0), "°C"),
            ("k", "rate constant", _n(k), "per minute"),
        ],
        "as t tends to infinity",
        f"{_n(ta)} °C",
        f"the bench is in a room at {_n(room)} °C and the drink should end up at room temperature",
        ok,
        [
            f"e^(-kt) tends to 0 as t tends to infinity, so T(t) tends to Ta = {_n(ta)} °C.",
            f"The room temperature is {_n(room)} °C; the limit {'equals' if ok else 'differs from'} it.",
        ],
        "the limit equals room temperature"
        if ok
        else f"the limit {_n(ta)} °C is not the room temperature {_n(room)} °C",
    )


def _rc(rng) -> dict:
    e = rng.choice((5, 6, 9, 12, 15, 24))
    vs = e if rng.random() < 0.5 else rng.choice((4, 5, 6, 9, 12))
    tau = rng.choice((2, 3, 4, 5, 10))
    ok = e <= vs
    fm = Fm("V(t)=E\\left(1-e^{-t/\\tau}\\right)", "V(t) = E (1 - e^(-t/τ))")
    return _model(
        "the voltage across a charging capacitor",
        fm,
        [
            ("V", "voltage across the capacitor", None, "volts"),
            ("t", "time", None, "seconds"),
            ("E", "voltage the model charges towards", _n(e), "volts"),
            ("τ", "time constant", _n(tau), "seconds"),
        ],
        "as t tends to infinity",
        f"{_n(e)} V",
        f"the supply is {_n(vs)} V and a capacitor charged from a supply cannot end with a voltage above the supply voltage",
        ok,
        [
            f"e^(-t/τ) tends to 0 as t tends to infinity, so V(t) tends to E = {_n(e)} V.",
            f"The supply voltage is {_n(vs)} V and {_n(e)} is {'not above' if ok else 'above'} {_n(vs)}.",
        ],
        f"the limit {_n(e)} V does not exceed the {_n(vs)} V supply"
        if ok
        else f"the limit {_n(e)} V is above the {_n(vs)} V supply",
    )


def _decay(rng) -> dict:
    n0 = rng.choice((800, 1000, 2000, 5000))
    lam = rng.choice((0.02, 0.05, 0.1, 0.2))
    if rng.random() < 0.55:
        fm = Fm("N(t)=N_{0}e^{-\\lambda t}", "N(t) = N0 e^(-λt)")
        limit, ok = "0 atoms", True
        work = ["e^(-λt) tends to 0 as t tends to infinity, so N(t) tends to 0 atoms."]
        words = "the limit 0 atoms fits: the count never goes negative and the sample decays away"
    else:
        lam_l = Fraction(rng.choice((1, 2, 5)), 100)
        fm = Fm("N(t)=N_{0}\\left(1-\\mu t\\right)", "N(t) = N0 (1 - μt)")
        lam = lam_l
        limit, ok = "minus infinity", False
        work = [
            f"N(t) = N0 - N0 μ t decreases by a fixed amount every day, so N(t) tends to minus infinity as t tends to infinity.",
            "A number of atoms cannot be negative.",
        ]
        words = "the model gives negative numbers of atoms, which is impossible"
    sym = "λ" if ok else "μ"
    unit = "per day"
    return _model(
        "the number of atoms of a radioactive sample",
        fm,
        [
            ("N", "number of atoms", None, "atoms"),
            ("t", "time", None, "days"),
            ("N0", "starting number of atoms", _n(n0), "atoms"),
            (sym, "decay constant", _n(lam), unit),
        ],
        "as t tends to infinity",
        limit,
        "the number of atoms can never be negative, and a sample decays away completely after a long time",
        ok,
        work,
        words,
    )


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------
def _logistic(rng) -> dict:
    kcap = rng.choice((200, 300, 400, 500, 800))
    cap = kcap if rng.random() < 0.4 else rng.choice((150, 250, 300, 350, 600, 1000))
    a = rng.choice((3, 4, 9, 19))
    r = rng.choice((0.2, 0.3, 0.5))
    ok = kcap <= cap
    fm = Fm("P(t)=\\frac{K}{1+Ae^{-rt}}", "P(t) = K/(1 + A e^(-rt))")
    return _model(
        "the number of fish in a pond",
        fm,
        [
            ("P", "number of fish", None, "fish"),
            ("t", "time", None, "years"),
            ("K", "constant in the model", _n(kcap), "fish"),
            ("A", "constant in the model", _n(a), "no units"),
            ("r", "growth rate", _n(r), "per year"),
        ],
        "as t tends to infinity",
        f"{_n(kcap)} fish",
        f"the pond can hold at most {_n(cap)} fish",
        ok,
        [
            f"e^(-rt) tends to 0 as t tends to infinity, so the denominator tends to 1 and P(t) tends to K = {_n(kcap)} fish.",
            f"{_n(kcap)} {'does not exceed' if ok else 'is more than'} the pond's limit of {_n(cap)} fish.",
        ],
        f"the limit {_n(kcap)} fish does not exceed {_n(cap)}"
        if ok
        else f"the limit {_n(kcap)} fish is more than the {_n(cap)} the pond can hold",
    )


def _terminal(rng) -> dict:
    vt = rng.choice((35, 40, 49, 56, 70))
    k = Fraction(98, 10) / vt
    measured = vt if rng.random() < 0.5 else rng.choice((30, 45, 60, 80))
    ok = measured == vt
    fm = Fm("v(t)=\\frac{g}{k}\\left(1-e^{-kt}\\right)", "v(t) = (g/k)(1 - e^(-kt))")
    return _model(
        "the speed of an object falling through air",
        fm,
        [
            ("v", "downward speed", None, "m/s"),
            ("t", "time", None, "seconds"),
            ("g", "acceleration due to gravity", "9.8", "m/s^2"),
            ("k", "drag constant", _n(k), "per second"),
        ],
        "as t tends to infinity",
        f"{_n(vt)} m/s",
        f"experiments measure the terminal speed of this object as {_n(measured)} m/s",
        ok,
        [
            f"e^(-kt) tends to 0, so v(t) tends to g/k = 9.8/{_n(k)} = {_n(vt)} m/s.",
            f"The measured terminal speed is {_n(measured)} m/s, which {'matches' if ok else 'does not match'} {_n(vt)} m/s.",
        ],
        "the limit matches the measured terminal speed"
        if ok
        else f"the limit {_n(vt)} m/s does not match the measured {_n(measured)} m/s",
    )


def _doses(rng) -> dict:
    for _ in attempts(200, "a dose with a terminating limit"):
        d = rng.choice((3, 4, 5, 6, 8, 10, 12))
        q = rng.choice((Fraction(1, 2), Fraction(1, 5), Fraction(1, 4), Fraction(2, 5), Fraction(3, 5), Fraction(4, 5)))
        limit = d / (1 - q)
        if (limit * 1000).denominator == 1:
            break
    toxic = limit + rng.choice((-1, 2, 5, 10)) if rng.random() < 0.7 else limit + 1
    ok = limit < toxic
    fm = Fm("C_{n}=D\\,\\frac{1-q^{n}}{1-q}", "C_n = D (1 - q^n)/(1 - q)")
    return _model(
        "the concentration of a drug just after the n-th daily dose",
        fm,
        [
            ("C_n", "concentration just after the n-th dose", None, "mg/L"),
            ("n", "number of doses given", None, "no units"),
            ("D", "increase from one dose", _n(d), "mg/L"),
            ("q", "fraction left after a day", _n(q), "no units"),
        ],
        "as n tends to infinity",
        f"{_n(limit)} mg/L",
        f"the concentration must stay below the toxic level of {_n(toxic)} mg/L",
        ok,
        [
            f"q^n tends to 0 as n tends to infinity since 0 < q < 1, so C_n tends to D/(1 - q) = {_n(d)}/(1 - {_n(q)}) = {_n(limit)} mg/L.",
            f"{_n(limit)} is {'below' if ok else 'not below'} the toxic level {_n(toxic)}.",
        ],
        f"the limit {_n(limit)} mg/L stays below the toxic level"
        if ok
        else f"the limit {_n(limit)} mg/L reaches or passes the toxic level {_n(toxic)} mg/L",
    )


def _ratio(rng) -> dict:
    lim = Fraction(rng.choice((55, 70, 80, 90, 100, 110, 120, 150)))
    c = rng.choice((1, 2, 4))
    a = int(lim * c)
    b = rng.choice((10, 20, 30, 40))
    d = rng.choice((2, 4, 5, 8))
    ok = lim <= 100
    fm = Fm("y(t)=\\frac{at+b}{ct+d}", "y(t) = (a t + b)/(c t + d)")
    return _model(
        "the percentage yield of a chemical reaction after t minutes",
        fm,
        [
            ("y", "percentage yield", None, "per cent"),
            ("t", "time", None, "minutes"),
            ("a", "constant", _n(a), "per cent per minute"),
            ("b", "constant", _n(b), "per cent"),
            ("c", "constant", _n(c), "no units"),
            ("d", "constant", _n(d), "minutes"),
        ],
        "as t tends to infinity",
        f"{_n(lim)} per cent",
        "a yield is a percentage of what is possible, so it cannot exceed 100 per cent",
        ok,
        [
            f"Divide the top and bottom by t: y(t) = (a + b/t)/(c + d/t), which tends to a/c = {_n(a)}/{_n(c)} = {_n(lim)} per cent.",
            f"{_n(lim)} {'does not exceed' if ok else 'exceeds'} 100.",
        ],
        f"the limit {_n(lim)} per cent does not exceed 100" if ok else f"the limit {_n(lim)} per cent is above 100",
    )


# ---------------------------------------------------------------------------
# Hard: a parameter limit
# ---------------------------------------------------------------------------
def _drag_to_zero(rng) -> dict:
    c = rng.choice((1, 1, 2, Fraction(1, 2), 3))
    g, world = rng.choice(
        (
            (Fraction(98, 10), "on Earth"),
            (Fraction(37, 10), "on Mars"),
            (Fraction(16, 10), "on the Moon"),
            (Fraction(248, 10), "on Jupiter"),
        )
    )
    ok = c == 1
    cstr = "" if c == 1 else _n(c)
    t_expr = f"{cstr}kt" if c != 1 else "kt"
    fm = Fm(f"v(t)=\\frac{{g}}{{k}}\\left(1-e^{{-{cstr}kt}}\\right)", f"v(t) = (g/k)(1 - e^(-{t_expr}))")
    gt = g * Fraction(c)
    return _model(
        f"the speed of an object falling through a thin atmosphere {world}",
        fm,
        [
            ("v", "downward speed", None, "m/s"),
            ("t", "time (held fixed)", None, "seconds"),
            ("g", "acceleration due to gravity", _n(g), "m/s^2"),
            ("k", "drag constant", None, "per second"),
        ],
        "as k tends to 0 from above (with t fixed)",
        f"{_n(gt)} t m/s",
        f"with no air resistance an object dropped from rest has speed {_n(g)}t after t seconds",
        ok,
        [
            f"As k tends to 0 the expression is 0/0 in the form (1 - e^(-{t_expr}))/k. Use e^(-x) ≈ 1 - x for small x: 1 - e^(-{t_expr}) ≈ {t_expr}, so v ≈ (g/k)({t_expr}) = {_n(Fraction(c))}gt.",
            f"So the limit is {_n(gt)} t m/s, which {'is' if ok else 'is not'} {_n(g)}t.",
        ],
        f"the limit is {_n(g)}t, the drag-free speed"
        if ok
        else f"the limit {_n(gt)}t is not the drag-free speed {_n(g)}t",
    )


def _leaving_to_zero(rng) -> dict:
    c = rng.choice((1, 1, 2, 3))
    b = rng.choice((200, 300, 500, 800))
    p0 = rng.choice((5000, 8000, 12000, 20000))
    ok = c == 1
    cstr = "" if c == 1 else _n(c)
    fm = Fm(
        f"P(t)=P_{{0}}e^{{-kt}}+\\frac{{b}}{{k}}\\left(1-e^{{-{cstr}kt}}\\right)",
        f"P(t) = P0 e^(-kt) + (b/k)(1 - e^(-{cstr}kt))",
    )
    return _model(
        "the population of a town with steady arrivals and a rate of leaving",
        fm,
        [
            ("P", "population", None, "people"),
            ("t", "time (held fixed)", None, "years"),
            ("P0", "starting population", _n(p0), "people"),
            ("b", "people arriving each year", _n(b), "people per year"),
            ("k", "rate of leaving", None, "per year"),
        ],
        "as k tends to 0 from above (with t fixed)",
        f"{_n(p0)} + {_n(b * c)} t people",
        f"if nobody ever leaves, the population grows by {_n(b)} people each year, so it is {_n(p0)} + {_n(b)}t after t years",
        ok,
        [
            f"P0 e^(-kt) tends to P0. The second term is (b/k)(1 - e^(-{cstr}kt)) = b t × (1 - e^(-x))/x with x = {cstr}kt, which tends to {_n(c)}bt as k tends to 0 (using (1 - e^(-x))/x → 1).",
            f"So the limit is {_n(p0)} + {_n(b * c)}t people, which {'is' if ok else 'is not'} {_n(p0)} + {_n(b)}t.",
        ],
        f"the limit equals {_n(p0)} + {_n(b)}t, the no-leaving growth"
        if ok
        else f"the limit {_n(p0)} + {_n(b * c)}t is not the no-leaving growth {_n(p0)} + {_n(b)}t",
    )


def _resistance_to_zero(rng) -> dict:
    e = rng.choice((5, 6, 9, 12))
    ok = rng.random() < 0.55
    if ok:
        fm = Fm("V(t)=E\\left(1-e^{-t/(RC)}\\right)", "V(t) = E (1 - e^(-t/(RC)))")
        limit = f"{_n(e)} V"
        work = [
            "For fixed t > 0, t/(RC) tends to infinity as R tends to 0 from above, so e^(-t/(RC)) tends to 0 and V(t) tends to E = "
            + f"{_n(e)} V."
        ]
        words = f"the limit {_n(e)} V is the supply voltage reached straight away"
    else:
        fm = Fm("V(t)=E\\left(1-e^{-RCt}\\right)", "V(t) = E (1 - e^(-RCt))")
        limit = "0 V"
        work = [
            "For fixed t, RCt tends to 0 as R tends to 0 from above, so e^(-RCt) tends to 1 and V(t) tends to E(1 - 1) = 0 V."
        ]
        words = f"the limit 0 V is not the supply voltage {_n(e)} V"
    cc = rng.choice((2, 4, 5))
    return _model(
        "the voltage across a capacitor charged through a resistor",
        fm,
        [
            ("V", "voltage across the capacitor", None, "volts"),
            ("t", "time (held fixed, positive)", None, "seconds"),
            ("E", "supply voltage", _n(e), "volts"),
            ("R", "resistance", None, "ohms"),
            ("C", "capacitance", _n(cc), "farads"),
        ],
        "as R tends to 0 from above (with t fixed and positive)",
        limit,
        f"with no resistance in the circuit the capacitor charges to the full supply voltage of {_n(e)} V straight away",
        ok,
        work + [f"The supply voltage is {_n(e)} V and the limit is {limit}."],
        words,
    )


# ---------------------------------------------------------------------------
# Exam
# ---------------------------------------------------------------------------
def _damped(rng) -> dict:
    a = rng.choice((0.05, 0.1, 0.2, 0.3))
    which = rng.choice(("damped", "undamped", "growing"))
    if which == "damped":
        fm = Fm("x(t)=Ae^{-ct}\\cos(\\omega t)", "x(t) = A e^(-ct) cos(ωt)")
        c_txt, limit, ok = _n(a), "0 m", True
        work = [
            "|cos(ωt)| ≤ 1 and e^(-ct) tends to 0, so |x(t)| ≤ A e^(-ct) tends to 0 (squeeze); x(t) tends to 0 m.",
            "The mass comes to rest at x = 0.",
        ]
        words = "the limit 0 m fits: the oscillation dies away to rest at x = 0"
    elif which == "undamped":
        fm = Fm("x(t)=A\\cos(\\omega t)", "x(t) = A cos(ωt)")
        c_txt, limit, ok = "0", "no limit (it keeps oscillating between -A and A)", False
        work = [
            "cos(ωt) keeps oscillating between -1 and 1, so x(t) has no limit.",
            "Friction should make the oscillation die away, which this model does not do.",
        ]
        words = "there is no limit, but friction should bring the mass to rest"
    else:
        fm = Fm("x(t)=Ae^{ct}\\cos(\\omega t)", "x(t) = A e^(ct) cos(ωt)")
        c_txt, limit, ok = _n(a), "no limit (the oscillations grow without bound)", False
        work = [
            "e^(ct) tends to infinity, so the amplitude of x(t) grows without bound and x(t) has no limit.",
            "A mass with friction cannot gain amplitude.",
        ]
        words = "the oscillation grows, but friction should bring the mass to rest"
    A = rng.choice((0.05, 0.1, 0.2))
    om = rng.choice((2, 3, 5, 8))
    return _model(
        "the position of a mass on a spring with friction",
        fm,
        [
            ("x", "displacement from rest", None, "m"),
            ("t", "time", None, "seconds"),
            ("A", "starting displacement", _n(A), "m"),
            ("c", "friction constant in the model", c_txt, "per second"),
            ("ω", "angular frequency", _n(om), "radians per second"),
        ],
        "as t tends to infinity",
        limit,
        "friction is present, so the mass should come to rest at x = 0 after a long time",
        ok,
        work,
        words,
    )


def _terminal_and_drag(rng) -> dict:
    vt = rng.choice((35, 40, 49, 56, 70))
    k = Fraction(98, 10) / vt
    c = rng.choice((1, 2))
    measured = vt if rng.random() < 0.5 else rng.choice((30, 45, 60, 80))
    ok1 = measured == vt
    ok2 = c == 1
    cstr = "" if c == 1 else "2"
    fm = Fm(f"v(t)=\\frac{{g}}{{k}}\\left(1-e^{{-{cstr}kt}}\\right)", f"v(t) = (g/k)(1 - e^(-{cstr}kt))")
    # as t -> infinity the limit is g/k; the terminal speed in the model is g/k whatever c is
    return dict(
        _model(
            "the speed of an object falling through air",
            fm,
            [
                ("v", "downward speed", None, "m/s"),
                ("t", "time", None, "seconds"),
                ("g", "acceleration due to gravity", "9.8", "m/s^2"),
                ("k", "drag constant", _n(k), "per second"),
            ],
            "as t tends to infinity",
            f"{_n(vt)} m/s",
            f"experiments measure the terminal speed as {_n(measured)} m/s",
            ok1,
            [
                f"As t tends to infinity e^(-{cstr}kt) tends to 0 so v tends to g/k = 9.8/{_n(k)} = {_n(vt)} m/s.",
                f"The measured value is {_n(measured)} m/s: {'consistent' if ok1 else 'not consistent'}.",
            ],
            "the limit matches the measured terminal speed"
            if ok1
            else f"the limit {_n(vt)} m/s does not match the measured {_n(measured)} m/s",
        ),
        second=dict(
            limit_phrase="as k tends to 0 from above (with t fixed)",
            limit_text=f"{_n(Fraction(98, 10) * c)} t m/s",
            assumption="with no air resistance an object dropped from rest has speed 9.8t after t seconds",
            consistent=ok2,
            working=[
                f"As k tends to 0, (1 - e^(-{cstr}kt))/k tends to {cstr or '1'}t (since 1 - e^(-x) ≈ x), so v tends to {_n(Fraction(98, 10) * c)} t m/s.",
                f"This {'equals' if ok2 else 'is not'} 9.8t.",
            ],
            verdict_words="the drag-free limit is 9.8t"
            if ok2
            else f"the drag-free limit {_n(Fraction(98, 10) * c)}t is not 9.8t",
        ),
    )


def _build(rng, d: dict, *, hint: bool, ask_units: bool = True) -> dict:
    p = Prompt()
    p.t(f"A model of {d['context']} is ").m(d["model"]).t(". ")
    parts = []
    for sym, meaning, value, unit in d["params"]:
        if value is None:
            parts.append(f"{sym} is the {meaning}, in {unit}")
        else:
            parts.append(
                f"{sym} = {value} {unit if unit != 'no units' else ''} ({meaning})".replace("  ", " ").replace(
                    " )", ")"
                )
            )
    p.t("Here " + "; ".join(parts) + ". ")
    p.t(f"A physical fact is that {d['assumption']}. ")
    sec = d.get("second")
    if sec:
        p.t(f"A second fact is that {sec['assumption']}. ")
        p.t(
            f"State the limit of the model {d['limit_phrase']} and the limit {sec['limit_phrase']}, each with units, and say for each whether it is consistent with the fact about it. "
        )
    else:
        p.t(
            f"State the limit of the model {d['limit_phrase']}, with units, and say whether it is consistent with this fact. "
        )
    p.t("Show how you found the limit. ").t(rng.choice(SHORT_TAILS))
    verd = "consistent" if d["consistent"] else "not consistent"
    ref = list(d["working"]) + [f"Verdict: {verd}, because {d['verdict_words']}."]
    steps = [
        ("correct_idea", f"finds the limit {d['limit_phrase']}: {d['limit_text']}"),
        ("correct_idea", f"gives the verdict '{verd}' with the reason: {d['verdict_words']}"),
        (
            "completeness",
            "explains why the limit is that value (for example which term vanishes) and compares it with the number in the fact",
        ),
        ("use_of_context", "shows the working, with the numbers substituted"),
        ("communication", "writes the units with the limit and ends with a clear one-line conclusion"),
    ]
    traps = [
        ("limit_without_units", "is the limit given without units, or with the wrong units?"),
        ("wrong_verdict", f"does it say the limit is {'not ' if d['consistent'] else ''}consistent with the fact?"),
        ("limit_not_justified", "does it state the limit without saying why (for example which term tends to 0)?"),
    ]
    answer = f"Limit {d['limit_phrase']}: {d['limit_text']}; {verd}"
    if sec:
        verd2 = "consistent" if sec["consistent"] else "not consistent"
        ref += list(sec["working"]) + [f"Second verdict: {verd2}, because {sec['verdict_words']}."]
        steps.insert(
            2,
            (
                "correct_idea",
                f"finds the second limit {sec['limit_phrase']}: {sec['limit_text']}, and its verdict '{verd2}'",
            ),
        )
        traps.append(("second_limit_missed", "does it deal with both limits and both facts?"))
        answer += f". Limit {sec['limit_phrase']}: {sec['limit_text']}; {verd2}"
    mark = short_marking(reference=ref, steps=steps, traps=traps)
    return finish(p, answer, mark)


def _easy(rng) -> dict:
    return _build(rng, rng.choice((_cooling, _rc, _decay))(rng), hint=True)


def _medium(rng) -> dict:
    return _build(rng, rng.choice((_logistic, _terminal, _doses, _ratio))(rng), hint=False)


def _hard(rng) -> dict:
    return _build(rng, rng.choice((_drag_to_zero, _leaving_to_zero, _resistance_to_zero))(rng), hint=False)


def _exam(rng) -> dict:
    if rng.random() < 0.5:
        return _build(rng, _terminal_and_drag(rng), hint=False)
    return _build(rng, _damped(rng), hint=False)


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("model_limit_validation", level=17, presets=_PRESETS)
def model_limit_validation(rng, variant="easy"):
    """`ucv1-model-limit-validation`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
