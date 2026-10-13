"""Generates `model_assumption_audit` items: name the variables, parameters and units of a small dynamical model
and identify the one limiting assumption that a stated fact violates.

Every model is a closed-form or one-line differential model with numbers (an exponential population, Newton
cooling, free fall, a filling tank, zero-order drug elimination, a spring, a pendulum). Each lists its
assumptions and states a physical fact; the fact is contradicted by the model at a computed time or value, and
the generator computes it (and a margin from the nearest rounding boundary) so the evidence a marker looks for
is exact. The answer is marked against a rubric, so `answer` is a marking dict (see `_crit_kit`), not a value.

Ladder:

- easy: three assumptions are listed and one fact contradicts one of them; name the variables with units and the
  violated assumption.
- medium: the assumptions are not listed; the learner states the violated assumption in words and shows the
  contradiction with a calculation of when it happens.
- hard: four assumptions, a second fact that supports one of them (not a violation), and a calculation; the
  learner must say which assumption fails and why the others are not contradicted.
- exam: two physical limits apply; the learner works out which is reached first, so which assumption the model
  breaks first, and when.
"""

from __future__ import annotations

import math

from ..core import check, generator
from ._crit_kit import attempts, Fm, Prompt, SHORT_TAILS, finish, short_marking

_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}


def _r1(x: float) -> str:
    return f"{x:.1f}"


def _safe(x: float) -> bool:
    """x to one decimal place is not within 0.02 of a rounding boundary."""
    return abs(((x * 10) % 1) - 0.5) > 0.2


# ---------------------------------------------------------------------------
# Model families. Each returns a dict:
#   context, model (Fm), vars [(symbol, meaning, unit)], params [(symbol, meaning, value, unit)],
#   assumptions [str] (the first listed is not necessarily the violated one), violated (index),
#   fact (str), support (str, a fact consistent with another assumption), t_star (float), t_unit, event (str),
#   other: second limit for the exam family.
# ---------------------------------------------------------------------------
def _exponential(rng) -> dict:
    for _ in attempts(500, "an exponential population"):
        p0 = rng.choice((20, 30, 40, 50, 60, 80))
        r = rng.choice((0.08, 0.1, 0.12, 0.15, 0.2, 0.25))
        cap = rng.choice((300, 400, 500, 800, 1000, 1500))
        t = math.log(cap / p0) / r
        if cap / p0 >= 5 and _safe(t):
            break
    a = [
        "the growth rate per animal is the same at every population size",
        "food and space are unlimited, so the population can grow without any upper limit",
        "there is no migration into or out of the island",
        "births and deaths happen continuously",
    ]
    return dict(
        key="exponential",
        context="the number of rabbits on an island",
        model=Fm("\\frac{dP}{dt}=rP,\\quad P(0)=P_{0}", "dP/dt = r P, P(0) = P0"),
        solution=f"P(t) = {p0} e^({r}t)",
        vars=[("P", "number of rabbits", "rabbits"), ("t", "time", "years")],
        params=[("r", "growth rate per rabbit", f"{r}", "per year"), ("P0", "starting population", f"{p0}", "rabbits")],
        assumptions=a,
        violated=1,
        fact=f"the island's food and space can support at most {cap} rabbits",
        support="a count shows no rabbits have arrived or left the island, and the births per rabbit have stayed steady",
        support_ok=[0, 2],
        t_star=t,
        t_unit="years",
        event=f"the model first predicts more than {cap} rabbits after t = {_r1(t)} years (P0 e^(rt) = {cap} gives t = ln({cap}/{p0})/{r})",
        limit_value=cap,
        limit_name=f"{cap} rabbits",
    )


def _cooling(rng) -> dict:
    for _ in attempts(500, "a cooling liquid"):
        t0 = rng.choice((70, 80, 85, 90, 95))
        ta = rng.choice((-3, -4, -5, -8, -10))
        k = rng.choice((0.02, 0.03, 0.04, 0.05, 0.06))
        t = math.log((t0 - ta) / (0 - ta)) / k
        if _safe(t):
            break
    a = [
        "the surroundings stay at the same temperature",
        "the liquid stays liquid (it does not freeze or boil)",
        "the temperature is the same throughout the liquid",
        "the rate of cooling is proportional to the temperature difference",
    ]
    return dict(
        key="cooling",
        context="a cup of water left in a cold store",
        model=Fm("\\frac{dT}{dt}=-k(T-T_{a}),\\quad T(0)=T_{0}", "dT/dt = -k (T - Ta), T(0) = T0"),
        solution=f"T(t) = {ta} + {t0 - ta} e^(-{k}t)",
        vars=[("T", "temperature of the water", "°C"), ("t", "time", "minutes")],
        params=[
            ("k", "cooling constant", f"{k}", "per minute"),
            ("Ta", "temperature of the cold store", f"{ta}", "°C"),
            ("T0", "starting temperature", f"{t0}", "°C"),
        ],
        assumptions=a,
        violated=1,
        fact="pure water freezes at 0 °C",
        support="a thermometer in the store reads the same temperature all through the experiment and the water is stirred",
        support_ok=[0, 2, 3],
        t_star=t,
        t_unit="minutes",
        event=f"the model first gives T below 0 °C after t = {_r1(t)} minutes ({ta} + {t0 - ta} e^(-{k}t) = 0)",
        limit_value=0,
        limit_name="0 °C",
    )


def _free_fall(rng) -> dict:
    for _ in attempts(500, "a falling object"):
        vt = rng.choice((40, 45, 50, 55, 60, 70))
        g = 9.8
        t = vt / g
        if _safe(t):
            break
    a = [
        "gravity gives a constant acceleration",
        "there is no air resistance",
        "the object is a point that does not spin",
        "the ground is level and the object falls vertically",
    ]
    return dict(
        key="free_fall",
        context="a skydiver falling from a plane",
        model=Fm("\\frac{dv}{dt}=g,\\quad v(0)=0", "dv/dt = g, v(0) = 0"),
        solution="v(t) = 9.8 t",
        vars=[("v", "downward speed", "m/s"), ("t", "time since leaving the plane", "seconds")],
        params=[("g", "acceleration due to gravity", "9.8", "m/s^2")],
        assumptions=a,
        violated=1,
        fact=f"measurements show the skydiver never goes faster than {vt} m/s",
        support="the skydiver falls straight down over level ground",
        support_ok=[0, 3],
        t_star=t,
        t_unit="seconds",
        event=f"the model first predicts a speed above {vt} m/s after t = {_r1(t)} seconds (9.8t = {vt})",
        limit_value=vt,
        limit_name=f"{vt} m/s",
    )


def _tank(rng) -> dict:
    for _ in attempts(500, "a filling tank"):
        v0 = rng.choice((100, 150, 200, 250, 300))
        rin = rng.choice((18, 20, 25, 30, 35))
        rout = rng.choice((8, 10, 12, 15))
        cap = rng.choice((1000, 1200, 1500, 2000))
        if rin <= rout:
            continue
        t = (cap - v0) / (rin - rout)
        if t > 5 and _safe(t):
            break
    a = [
        "water flows in at a constant rate",
        "water flows out at a constant rate",
        "the tank never overflows",
        "the water is well mixed",
    ]
    return dict(
        key="tank",
        context="water in a storage tank with a tap in and a tap out",
        model=Fm("\\frac{dV}{dt}=r_{in}-r_{out},\\quad V(0)=V_{0}", "dV/dt = r_in - r_out, V(0) = V0"),
        solution=f"V(t) = {v0} + {rin - rout}t",
        vars=[("V", "volume of water in the tank", "litres"), ("t", "time", "minutes")],
        params=[
            ("r_in", "rate water flows in", f"{rin}", "litres per minute"),
            ("r_out", "rate water flows out", f"{rout}", "litres per minute"),
            ("V0", "starting volume", f"{v0}", "litres"),
        ],
        assumptions=a,
        violated=2,
        fact=f"the tank holds at most {cap} litres",
        support="both taps are measured to flow steadily, and a dye test shows the water is evenly mixed",
        support_ok=[0, 1, 3],
        t_star=t,
        t_unit="minutes",
        event=f"the model first gives V above {cap} litres after t = {_r1(t)} minutes ({v0} + {rin - rout}t = {cap})",
        limit_value=cap,
        limit_name=f"{cap} litres",
    )


def _drug(rng) -> dict:
    for _ in attempts(500, "a drug concentration"):
        c0 = rng.choice((6, 8, 10, 12, 15, 20))
        k = rng.choice((0.8, 1.2, 1.5, 2.0, 2.5, 3.0))
        t = c0 / k
        if _safe(t):
            break
    a = [
        "the drug leaves the blood at the same rate whatever the concentration",
        "the concentration of the drug can never be negative",
        "the drug is spread evenly through the blood",
        "no more drug is given after the first dose",
    ]
    return dict(
        key="drug",
        context="the concentration of a drug in a patient's blood after a single dose",
        model=Fm("\\frac{dC}{dt}=-k,\\quad C(0)=C_{0}", "dC/dt = -k, C(0) = C0"),
        solution=f"C(t) = {c0} - {k}t",
        vars=[("C", "concentration of the drug", "mg/L"), ("t", "time since the dose", "hours")],
        params=[
            ("k", "rate the drug leaves the blood", f"{k}", "mg/L per hour"),
            ("C0", "starting concentration", f"{c0}", "mg/L"),
        ],
        assumptions=a,
        violated=0,
        fact="a drug leaves the body more slowly as its concentration falls, so the concentration only approaches 0 and never goes below 0",
        support="no second dose was given, and samples from different veins gave matching concentrations",
        support_ok=[2, 3],
        t_star=t,
        t_unit="hours",
        event=f"the model reaches C = 0 at t = {_r1(t)} hours and gives negative concentrations after it ({c0} - {k}t = 0)",
        limit_value=0,
        limit_name="0 mg/L",
    )


def _spring(rng) -> dict:
    for _ in attempts(500, "a spring"):
        amp = rng.choice((0.15, 0.18, 0.2, 0.22, 0.25))
        lim = rng.choice((0.08, 0.1, 0.12))
        k = rng.choice((40, 60, 90, 120))
        m = rng.choice((0.4, 0.5, 0.8, 1.0, 1.5))
        w = math.sqrt(k / m)
        t = math.acos(lim / amp) / w
        if amp > lim * 1.3 and 0 < t and abs(((t * 100) % 1) - 0.5) > 0.2:
            break
    a = [
        "the spring force is proportional to the stretch (Hooke's law), however large the stretch",
        "there is no friction or air resistance",
        "the mass moves only up and down",
        "the spring itself has no mass",
    ]
    return dict(
        key="spring",
        context="a mass hanging on a spring and pulled down then released",
        model=Fm("m\\frac{d^{2}x}{dt^{2}}=-kx,\\quad x(0)=A,\\ x'(0)=0", "m d^2x/dt^2 = -k x, x(0) = A, x'(0) = 0"),
        solution=f"x(t) = {amp} cos({w:.2f}t)",
        vars=[("x", "displacement of the mass from rest", "m"), ("t", "time", "seconds")],
        params=[
            ("m", "mass", f"{m}", "kg"),
            ("k", "spring constant", f"{k}", "N/m"),
            ("A", "starting displacement", f"{amp}", "m"),
        ],
        assumptions=a,
        violated=0,
        fact=f"this spring stays linear only for stretches up to {lim} m; beyond that the force is no longer proportional to the stretch",
        support="the apparatus is oiled and measured displacement shows no loss of energy over a few cycles",
        support_ok=[1],
        t_star=t,
        t_unit="seconds",
        event=f"the model's displacement first reaches {lim} m (the end of the linear range) at t = {t:.2f} seconds, from {amp} cos({w:.2f}t) = {lim}, and goes on to {amp} m, beyond the linear range",
        limit_value=lim,
        limit_name=f"{lim} m",
        t_decimals=2,
    )


def _pendulum(rng) -> dict:
    theta = rng.choice((0.6, 0.7, 0.9, 1.0, 1.1, 1.2))
    err = theta / math.sin(theta) - 1
    check(err > 0.05, "a visible small-angle error")
    a = [
        "the angle stays small enough that sin(θ) can be replaced by θ",
        "there is no friction at the pivot or air resistance",
        "the string is light and does not stretch",
        "the bob is a point mass",
    ]
    pct = 100 * err
    return dict(
        key="pendulum",
        context="a pendulum released from rest",
        model=Fm(
            "\\frac{d^{2}\\theta}{dt^{2}}=-\\frac{g}{L}\\theta,\\quad \\theta(0)=\\theta_{0}",
            "d^2θ/dt^2 = -(g/L) θ, θ(0) = θ0",
        ),
        solution="θ(t) = θ0 cos(sqrt(g/L) t)",
        vars=[("θ", "angle from the vertical", "radians"), ("t", "time", "seconds")],
        params=[
            ("g", "acceleration due to gravity", "9.8", "m/s^2"),
            ("L", "length of the string", "1.5", "m"),
            ("θ0", "starting angle", f"{theta}", "radians"),
        ],
        assumptions=a,
        violated=0,
        fact="the model is only accurate when replacing sin(θ) by θ makes an error of less than 1%",
        support="the string is a thin cord that does not stretch, and the pivot is well oiled",
        support_ok=[1, 2],
        t_star=None,
        t_unit="",
        event=f"at the starting angle θ0 = {theta} radians, θ/sin(θ) - 1 = {pct:.0f}%, far above 1%",
        limit_value=None,
        limit_name="1% error",
        pct=pct,
    )


FAMS = [_exponential, _cooling, _free_fall, _tank, _drug, _spring, _pendulum]
CLEAN_FAMS = [_exponential, _cooling, _free_fall, _tank, _drug]  # those with a time at which the limit is reached


def _letters(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _model_paragraph(p: Prompt, d: dict) -> None:
    p.t(f"A model of {d['context']} is ").m(d["model"]).t(". ")
    parts = []
    for sym, meaning, value, unit in d["params"]:
        parts.append(f"{sym} = {value} {unit} ({meaning})")
    p.t("The numbers are " + "; ".join(parts) + ". ")


def _assumption_list(p: Prompt, assumptions: list[str]) -> None:
    items = [f"{lab}: {txt}" for lab, txt in zip(_letters(len(assumptions)), assumptions)]
    p.t("The model assumes that (" + "; ".join(items) + "). ")


def _vars_phrase(d: dict) -> str:
    return (
        "; ".join(f"{s} is the {meaning}, in {unit}" for s, meaning, unit in d["vars"])
        + "; "
        + "; ".join(f"{s} is the {meaning}, in {unit}" for s, meaning, value, unit in d["params"])
    )


def _ref_common(d: dict, violated: int, order: list[int]) -> list[str]:
    lab = _letters(len(d["assumptions"]))
    ref = [
        "Variables: "
        + "; ".join(f"{s} ({meaning}) in {unit}" for s, meaning, unit in d["vars"])
        + ". Parameters: "
        + "; ".join(f"{s} ({meaning}) = {v} {u}" for s, meaning, v, u in d["params"])
        + ".",
        f"The violated assumption is {lab[order.index(violated)]}: {d['assumptions'][violated]}.",
        f"Evidence: {d['event']}.",
    ]
    return ref


def _marking_for(d: dict, violated: int, order: list[int], *, with_calc: bool, with_support: bool) -> dict:
    lab = _letters(len(d["assumptions"]))
    vlab = lab[order.index(violated)]
    steps = [
        (
            "correct_idea",
            "names every variable and parameter with its unit: "
            + "; ".join(f"{s} in {u}" for s, m, u in d["vars"])
            + "; "
            + "; ".join(f"{s} in {u}" for s, m, v, u in d["params"]),
        ),
        (
            "correct_idea",
            f"identifies the violated assumption: {d['assumptions'][violated]}"
            + (f" ({vlab})" if with_support or True else ""),
        ),
        ("completeness", f"links the fact ({d['fact']}) to the model's prediction to show the assumption fails"),
    ]
    if with_calc:
        steps.append(("use_of_context", f"calculates the contradiction: {d['event']}"))
    else:
        steps.append(
            (
                "use_of_context",
                f"uses the numbers in the question (the stated limit and the model's equation) to show the model passes the limit",
            )
        )
    steps.append(("communication", "states units with every number and gives a clear one-line conclusion"))
    traps = [
        ("wrong_assumption", f"does it name a different assumption from: {d['assumptions'][violated]}?"),
        (
            "units_missing",
            "are the units of every variable and parameter stated, and are they right (for example 'per year' for a rate)?",
        ),
        (
            "model_is_just_wrong",
            "does it say only that the model is wrong or unrealistic, without naming the assumption and the evidence?",
        ),
    ]
    if with_calc:
        traps.append(("wrong_time", "does the time it calculates match the equation and the stated limit?"))
    if with_support:
        traps.append(
            (
                "accepts_support_as_violation",
                "does it treat the supporting fact (which is consistent with the model) as a violation?",
            )
        )
    return dict(steps=steps, traps=traps)


def _order(rng, n: int, violated: int) -> list[int]:
    """The order in which the assumptions are listed (a shuffle)."""
    idx = list(range(n))
    rng.shuffle(idx)
    return idx


def _item(rng, d: dict, *, list_assumptions: bool, with_calc: bool, with_support: bool, n_assumptions: int) -> dict:
    d = dict(d)
    pool = list(range(len(d["assumptions"])))
    # keep the violated one and the supported ones (if used) and fill with others up to n_assumptions
    chosen = [d["violated"]]
    if with_support:
        chosen += d["support_ok"][:1]
    for i in pool:
        if len(chosen) >= n_assumptions:
            break
        if i not in chosen:
            chosen.append(i)
    order_idx = list(chosen)
    rng.shuffle(order_idx)
    chosen_assumptions = [d["assumptions"][i] for i in order_idx]
    new_violated = order_idx.index(d["violated"])
    sub = dict(d)
    sub["assumptions"] = chosen_assumptions
    sub["violated"] = new_violated
    p = Prompt()
    _model_paragraph(p, sub)
    if list_assumptions:
        _assumption_list(p, chosen_assumptions)
    p.t(f"It is known that {sub['fact']}")
    if with_support:
        p.t(f", and also that {d['support']}")
    p.t(". ")
    if list_assumptions:
        q = "State each variable and parameter with its unit, and say which one of the listed assumptions the model violates. "
    else:
        q = "State each variable and parameter with its unit, and say in words which limiting assumption the model must be making that this fact contradicts. "
    if with_calc:
        if d["t_star"] is not None:
            places = "2 decimal places" if d["key"] == "spring" else "1 decimal place"
            q += f"Support your answer with a calculation of the time at which the contradiction first happens, to {places}, with units. "
        else:
            q += "Support your answer with a calculation of the size of the error as a percentage, to the nearest whole number. "
    else:
        q += "Use the numbers given. "
    p.t(q).t(rng.choice(SHORT_TAILS))
    ref = [
        "Variables: "
        + "; ".join(f"{s} ({m}) in {u}" for s, m, u in sub["vars"])
        + ". Parameters: "
        + "; ".join(f"{s} ({m}) = {v} {u}" for s, m, v, u in sub["params"])
        + ".",
        f"The violated assumption is: {sub['assumptions'][new_violated]}.",
        f"Evidence: {d['event']}.",
    ]
    if with_support:
        others = [a for i, a in enumerate(sub["assumptions"]) if i != new_violated]
        ref.append(f"The other assumptions are not contradicted: {d['support']}.")
    mk = _marking_for(
        sub, new_violated, list(range(len(chosen_assumptions))), with_calc=with_calc, with_support=with_support
    )
    mark = short_marking(reference=ref, steps=mk["steps"], traps=mk["traps"])
    # The answer states only the asked-for figure, at the asked-for precision; the working stays in the reference.
    if not with_calc:
        evidence = f"{d['event'][0].upper()}{d['event'][1:]}."
    elif d["t_star"] is not None:
        t_shown = f"{d['t_star']:.2f}" if d["key"] == "spring" else _r1(d["t_star"])
        evidence = f"The contradiction first happens at t = {t_shown} {d['t_unit']}."
    else:
        evidence = f"The small-angle error is {d['pct']:.0f}%."
    answer = f"Violated assumption: {sub['assumptions'][new_violated]}. {evidence}"
    return finish(p, answer, mark)


def _easy(rng) -> dict:
    return _item(
        rng, rng.choice(FAMS)(rng), list_assumptions=True, with_calc=False, with_support=False, n_assumptions=3
    )


def _medium(rng) -> dict:
    return _item(
        rng,
        rng.choice(CLEAN_FAMS + [_spring])(rng),
        list_assumptions=False,
        with_calc=True,
        with_support=False,
        n_assumptions=3,
    )


def _hard(rng) -> dict:
    return _item(
        rng,
        rng.choice(CLEAN_FAMS + [_spring, _pendulum])(rng),
        list_assumptions=True,
        with_calc=True,
        with_support=True,
        n_assumptions=4,
    )


def _exam_two_limits_population(rng) -> dict:
    for _ in attempts(500, "two population limits"):
        p0 = rng.choice((20, 30, 40, 50))
        r = rng.choice((0.1, 0.12, 0.15, 0.2))
        k_space = rng.choice((400, 500, 600, 800))
        k_food = rng.choice((300, 450, 700, 1000))
        if k_space == k_food:
            continue
        t_space = math.log(k_space / p0) / r
        t_food = math.log(k_food / p0) / r
        if _safe(t_space) and _safe(t_food) and abs(t_space - t_food) > 0.3:
            break
    first_is_space = t_space < t_food
    first = "space" if first_is_space else "food"
    t1 = min(t_space, t_food)
    a_space = "there is always enough space, so the population can keep growing"
    a_food = "there is always enough food, so each animal reproduces at the same rate"
    a_other = "the number of animals leaving the reserve equals the number arriving"
    p = Prompt()
    p.t("A model of the number of deer in a reserve is ").m(
        Fm("\\frac{dP}{dt}=rP,\\quad P(0)=P_{0}", "dP/dt = r P, P(0) = P0")
    ).t(
        f". The numbers are r = {r} per year (growth rate per deer) and P0 = {p0} deer. The model assumes that (A1: {a_space}; A2: {a_food}; A3: {a_other}). "
    )
    p.t(
        f"The reserve has space for at most {k_space} deer, and its food supply can feed at most {k_food} deer. Say which assumption the model breaks first, and when, giving the time to 1 decimal place with units. Name each variable and parameter with its unit. "
    )
    p.t(rng.choice(SHORT_TAILS))
    fact_first = f"{k_space} deer (space)" if first_is_space else f"{k_food} deer (food)"
    ref = [
        "Variables: P (number of deer) in deer; t (time) in years. Parameters: r in per year; P0 in deer.",
        f"Space limit {k_space}: t = ln({k_space}/{p0})/{r} = {_r1(t_space)} years. Food limit {k_food}: t = ln({k_food}/{p0})/{r} = {_r1(t_food)} years.",
        f"The {first} limit is reached first, at t = {_r1(t1)} years, so the model first breaks assumption {'A1' if first_is_space else 'A2'}: {a_space if first_is_space else a_food}.",
    ]
    steps = [
        ("correct_idea", "names P in deer, t in years, r per year, P0 in deer"),
        (
            "correct_idea",
            f"finds that the {first} limit is reached first ({fact_first}), at t = {_r1(t1)} years, so the first assumption broken is that {'there is always enough space' if first_is_space else 'there is always enough food'}",
        ),
        ("completeness", "compares the two times from P0 e^(rt) = limit, one for each limit, and picks the smaller"),
        ("use_of_context", f"calculates both times: {_r1(t_space)} years (space) and {_r1(t_food)} years (food)"),
        ("communication", "states units with every number and gives a one-line conclusion"),
    ]
    traps = [
        (
            "lower_limit_is_not_first",
            "does it compare the times, and not just name the larger or the first-mentioned limit?",
        ),
        ("both_at_once", "does it say both assumptions fail at the same time?"),
        ("units_missing", "does it give years and per year correctly?"),
    ]
    mark = short_marking(reference=ref, steps=steps, traps=traps)
    return finish(
        p,
        f"The {first} limit is reached first, after {_r1(t1)} years; the other limit is reached after {_r1(max(t_space, t_food))} years",
        mark,
    )


def _exam_free_fall_ground(rng) -> dict:
    for _ in attempts(500, "a fall with a ground and a terminal speed"):
        vt = rng.choice((40, 45, 50, 55, 60))
        h = rng.choice((200, 300, 500, 800, 1000, 1500, 2000))
        g = 9.8
        t_term = vt / g
        t_ground = math.sqrt(2 * h / g)
        if _safe(t_term) and _safe(t_ground) and abs(t_term - t_ground) > 0.5:
            break
    first_is_term = t_term < t_ground
    p = Prompt()
    p.t("A model of a stone dropped from a cliff is ").m(Fm("\\frac{dv}{dt}=g,\\quad v(0)=0", "dv/dt = g, v(0) = 0")).t(
        f". The numbers are g = 9.8 m/s^2 and the cliff is {h} m high. The model assumes that (A1: air resistance is negligible, so the only force is gravity; A2: the stone keeps falling for as long as the model is used). "
    )
    p.t(
        f"A stone of this kind has a terminal speed of {vt} m/s in air. Say which of the two assumptions fails first, and at what time to 1 decimal place with units, by comparing the time to reach the terminal speed with the time to reach the ground. Name each variable and parameter with its unit. "
    )
    p.t(rng.choice(SHORT_TAILS))
    ref = [
        "Variables: v (downward speed) in m/s; t (time) in seconds. Parameter: g in m/s^2.",
        f"The model reaches {vt} m/s when 9.8t = {vt}, at t = {_r1(t_term)} seconds. It reaches the ground when (1/2)(9.8)t^2 = {h}, at t = {_r1(t_ground)} seconds.",
        (
            f"The terminal speed comes first ({_r1(t_term)} < {_r1(t_ground)}), so the model first breaks A1 (air resistance is negligible) at t = {_r1(t_term)} seconds."
            if first_is_term
            else f"The ground comes first ({_r1(t_ground)} < {_r1(t_term)}), so the stone lands before air resistance matters in the model: the first assumption to fail is A2, when it hits the ground at t = {_r1(t_ground)} seconds, and A1 is not violated before then."
        ),
    ]
    steps = [
        ("correct_idea", "names v in m/s, t in seconds, g in m/s^2"),
        (
            "correct_idea",
            f"decides that {'A1 (air resistance) fails first' if first_is_term else 'A2 (the stone stops at the ground) happens first, so A1 is not contradicted in the fall'}, at t = {_r1(min(t_term, t_ground))} seconds",
        ),
        ("completeness", "compares the two times, one from v = gt and one from the distance (1/2) g t^2"),
        (
            "use_of_context",
            f"calculates {_r1(t_term)} seconds for the terminal speed and {_r1(t_ground)} seconds for the fall",
        ),
        ("communication", "gives units with each number and a clear conclusion"),
    ]
    traps = [
        ("ignores_the_ground", "does it use only the terminal speed and forget the stone reaches the ground?"),
        ("distance_formula_wrong", "does it use distance = (1/2) g t^2 and not g t^2 or v t?"),
        ("units_missing", "are m/s and seconds stated?"),
    ]
    mark = short_marking(reference=ref, steps=steps, traps=traps)
    verdict = (
        f"The air resistance assumption fails first, at {_r1(t_term)} seconds (the ground is reached at {_r1(t_ground)} seconds)"
        if first_is_term
        else f"The stone hits the ground first, at {_r1(t_ground)} seconds, before reaching terminal speed at {_r1(t_term)} seconds"
    )
    return finish(p, verdict, mark)


def _exam(rng) -> dict:
    return rng.choice((_exam_two_limits_population, _exam_free_fall_ground))(rng)


FAMILIES = {"easy": [_easy], "medium": [_medium], "hard": [_hard], "exam": [_exam]}


@generator("model_assumption_audit", level=17, presets=_PRESETS)
def model_assumption_audit(rng, variant="easy"):
    """`ucv1-model-assumption-audit`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
