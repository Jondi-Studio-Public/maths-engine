"""Two text-only, `answer: mcq` skills about first-order ODEs.

``senior-examine-slope-direction-or-gradient-fields-of-a`` (topic ``p13b_slope_field_examine``, ACMSM131): the
skill name is about EXAMINING a slope/direction field's properties. This skill carries no visual capability
gate (``capability_gate: []``), so every question is TEXT ONLY -- substitution/reasoning about ``dy/dx = f(x,
y)``, never a rendered diagram. Four variants, one per difficulty tier, all rendered as `answer: mcq`:

- easy/medium: evaluate ``f(x0, y0)`` exactly; the 4 options are the correct
  integer value plus misconception values (swapped coordinates, a dropped
  sign, a zeroed-out coordinate).
- hard: a table of 4 named points, each with a WORD-only stated sign
  ("positive"/"negative"); exactly one point's stated sign disagrees with
  the true sign of ``f`` there. The 4 options are the point labels.
- exam: 2-3 word-only qualitative features of a slope field; the 4 options
  are candidate ODE forms ``dy/dx = ...``; exactly one form satisfies every
  stated feature, verified over an explicit integer grid before the draw is
  ever accepted.

``senior-formulate-differential-equations-including-the-logistic-equation-that`` (topic ``p13b_formulate_de``,
ACMSM132): the descriptor verb is "formulate", not "solve" -- this skill NEVER asks for a solved ``y(t)``, only
for which equation FORM models a verbally-described rate relationship. Boundary against
``solve-ode-exponential`` (which gives you the DE and asks you to solve it for ``y(x)``): every rendered
question here ends "...model this situation?" with 4 equation-shaped mcq options, never "Find y(...)". Four
scenarios, one per tier (proportional growth/decay, Newton's cooling, logistic growth, combined external-input
+ proportional decay). Every scenario STATES THE PROPORTIONALITY CONSTANT IS POSITIVE, in words, and names the
physical direction of change in words. Concretely:

- easy: "the rate ... is proportional to the amount of Q present, with a
  positive proportionality constant k" -- key ``kQ`` for an increasing
  context, ``-kQ`` for a decreasing one (the sign must track the stated
  direction; a single shared key regardless of direction was the same class
  of bug as the medium tier, just less visible).
- medium (Newton's cooling): states the object "cools toward the ambient
  ... temperature ... when Q is above it, and warms toward it when Q is
  below it" with a positive constant -- key ``k(A - Q)``, the only form
  that is negative when Q > A and positive when Q < A under a positive k.
- hard (logistic): states growth is "toward the carrying capacity" with a
  positive constant -- key ``kP(N - P)`` (unchanged: this tier was already
  physically correct, positive strictly inside (0, N) and zero at the
  boundary P = N).
- exam: "receives a constant input" / "loses/shrinks ... proportional to Q"
  already fix both terms' directions in ordinary English regardless of any
  sign assumption; the positive-constant wording is added here too for
  consistency across all four tiers, and the build-time invariant confirms
  no second option becomes defensible under it.

Every scenario's build-time invariant confirms EVERY decoy is wrong under the stated positive-constant
convention, not just the previously-checked ones. Letters for the quantity/rate/time symbols are drawn fresh
each seed from a pool so the correct option cannot be spotted by symbol-matching.
"""

from __future__ import annotations

import sympy

from ..core import check, generator, phrase, pick_distractors

# ===========================================================================
# Skill A: senior-examine-slope-direction-or-gradient-fields-of-a
# ===========================================================================

_X, _Y = sympy.symbols("x y")

# Each entry: (display text, sympy-parseable text, f(x, y) -> int).
# The sympy text differs from the display text only in using explicit '*'
# and '**' (never shown to a learner) -- an independent parse of the SAME
# parameters, built from a separate string, evaluated by .subs() rather
# than reusing the python lambda's own arithmetic route.
EASY_FORMS = [
    ("x + y", "x + y", lambda x, y: x + y),
    ("x - y", "x - y", lambda x, y: x - y),
    ("2x - y", "2*x - y", lambda x, y: 2 * x - y),
    ("x - 2y", "x - 2*y", lambda x, y: x - 2 * y),
    ("y - x", "y - x", lambda x, y: y - x),
    ("2y - x", "2*y - x", lambda x, y: 2 * y - x),
]

MEDIUM_FORMS = [
    ("xy", "x*y", lambda x, y: x * y),
    ("x^2 - y", "x**2 - y", lambda x, y: x**2 - y),
    ("y - x^2", "y - x**2", lambda x, y: y - x**2),
    ("xy - x", "x*y - x", lambda x, y: x * y - x),
    ("2xy", "2*x*y", lambda x, y: 2 * x * y),
    ("y^2 - x", "y**2 - x", lambda x, y: y**2 - x),
]

CLASSIFY_FORMS = EASY_FORMS + MEDIUM_FORMS


def _independent_eval(sympy_text: str, x0: int, y0: int) -> int:
    """Re-derive f(x0, y0) from a SEPARATE parse of the same sympy text,
    never reusing the generator's own python lambda arithmetic."""
    expr = sympy.sympify(sympy_text, locals={"x": _X, "y": _Y})
    return int(expr.subs({_X: x0, _Y: y0}))


_VALUE_TEMPLATES = (
    "A curve satisfies dy/dx = {form}. Find the slope of the direction field at the point ({x0}, {y0}).",
    "For the differential equation dy/dx = {form}, what is the slope shown by the direction field at ({x0}, {y0})?",
    "The direction field of dy/dx = {form} is drawn without labels. Determine the slope value at the point ({x0}, {y0}).",
)


def _slope_value_question(rng, forms):
    for _ in range(30):
        form_text, sympy_text, f = rng.choice(forms)
        x0 = rng.randint(-6, 6)
        y0 = rng.randint(-6, 6)
        answer = f(x0, y0)
        independent = _independent_eval(sympy_text, x0, y0)
        check(answer == independent, "slope evaluation disagrees with the independent sympy parse")
        candidates = [
            f(y0, x0),  # swapped coordinates
            -answer,  # dropped sign
            f(-x0, y0),  # sign-dropped x0
            f(x0, -y0),  # sign-dropped y0
            f(0, y0),  # x substituted with 0
            f(x0, 0),  # y substituted with 0
        ]
        distractors = pick_distractors(str(answer), [str(c) for c in candidates], n=3)
        if len(distractors) == 3:
            break
    else:
        raise AssertionError("could not find 3 distinct misconception values for a slope-value question")

    text = phrase(rng, _VALUE_TEMPLATES, form=form_text, x0=x0, y0=y0)
    return dict(
        text=text,
        answer=str(answer),
        answer_text=str(answer),
        working=[
            f"Substitute x = {x0}, y = {y0} into dy/dx = {form_text}.",
            f"dy/dx = {answer} at this point.",
        ],
        distractors=distractors,
    )


_CLASSIFY_TEMPLATES = (
    "A slope field satisfies dy/dx = {form}. Four points and the slope sign observed at each are listed: {table}. "
    "Which point's stated slope sign is NOT consistent with the equation?",
    "For dy/dx = {form}, a student records the slope's sign (never its value) at four points: {table}. "
    "Exactly one recorded sign is wrong. Which point is it?",
)


def _slope_classify_question(rng):
    form_text, sympy_text, f = rng.choice(CLASSIFY_FORMS)
    labels = ["A", "B", "C", "D"]
    points = []
    seen_xy = set()
    while len(points) < 4:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        if (x, y) in seen_xy:
            continue
        value = f(x, y)
        if value == 0:
            continue
        seen_xy.add((x, y))
        points.append((x, y, value))

    true_signs = ["positive" if v > 0 else "negative" for (_, _, v) in points]
    wrong_idx = rng.randrange(4)
    stated_signs = list(true_signs)
    stated_signs[wrong_idx] = "negative" if true_signs[wrong_idx] == "positive" else "positive"

    disagreements = [i for i in range(4) if stated_signs[i] != true_signs[i]]
    check(disagreements == [wrong_idx], "exactly one stated point sign must disagree with the true sign")

    for x, y, v in points:
        independent = _independent_eval(sympy_text, x, y)
        check(v == independent, "classification point disagrees with the independent sympy parse")

    table = "; ".join(
        f"Point {labels[i]} = ({points[i][0]}, {points[i][1]}), slope stated as {stated_signs[i]}" for i in range(4)
    )
    text = phrase(rng, _CLASSIFY_TEMPLATES, form=form_text, table=table)
    answer_text = f"Point {labels[wrong_idx]}"
    distractors = [f"Point {l}" for l in labels if l != labels[wrong_idx]]
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        working=[
            f"Evaluate dy/dx = {form_text} at every listed point.",
            f"Point {labels[wrong_idx]} is the only one whose stated sign does not match the evaluated sign.",
        ],
        distractors=distractors,
    )


# --- exam tier: match a verbal feature description to one of a fixed ODE pool

_GRID = tuple(range(-3, 4))

MATCH_FORM_POOL = [
    ("y", lambda x, y: y),
    ("-y", lambda x, y: -y),
    ("x", lambda x, y: x),
    ("-x", lambda x, y: -x),
    ("y^2", lambda x, y: y * y),
    ("x - y", lambda x, y: x - y),
    ("x + y", lambda x, y: x + y),
    ("xy", lambda x, y: x * y),
    # These 3 new forms, plus the FEATURES entries added below, both widen the pool AND give x - y
    # and x + y a distinguishing feature pair each for the first time.
    ("y - 2", lambda x, y: y - 2),
    ("y*(3 - y)", lambda x, y: y * (3 - y)),
    ("-xy", lambda x, y: -x * y),
]

FEATURES = [
    ("the slope is zero whenever y = 0", lambda f: all(f(x, 0) == 0 for x in _GRID)),
    ("the slope is zero whenever x = 0", lambda f: all(f(0, y) == 0 for y in _GRID)),
    (
        "the slope is always positive when y is positive",
        lambda f: all(f(x, y) > 0 for x in _GRID for y in _GRID if y > 0),
    ),
    (
        "the slope is always negative when y is negative",
        lambda f: all(f(x, y) < 0 for x in _GRID for y in _GRID if y < 0),
    ),
    (
        "the slope is always positive when x is positive",
        lambda f: all(f(x, y) > 0 for x in _GRID for y in _GRID if x > 0),
    ),
    (
        "the slope is always negative when x is negative",
        lambda f: all(f(x, y) < 0 for x in _GRID for y in _GRID if x < 0),
    ),
    (
        "the slope never depends on x",
        lambda f: all(f(x1, y) == f(x2, y) for y in _GRID for x1 in _GRID for x2 in _GRID),
    ),
    ("the slope is never negative", lambda f: all(f(x, y) >= 0 for x in _GRID for y in _GRID)),
    ("the slope is zero whenever x equals y", lambda f: all(f(x, x) == 0 for x in _GRID)),
    ("the slope is zero whenever y equals negative x", lambda f: all(f(x, -x) == 0 for x in _GRID)),
    (
        "the slope is always positive when x is greater than y",
        lambda f: all(f(x, y) > 0 for x in _GRID for y in _GRID if x > y),
    ),
    (
        "the slope is always positive when y is greater than negative x",
        lambda f: all(f(x, y) > 0 for x in _GRID for y in _GRID if y > -x),
    ),
    ("the slope is zero when y = 2", lambda f: all(f(x, 2) == 0 for x in _GRID)),
    (
        "the slope is always negative when y is less than 2",
        lambda f: all(f(x, y) < 0 for x in _GRID for y in _GRID if y < 2),
    ),
    (
        "the slope is zero when y = 0 or y = 3",
        lambda f: all(f(x, 0) == 0 for x in _GRID) and all(f(x, 3) == 0 for x in _GRID),
    ),
    (
        "the slope is positive when x and y have opposite signs, and negative when x and y have the same sign",
        lambda f: (
            all(f(x, y) > 0 for x in _GRID for y in _GRID if x * y < 0)
            and all(f(x, y) < 0 for x in _GRID for y in _GRID if x * y > 0)
        ),
    ),
]

# Precomputed once at import: form name -> frozenset of true feature indices.
_FORM_FEATURES = {
    name: frozenset(i for i, (_, pred) in enumerate(FEATURES) if pred(fn)) for name, fn in MATCH_FORM_POOL
}

# Precomputed once at import: for every pool form, every k-in-{2,3} subset of its own true features
# that is NOT also a subset of any OTHER pool form's true features -- i.e. every feature-clause set
# that pins this form uniquely within the whole pool. A form with no such subset (e.g. "-y", whose
# only 2-feature combination is shared by "y" and "y^2") can never be the target; drawing it below
# always fails immediately and re-draws, rather than looping through doomed subset attempts, so the
# forms that CAN be a target are chosen uniformly.
from itertools import combinations as _combinations

_VALID_TARGET_SUBSETS: dict[str, list[frozenset]] = {}
for _name, _feats in _FORM_FEATURES.items():
    _feats_sorted = sorted(_feats)
    _subsets = []
    for _k in (2, 3):
        if _k > len(_feats_sorted):
            continue
        for _combo in _combinations(_feats_sorted, _k):
            _s = frozenset(_combo)
            if not any(_s <= _FORM_FEATURES[_other] for _other in _FORM_FEATURES if _other != _name):
                _subsets.append(_s)
    if _subsets:
        _VALID_TARGET_SUBSETS[_name] = _subsets

TARGET_NAMES = sorted(_VALID_TARGET_SUBSETS)

_MATCH_TEMPLATES = (
    "Consider a slope field with the following features: {clauses}. Which of these differential "
    "equations is consistent with ALL the stated features?",
    "A direction field is known to have these properties: {clauses}. Which equation below matches every one of them?",
)


def _slope_match_question(rng):
    all_names = [name for name, _ in MATCH_FORM_POOL]
    for _ in range(50):
        target = rng.choice(all_names)
        subsets = _VALID_TARGET_SUBSETS.get(target)
        if not subsets:
            continue
        chosen = rng.choice(subsets)
        others = [name for name in all_names if name != target]
        for d in others:
            check(not (chosen <= _FORM_FEATURES[d]), f"decoy {d} satisfies every stated feature")
        decoys = rng.sample(others, 3)
        break
    else:
        raise AssertionError("could not find a unique feature subset for the exam-tier match question")

    clauses = "; ".join(f"({i + 1}) {FEATURES[idx][0]}" for i, idx in enumerate(sorted(chosen)))
    text = phrase(rng, _MATCH_TEMPLATES, clauses=clauses)
    answer_text = f"dy/dx = {target}"
    distractors = [f"dy/dx = {d}" for d in decoys]
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        working=[
            "Test each candidate equation against the stated features on a grid of sample points.",
            f"Only dy/dx = {target} satisfies every stated feature.",
        ],
        distractors=distractors,
    )


@generator(
    "p13b_slope_field_examine",
    level=18,
    presets={
        "easy": dict(variant="value_easy"),
        "medium": dict(variant="value_medium"),
        "hard": dict(variant="classify_point"),
        "exam": dict(variant="match_ode"),
    },
)
def p13b_slope_field_examine(rng, variant="value_easy"):
    """Text-only slope/direction-field questions (no drawn diagram)."""
    if variant == "value_easy":
        return _slope_value_question(rng, EASY_FORMS)
    if variant == "value_medium":
        return _slope_value_question(rng, MEDIUM_FORMS)
    if variant == "classify_point":
        return _slope_classify_question(rng)
    if variant == "match_ode":
        return _slope_match_question(rng)
    raise ValueError(f"unknown variant {variant!r}")


# ===========================================================================
# Skill B: senior-formulate-differential-equations-including-the-logistic-...
# ===========================================================================

QUANTITY_POOL = ["Q", "N", "M", "V", "W", "S"]
CAPACITY_POOL = ["N", "K", "L", "M", "C"]
AMBIENT_POOL = ["A", "R", "E", "B"]
RATE_POOL = ["k", "c", "b"]
TIME_POOL = ["t", "s"]

EASY_CONTEXTS = [
    ("a colony of bacteria", "increases"),
    ("a radioactive sample", "decreases"),
    ("an investment balance", "increases"),
    ("the concentration of a drug in the bloodstream", "decreases"),
]


def _draw_distinct(rng, *pools):
    """One letter from each pool, retrying whenever two pools would draw the
    same letter, ignoring case: the MCQ text matcher casefolds, so options that differ only by case
    (``dS/ds = bS`` against ``dS/ds = bs``) would reject the key (small pools, so a handful of retries
    always succeeds)."""
    for _ in range(50):
        chosen = [rng.choice(pool) for pool in pools]
        if len({c.lower() for c in chosen}) == len(chosen):
            return chosen
    raise AssertionError("could not draw distinct symbols")


def _formulate_easy(rng):
    q, k, t = _draw_distinct(rng, QUANTITY_POOL, RATE_POOL, TIME_POOL)
    context, direction = rng.choice(EASY_CONTEXTS)
    sign = "" if direction == "increases" else "-"
    key = f"d{q}/d{t} = {sign}{k}{q}"
    decoys_raw = [f"d{q}/d{t} = {k}{t}", f"d{q}/d{t} = {k}", f"d{q}/d{t} = {k}{q}^2"]

    # Build-time invariant: under a POSITIVE rate constant, the defining properties of "proportional
    # to Q itself, {direction}" are (a) the rate's sign at a positive Q matches the stated
    # direction, and (b) doubling Q doubles the rate -- (b) alone would NOT catch a wrong-sign key
    # (+kQ and -kQ both double), and (a) alone does not uniquely pin the key for an "increases"
    # context (kt, k and kQ^2 are all positive too at a single positive-Q, positive-k probe) -- only
    # the two together are unique.
    k_val, t_val = 2, 5
    q1, q2 = 3, 6
    direction_sign = 1 if direction == "increases" else -1
    candidates = {
        "key": lambda Qv: direction_sign * k_val * Qv,
        "decoy_time": lambda Qv: k_val * t_val,
        "decoy_const": lambda Qv: k_val,
        "decoy_square": lambda Qv: k_val * Qv * Qv,
    }
    satisfying = [
        name
        for name, fn in candidates.items()
        if fn(q1) != 0 and fn(q2) / fn(q1) == 2 and ((fn(q1) > 0) if direction == "increases" else (fn(q1) < 0))
    ]
    check(satisfying == ["key"], f"easy-tier proportionality invariant not unique: {satisfying}")

    text = (
        f"Let {q} denote {context}, changing over time {t}, with a positive proportionality "
        f"constant {k}. The rate at which {q} {direction} is proportional to the amount of {q} "
        f"present. Which differential equation models this situation?"
    )
    distractors = pick_distractors(key, decoys_raw, n=3)
    return text, key, distractors


def _formulate_medium(rng):
    q, a, k, t = _draw_distinct(rng, QUANTITY_POOL, AMBIENT_POOL, RATE_POOL, TIME_POOL)
    key = f"d{q}/d{t} = {k}({a} - {q})"
    decoys_raw = [f"d{q}/d{t} = {k}{q}", f"d{q}/d{t} = {k}({q} - {a})", f"d{q}/d{t} = {k}{q}{a}"]

    # Build-time invariant: under a positive k, the defining property of
    # "cools toward ambient above it, warms toward ambient below it" is that
    # the rate's sign is NEGATIVE when Q > A and POSITIVE when Q < A. Both
    # probes are required together -- at Q > A alone, the drop-the-ambient
    # decoy (kQ) is positive (not negative) for this exact probe pair, so it
    # is already excluded by the first probe, but checking both keeps the
    # invariant robust to the exact numbers chosen.
    candidates = {
        "key": lambda qv, av: k_val * (av - qv),
        "decoy_drop_a": lambda qv, av: k_val * qv,
        "decoy_reverse": lambda qv, av: k_val * (qv - av),
        "decoy_product": lambda qv, av: k_val * qv * av,
    }
    k_val = 2
    satisfying = [name for name, fn in candidates.items() if fn(10, 4) < 0 and fn(4, 10) > 0]
    check(satisfying == ["key"], f"medium-tier sign invariant not unique: {satisfying}")

    text = (
        f"An object's temperature, {q}, changes over time {t}. The rate of change of {q} is "
        f"proportional to ({a} - {q}), with a positive proportionality constant {k}: the object "
        f"cools toward the ambient (surrounding) temperature {a} when {q} is above it, and warms "
        f"toward {a} when {q} is below it. Which differential equation models this situation?"
    )
    distractors = pick_distractors(key, decoys_raw, n=3)
    return text, key, distractors


def _formulate_hard(rng):
    p, n, k, t = _draw_distinct(rng, QUANTITY_POOL, CAPACITY_POOL, RATE_POOL, TIME_POOL)
    key = f"d{p}/d{t} = {k}{p}({n} - {p})"
    decoys_raw = [
        f"d{p}/d{t} = {k}{p}({p} - {n})",
        f"d{p}/d{t} = {k}({p} + ({n} - {p}))",
        f"d{p}/d{t} = {k}{p}",
    ]

    # Build-time invariant: the logistic form is the only candidate that is
    # positive strictly inside (0, N) and non-positive outside it.
    k_val, n_val = 2, 10
    candidates = {
        "key": lambda pv: k_val * pv * (n_val - pv),
        "decoy_sign": lambda pv: k_val * pv * (pv - n_val),
        "decoy_add": lambda pv: k_val * (pv + (n_val - pv)),
        "decoy_no_capacity": lambda pv: k_val * pv,
    }
    p_in, p_out = 4, 15
    satisfying = [name for name, fn in candidates.items() if fn(p_in) > 0 and fn(p_out) <= 0]
    check(satisfying == ["key"], f"hard-tier logistic invariant not unique: {satisfying}")
    # "adds instead of multiplies" collapses to a constant -- confirm that
    # nonsense explicitly.
    check(
        candidates["decoy_add"](2) == candidates["decoy_add"](8) == k_val * n_val,
        "decoy_add should reduce to a p-independent constant",
    )

    text = (
        f"A population, {p}, grows over time {t} within an environment of carrying capacity {n}, "
        f"always moving toward that carrying capacity. Let ({n} - {p}) be the population's remaining "
        f"capacity. The rate of growth of {p} is proportional to the product of {p} and its remaining "
        f"capacity, ({n} - {p}), with a positive proportionality constant {k}. "
        f"Which differential equation models this situation?"
    )
    distractors = pick_distractors(key, decoys_raw, n=3)
    return text, key, distractors


_EXAM_TEMPLATES = (
    "A quantity {q} is added to a system at a constant rate {r}, while also decaying at a rate "
    "proportional to the amount {q} present, with a positive proportionality constant {k}, measured "
    "over time {t}. Which differential equation models this situation?",
    "A store of {q} receives a constant input of {r} per unit time, and simultaneously loses "
    "{q} at a rate proportional to the amount {q} present (a positive proportionality constant {k}), "
    "as a function of time {t}. Which differential equation models this situation?",
    "The quantity {q}, measured over time {t}, grows by a fixed constant amount {r} per unit time "
    "and shrinks at a rate proportional to {q} itself, with a positive proportionality constant {k}. "
    "Which differential equation models this situation?",
)


def _formulate_exam(rng):
    q, k, t = _draw_distinct(rng, QUANTITY_POOL, RATE_POOL, TIME_POOL)
    r = "r" if "r" not in (q, k, t) else "m"
    key = f"d{q}/d{t} = {r} - {k}{q}"
    decoys_raw = [
        f"d{q}/d{t} = {r} + {k}{q}",
        f"d{q}/d{t} = {k}({r} - {q})",
        f"d{q}/d{t} = {r}{k}{q}",
    ]

    # Build-time invariant: the key is the only candidate whose value at
    # Q=0 equals the stated constant input rate r AND that is strictly
    # decreasing as Q grows (an external input plus a proportional loss).
    r_val, k_val = 3, 2
    candidates = {
        "key": lambda qv: r_val - k_val * qv,
        "decoy_sign": lambda qv: r_val + k_val * qv,
        "decoy_scope": lambda qv: k_val * (r_val - qv),
        "decoy_mult": lambda qv: r_val * k_val * qv,
    }
    q1, q2 = 1, 5
    satisfying = [name for name, fn in candidates.items() if fn(0) == r_val and fn(q2) < fn(q1)]
    check(satisfying == ["key"], f"exam-tier combined-rate invariant not unique: {satisfying}")

    text = phrase(
        rng,
        _EXAM_TEMPLATES,
        q=q,
        r=r,
        k=k,
        t=t,
    )
    distractors = pick_distractors(key, decoys_raw, n=3)
    return text, key, distractors


_FORMULATE_FNS = {
    "easy": _formulate_easy,
    "medium": _formulate_medium,
    "hard": _formulate_hard,
    "exam": _formulate_exam,
}


@generator(
    "p13b_formulate_de",
    level=18,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p13b_formulate_de(rng, variant="easy"):
    """Pick which differential-equation FORM models a stated rate
    relationship -- never asks the learner to solve it."""
    fn = _FORMULATE_FNS.get(variant)
    if fn is None:
        raise ValueError(f"unknown variant {variant!r}")
    text, key, distractors = fn(rng)
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        working=[
            "Translate each clause of the verbal description into a symbolic term.",
            f"The equation that matches every stated clause is {key}.",
        ],
        distractors=distractors,
    )
