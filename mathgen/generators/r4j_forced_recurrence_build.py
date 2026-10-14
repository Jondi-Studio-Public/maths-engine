"""Generates `uc2-nonhomogeneous-recurrence-polynomial-or-exponential-forcing` questions (easy, medium, hard) and the exam
chain `exd_x_r4j_forced_recurrence`.

The answer is a closed form for a(n) (``expression``) of a linear recurrence with constant coefficients and a forcing term
that is a polynomial in n or an exponential c·s^n. A constant forcing term is the job of a different skill; here the forcing
varies with n. No characteristic equation, root or trial form is supplied in an easy, medium or hard stem.

* easy: first order, a(n) = r a(n-1) + c s^n with s different from r. The particular solution is a multiple of s^n.
* medium: first order, either a linear forcing term (particular solution An + B, r not 1) or resonance, s = r (particular
  solution c n r^n).
* hard: second order with distinct integer roots, one of: a linear forcing term; an exponential forcing term equal to one
  root (resonant); a linear forcing term with 1 as a root (particular solution n(An + B)).
* exam (``exd_x_r4j_forced_recurrence``): second order, with either a linear forcing term or a resonant exponential one.
  (a) the roots of the characteristic equation, (b) the particular solution, (c) the closed form, (d) a later term, which
  follows through from (c).

A closed form is a list of terms ``(coefficient, base, power of n)`` with exact `Fraction` arithmetic: the particular
solution comes from solving the undetermined-coefficient equations at sample points, and every closed form is checked
against terms 0 to 11 produced by running the recurrence directly.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, pick_distractors
from ._r4j_kit import keep_distinct, say

_TAGS = {
    "forcing_ignored": "recurrence-forcing-ignored",
    "sign": "recurrence-particular-sign",
    "not_fitted": "recurrence-particular-not-fitted",
    "forcing_as_particular": "recurrence-forcing-used-as-particular",
    "power_dropped": "recurrence-resonance-power-dropped",
    "root_sign": "recurrence-root-sign",
    "coefficients_as_roots": "recurrence-coefficients-as-roots",
    "index": "recurrence-index-off-by-one",
    "swap": "recurrence-coefficients-swapped",
    "coefficient": "recurrence-particular-coefficient-slip",
}

_SUB = str.maketrans("0123456789n-", "₀₁₂₃₄₅₆₇₈₉ₙ₋")


# -- closed forms as exact term lists ---------------------------------------------------------------------------------


def _value(terms, n):
    """The closed form at n: terms are ``(coefficient, base, power of n)``."""
    return sum((Fraction(c) * Fraction(b) ** n * n**k for c, b, k in terms), Fraction(0))


def _coef_text(c):
    c = Fraction(c)
    return str(c.numerator) if c.denominator == 1 else f"({c.numerator}/{c.denominator})"


def _term_body(base, k):
    factors = []
    if k == 1:
        factors.append("n")
    elif k >= 2:
        factors.append(f"n^{k}")
    if base != 1:
        factors.append(f"({base})^n" if base < 0 else f"{base}^n")
    return "*".join(factors)


def _expr_text(terms):
    """`3*2^n - n*2^n + 1`: terms in a fixed order, a true unit coefficient left off, a plain ASCII minus."""
    ordered = sorted((t for t in terms if t[0] != 0), key=lambda t: (-t[1] if t[1] != 1 else 1000, -t[2]))
    out = ""
    for c, b, k in ordered:
        c = Fraction(c)
        body = _term_body(b, k)
        mag = abs(c)
        piece = body if (mag == 1 and body) else (f"{_coef_text(mag)}*{body}" if body else _coef_text(mag))
        out += (("-" if c < 0 else "") + piece) if not out else ((" - " if c < 0 else " + ") + piece)
    return out or "0"


def _merge(terms):
    acc = {}
    for c, b, k in terms:
        acc[(b, k)] = acc.get((b, k), 0) + Fraction(c)
    return [(c, b, k) for (b, k), c in acc.items() if c != 0]


def _solve(matrix, rhs):
    """Solve a small square linear system exactly, or None when it is singular."""
    n = len(matrix)
    rows = [[Fraction(v) for v in row] + [Fraction(r)] for row, r in zip(matrix, rhs)]
    for col in range(n):
        pivot = next((i for i in range(col, n) if rows[i][col] != 0), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        rows[col] = [v / rows[col][col] for v in rows[col]]
        for i in range(n):
            if i != col and rows[i][col] != 0:
                f = rows[i][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[col])]
    return [rows[i][n] for i in range(n)]


def _residual(basis, p, q, n):
    """The recurrence operator y(n) - p y(n-1) - q y(n-2) applied to the basis term (base, power) at n."""
    base, k = basis

    def y(m):
        return Fraction(base) ** m * Fraction(m) ** k

    return y(n) - p * y(n - 1) - q * y(n - 2)


def _particular(p, q, forcing, base, top_power, low_power):
    """Terms ``(c, base, k)`` for k from low_power to top_power that satisfy y(n) = p y(n-1) + q y(n-2) + forcing(n), or
    None. `forcing` is a function of n."""
    powers = list(range(low_power, top_power + 1))
    pts = [3 + i for i in range(len(powers))]
    matrix = [[_residual((base, k), p, q, n) for k in powers] for n in pts]
    sol = _solve(matrix, [forcing(n) for n in pts])
    if sol is None:
        return None
    return [(c, base, k) for c, k in zip(sol, powers)]


def _run(p, q, forcing, a0, a1, count=12):
    out = [Fraction(a0), Fraction(a1)]
    for n in range(2, count):
        out.append(p * out[-1] + q * out[-2] + forcing(n))
    return out


def _fit(bases, target, n_terms):
    """Coefficients of the homogeneous bases ``(base, power)`` so that the sum matches `target[n]` for n = 0.. ."""
    matrix = [[Fraction(b) ** n * Fraction(n) ** k for b, k in bases] for n in range(len(bases))]
    return _solve(matrix, [target[n] for n in range(len(bases))])


def _plus(c):
    return f" + {c}" if c > 0 else f" − {-c}"


def _sub(i):
    return str(i).translate(_SUB)


def _rec_text(p, q, f_plain):
    parts = []
    for c, lag in ((p, "n-1"), (q, "n-2")):
        if c == 0:
            continue
        body = f"a{_sub(lag)}" if abs(c) == 1 else f"{abs(c)}a{_sub(lag)}"
        parts.append(("−" if c < 0 else "") + body if not parts else (" − " if c < 0 else " + ") + body)
    tail = " − " + f_plain[1:] if f_plain.startswith("−") else " + " + f_plain
    return "aₙ = " + "".join(parts) + tail


def _rec_tex(p, q, f_tex):
    out = "a_{n}="
    first = True
    for c, lag in ((p, "n-1"), (q, "n-2")):
        if c == 0:
            continue
        body = f"a_{{{lag}}}" if abs(c) == 1 else f"{abs(c)}a_{{{lag}}}"
        out += (("-" if c < 0 else "") + body) if first else (("-" if c < 0 else "+") + body)
        first = False
    return out + ("" if f_tex.startswith("-") else "+") + f_tex


def _lin_plain(alpha, beta):
    """`4n + 1`, `−3n − 2`, `n`."""
    out = ("−" if alpha < 0 else "") + ("n" if abs(alpha) == 1 else f"{abs(alpha)}n")
    if beta:
        out += (" − " if beta < 0 else " + ") + str(abs(beta))
    return out


def _lin_tex(alpha, beta):
    out = ("-" if alpha < 0 else "") + ("n" if abs(alpha) == 1 else f"{abs(alpha)}n")
    if beta:
        out += ("-" if beta < 0 else "+") + str(abs(beta))
    return out


def _exp_plain(c, s):
    base = f"({s})" if s < 0 else str(s)
    return (f"{c}·" if c != 1 else "") + f"{base}^n"


def _exp_tex(c, s):
    base = f"({s})" if s < 0 else str(s)
    return (f"{c}\\cdot " if c != 1 else "") + f"{base}^{{n}}"


# -- instances -----------------------------------------------------------------------------------------------------------


def _integer_terms(terms):
    return all(Fraction(c).denominator == 1 for c, _b, _k in terms)


def _build(rng, kind):
    """A random instance of `kind`; a dict with the recurrence, its forcing, the particular solution and closed form."""
    for _ in attempts(3000, f"an integer instance of {kind}"):
        if kind == "exp1":  # first order, s != r
            r, s = rng.sample([2, 3, 4, 5, -2, -3], 2)
            c = rng.randint(1, 5)
            p, q = r, 0
            forcing = lambda n, c=c, s=s: Fraction(c) * Fraction(s) ** n  # noqa: E731
            part = _particular(p, q, forcing, s, 0, 0)
            hom = [(r, 0)]
            f_plain, f_tex = _exp_plain(c, s), _exp_tex(c, s)
        elif kind == "res1":  # first order, s == r
            r = rng.choice([2, 3, 4, 5, -2, -3])
            c = rng.randint(1, 5)
            p, q = r, 0
            forcing = lambda n, c=c, r=r: Fraction(c) * Fraction(r) ** n  # noqa: E731
            part = _particular(p, q, forcing, r, 1, 1)
            hom = [(r, 0)]
            f_plain, f_tex = _exp_plain(c, r), _exp_tex(c, r)
        elif kind == "poly1":  # first order, linear forcing, r != 1
            r = rng.choice([2, 3, 4, -2, -3, 5])
            alpha, beta = rng.choice([1, 2, 3, 4, -1, -2, -3, 5]), rng.choice([0, 1, 2, 3, -1, -2, -4, 5])
            p, q = r, 0
            forcing = lambda n, a=alpha, b=beta: Fraction(a * n + b)  # noqa: E731
            part = _particular(p, q, forcing, 1, 1, 0)
            hom = [(r, 0)]
            f_plain, f_tex = _lin_plain(alpha, beta), _lin_tex(alpha, beta)
        elif kind == "poly2":  # second order, linear forcing, 1 not a root
            r1, r2 = rng.sample([2, 3, 4, -1, -2, -3], 2)
            alpha, beta = rng.choice([1, 2, 3, 4, -1, -2, -3, 5]), rng.choice([0, 1, 2, 3, -1, -2, -4, 5])
            p, q = r1 + r2, -r1 * r2
            forcing = lambda n, a=alpha, b=beta: Fraction(a * n + b)  # noqa: E731
            part = _particular(p, q, forcing, 1, 1, 0)
            hom = [(r1, 0), (r2, 0)]
            f_plain, f_tex = _lin_plain(alpha, beta), _lin_tex(alpha, beta)
        elif kind == "exp2":  # second order, s equal to one root
            r1, r2 = rng.sample([2, 3, 4, -1, -2, -3], 2)
            c = rng.randint(1, 4)
            p, q = r1 + r2, -r1 * r2
            forcing = lambda n, c=c, s=r1: Fraction(c) * Fraction(s) ** n  # noqa: E731
            part = _particular(p, q, forcing, r1, 1, 1)
            hom = [(r1, 0), (r2, 0)]
            f_plain, f_tex = _exp_plain(c, r1), _exp_tex(c, r1)
        else:
            check(kind == "poly2res", f"unknown kind {kind!r}")  # second order, linear forcing, 1 is a root
            r2 = rng.choice([2, 3, 4, -1, -2, -3])
            alpha, beta = rng.choice([1, 2, 3, 4, -1, -2, -3, 5]), rng.choice([0, 1, 2, 3, -1, -2, -4, 5])
            p, q = 1 + r2, -r2
            forcing = lambda n, a=alpha, b=beta: Fraction(a * n + b)  # noqa: E731
            part = _particular(p, q, forcing, 1, 2, 1)
            hom = [(1, 0), (r2, 0)]
            f_plain, f_tex = _lin_plain(alpha, beta), _lin_tex(alpha, beta)
        if part is None or not _integer_terms(part) or any(c == 0 for c, _b, _k in part[:1]):
            continue
        if len(hom) == 1:
            c1 = rng.choice([-3, -2, -1, 1, 2, 3, 4])
            coefs = [c1]
        else:
            coefs = [rng.choice([-3, -2, -1, 1, 2, 3, 4]), rng.choice([-3, -2, -1, 1, 2, 3])]
        closed = _merge([(c, b, k) for c, (b, k) in zip(coefs, hom)] + part)
        a0, a1 = _value(closed, 0), _value(closed, 1)
        if a0.denominator != 1 or a1.denominator != 1 or a0 == 0 or a1 == 0:
            continue
        if any(abs(c) > 40 for c, _b, _k in closed):
            continue
        seq = _run(p, q, forcing, a0, a1)
        check([_value(closed, n) for n in range(12)] == seq, "closed form does not reproduce the recurrence")
        return dict(
            kind=kind,
            order=1 if q == 0 else 2,
            p=p,
            q=q,
            forcing=forcing,
            f_plain=f_plain,
            f_tex=f_tex,
            hom=hom,
            coefs=coefs,
            part=part,
            closed=closed,
            a0=int(a0),
            a1=int(a1),
            seq=seq,
        )
    raise AssertionError  # pragma: no cover


def _show(text):
    """An answer as it is read in worked steps: a centred dot for the product sign."""
    return text.replace("*", "·")


def _sequence(terms):
    return tuple(_value(terms, n) for n in range(12))


def _wrong_forms(inst):
    """``(terms, tag)`` pairs for the usual wrong closed forms of an instance."""
    hom, part, closed = inst["hom"], inst["part"], inst["closed"]
    target = inst["seq"]
    out = []

    def refit(particular_terms):
        resid = [target[n] - _value(particular_terms, n) for n in range(len(hom))]
        coefs = _fit(hom, resid, len(hom))
        if coefs is None:
            return None
        return _merge([(c, b, k) for c, (b, k) in zip(coefs, hom)] + particular_terms)

    plain_fit = _fit(hom, target, len(hom))
    hom_only = [(c, b, k) for c, (b, k) in zip(plain_fit, hom)] if plain_fit else None
    if hom_only:
        out.append((_merge(hom_only), _TAGS["forcing_ignored"]))
    out.append((refit([(-c, b, k) for c, b, k in part]), _TAGS["sign"]))
    if hom_only:
        out.append((_merge(hom_only + part), _TAGS["not_fitted"]))
    f = inst["forcing"]
    if inst["kind"] in ("exp1", "res1", "exp2"):
        c = f(0)
        s = {"exp1": part[0][1], "res1": part[0][1], "exp2": part[0][1]}[inst["kind"]]
        out.append((refit([(c, s, 0)]), _TAGS["forcing_as_particular"]))
    else:
        beta, alpha = f(0), f(1) - f(0)
        out.append((refit([(alpha, 1, 1), (beta, 1, 0)]), _TAGS["forcing_as_particular"]))
    if inst["kind"] in ("res1", "exp2", "poly2res"):
        out.append((refit([(c, b, k - 1) for c, b, k in part]), _TAGS["power_dropped"]))
    # the leading coefficient of the particular solution slips by one, constants refitted
    c0, b0, k0 = part[-1]
    out.append((refit([*part[:-1], (c0 + 1, b0, k0)]), _TAGS["coefficient"]))
    out.append((refit([(c * 2, b, k) for c, b, k in part]), _TAGS["coefficient"]))
    # the two homogeneous constants swapped
    if len(hom) == 2 and inst["coefs"][0] != inst["coefs"][1]:
        c1, c2 = inst["coefs"]
        out.append((_merge([(c2, hom[0][0], hom[0][1]), (c1, hom[1][0], hom[1][1])] + part), _TAGS["swap"]))
    # a homogeneous root with the wrong sign, constants refitted
    flipped = [(-b, k) for b, k in hom]
    if len(set(flipped)) == len(flipped):
        coefs = _fit(flipped, [target[n] - _value(part, n) for n in range(len(hom))], len(hom))
        if coefs:
            out.append((_merge([(c, b, k) for c, (b, k) in zip(coefs, flipped)] + part), _TAGS["root_sign"]))
    return out


def _distractors(inst):
    cands = []
    for terms, tag in _wrong_forms(inst):
        if terms is None or any(Fraction(c).denominator > 6 for c, _b, _k in terms):
            continue
        cands.append((terms, tag, _expr_text(terms)))
    return [Distractor(d.text, d.misconception, d.text) for d in keep_distinct(inst["closed"], cands, key=_sequence)]


# -- wording -------------------------------------------------------------------------------------------------------------


def _stem(rng, inst, ask=True):
    p, q, a0, a1 = inst["p"], inst["q"], inst["a0"], inst["a1"]
    rec = _rec_text(p, q, inst["f_plain"])
    rec_t = _rec_tex(p, q, inst["f_tex"])
    second = inst["order"] == 2
    start = f"a₀ = {a0} and a₁ = {a1}" if second else f"a₀ = {a0}"
    start_t = f"a_{{0}}={a0},\\ a_{{1}}={a1}" if second else f"a_{{0}}={a0}"
    since = "n ≥ 2" if second else "n ≥ 1"
    form = rng.randrange(3)
    if form == 0:
        return say(
            f"A sequence has {start} and satisfies ",
            (rec, rec_t),
            f" for {since}." + (" Find a closed form for aₙ in terms of n." if ask else ""),
        )
    if form == 1:
        return say(
            "The sequence (aₙ) is defined by ",
            (start, start_t),
            " and ",
            (rec, rec_t),
            f" for {since}." + (" Solve the recurrence: give aₙ as a formula in n." if ask else ""),
        )
    if ask:
        return say("Solve ", (rec, rec_t), f" for {since}, given {start}. Express aₙ in closed form.")
    return say(f"Let {start}, and let ", (rec, rec_t), f" for {since}, so that (aₙ) is a sequence.")


def _single(rng, kind, working_fn):
    inst = _build(rng, kind)
    text, latex = _stem(rng, inst)
    answer = _expr_text(inst["closed"])
    distractors = _distractors(inst)[:5]
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        latex=latex,
        answer_kind="expression",
        answer=answer,
        answer_text=answer,
        working=working_fn(inst, answer),
        distractors=distractors,
    )


def _working(inst, answer):
    kind, part = inst["kind"], inst["part"]
    names = {
        "exp1": "The forcing term is a multiple of s^n and s is not a root, so try a multiple of s^n.",
        "res1": "The forcing term is r^n and r is a root of the homogeneous equation, so try a multiple of n·r^n.",
        "poly1": "The forcing term is linear and 1 is not a root, so try An + B.",
        "poly2": "The forcing term is linear and 1 is not a root, so try An + B.",
        "exp2": "The forcing term is s^n and s is a root of the homogeneous equation, so try a multiple of n·s^n.",
        "poly2res": "The forcing term is linear and 1 is a root, so try n(An + B).",
    }
    return [
        f"Homogeneous part: aₙ = {inst['p']}aₙ₋₁"
        + (f" + ({inst['q']})aₙ₋₂" if inst["q"] else "")
        + f" has the solutions {', '.join(_term_body(b, k) or '1' for b, k in inst['hom'])}.",
        names[kind] + f" Substituting gives the particular solution {_show(_expr_text(part))}.",
        f"Fit the constants to the starting terms: aₙ = {_show(answer)}.",
    ]


def _easy(rng):
    return _single(rng, "exp1", _working)


def _medium(rng):
    return _single(rng, rng.choice(["poly1", "res1"]), _working)


def _hard(rng):
    return _single(rng, rng.choice(["poly2", "exp2", "poly2res"]), _working)


@generator(
    "r4j_forced_recurrence",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4j_forced_recurrence(rng, tier="easy"):
    """A closed form for a recurrence with polynomial or exponential forcing; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------------


@generator("exd_x_r4j_forced_recurrence", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4j_forced_recurrence(rng, tier="exam"):
    """Exam chain: characteristic roots, the particular solution, the closed form, then a later term."""
    check(tier == "exam", f"unknown tier {tier!r}")
    kind = rng.choice(["poly2", "exp2"])
    inst = _build(rng, kind)
    stem, stem_tex = _stem(rng, inst, ask=False)
    p, q, closed = inst["p"], inst["q"], inst["closed"]
    roots = sorted(b for b, _k in inst["hom"])
    answer = _expr_text(closed)
    index = rng.choice([6, 7, 8])
    value = int(_value(closed, index))
    check(value == inst["seq"][index], "later term differs from the recurrence")
    r1, r2 = inst["hom"][0][0], inst["hom"][1][0]
    roots_text = f"{roots[0]}, {roots[1]}"
    root_dist = pick_distractors(
        roots_text,
        [
            _list_d([-roots[1], -roots[0]], _TAGS["root_sign"]),
            _list_d(sorted([p, q]), _TAGS["coefficients_as_roots"]),
            _list_d(sorted([-q, p]), _TAGS["coefficients_as_roots"]),
            _list_d(sorted([r1 * r2, r1 + r2 + 1]), _TAGS["coefficients_as_roots"]),
        ],
        3,
        unordered=True,
    )
    if kind == "poly2":
        part_terms = inst["part"]
        part_answer = _expr_text(part_terms)
        f = inst["forcing"]
        alpha, beta = f(1) - f(0), f(0)
        part_cands = [
            ([(-c, b, k) for c, b, k in part_terms], _TAGS["sign"]),
            ([(alpha, 1, 1), (beta, 1, 0)], _TAGS["forcing_as_particular"]),
            (
                [(part_terms[0][0], 1, 1)] if part_terms[0][2] == 1 else [(part_terms[1][0], 1, 1)],
                _TAGS["forcing_ignored"],
            ),
            ([(c + 1 if k == 1 else c, b, k) for c, b, k in part_terms], _TAGS["sign"]),
        ]
        part_prompt = "Find the particular solution of the form An + B. Give it as an expression in n."
        part_kind, part_value = "expression", part_answer
        part_dist = keep_distinct(
            part_value,
            [(t, tag, _expr_text(t)) for t, tag in part_cands if t and _integer_terms(t)],
            key=lambda t: _expr_text(t) if not isinstance(t, str) else t,
        )
        part_terms_for_working = part_answer
    else:
        (c_k, base, _k) = inst["part"][0]
        part_prompt = (
            f"Find the constant K for which aₙ = K·n·{base}ⁿ satisfies the recurrence for all n."
            if base > 0
            else f"Find the constant K for which aₙ = K·n·({base})ⁿ satisfies the recurrence for all n."
        )
        part_kind, part_value = "number", int(c_k)
        K = int(c_k)
        c_force = int(inst["forcing"](0))
        part_dist = keep_distinct(
            K,
            [
                (-K, _TAGS["sign"], str(-K)),
                (c_force, _TAGS["forcing_as_particular"], str(c_force)),
                (K + 1, _TAGS["sign"], str(K + 1)),
                (K * base, _TAGS["power_dropped"], str(K * base)),
            ],
        )
        part_terms_for_working = str(K)
    later = f"a{''.join('₀₁₂₃₄₅₆₇₈₉'[int(ch)] for ch in str(index))}"
    d_cands = [
        (int(inst["seq"][index - 1]), _TAGS["index"]),
        (int(inst["seq"][index + 1]) if index + 1 < len(inst["seq"]) else None, _TAGS["index"]),
    ]
    for terms, tag in _wrong_forms(inst):
        if terms is not None and _integer_terms(terms):
            d_cands.append((int(_value(terms, index)), tag))
    d_dist = keep_distinct(value, [(v, tag, str(v)) for v, tag in d_cands if v is not None])[:3]
    c_dist = _distractors(inst)[:3]
    parts = [
        dict(
            prompt="Write down the two roots of the characteristic equation of the homogeneous recurrence.",
            answer_kind="list",
            answer=roots,
            answer_text=roots_text,
            marks=2,
            working=[f"The roots are {roots[0]} and {roots[1]}, since r² − ({p})r − ({q}) = 0."],
            distractors=root_dist,
        ),
        dict(
            prompt=part_prompt,
            answer_kind=part_kind,
            answer=part_value,
            answer_text=str(part_value),
            marks=3,
            working=[f"Substituting into the recurrence and matching terms gives {_show(part_terms_for_working)}."],
            distractors=part_dist,
        ),
        dict(
            prompt="Find a closed form for aₙ in terms of n.",
            answer_kind="expression",
            answer=answer,
            answer_text=answer,
            marks=4,
            working=[
                f"aₙ = (homogeneous solution) + (particular solution); the starting terms fix the constants: aₙ = {_show(answer)}."
            ],
            distractors=c_dist,
        ),
        dict(
            prompt=f"Hence find {later}.",
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            marks=1,
            follow_through=f"Subs(part_c, (n,), ({index},))",
            working=[f"Substitute n = {index} into the closed form: {value}."],
            distractors=d_dist,
            sensible_range=None,
        ),
    ]
    check(all(len(x["distractors"]) >= 2 for x in parts), "too few distractors in a part")
    return dict(text=stem, latex=stem_tex, parts=parts)


def _list_d(values, tag):
    return Distractor(f"{values[0]}, {values[1]}", tag, list(values))
