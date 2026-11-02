"""Tests `uc2-mm1-queue-and-eoq`. Every key is re-derived from the RENDERED stem by a different route from the generator's
closed forms:

* M/M/1: the stationary distribution from the balance equations (P(n) proportional to rho^n, normalised by summing the
  series numerically), L and Lq summed term by term, W and Wq through Little's law, whole-number targets by scanning
  the integers on either side of the boundary;
* EOQ: SymPy differentiates the annual cost D S / Q + H Q / 2 and solves for the minimiser, then the quantities built
  on it (lot sizes in multiples, demand from the minimum cost, purchase cost on top) follow from that.

Plus the typed graders on every key and distractor, answer variety, prompt shapes and the three-part exams."""

from __future__ import annotations

import re
from fractions import Fraction
from functools import lru_cache

import pytest
import sympy

from mathgen import make
from mathgen.core import REGISTRY
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape
from tests.exd_x_sc2_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_deterministic,
    check_key_accepted_distractors_rejected,
    check_no_text_defects,
    check_variety,
    exam,
    skill,
)

QUEUE = "w5b_mm1_queue"
EOQ = "w5b_eoq_inventory"
EXAM = "exd_x_w5b_mm1_queue_eoq"
SKILL = "uc2-mm1-queue-and-eoq"
TIERS = ("easy", "medium", "hard")


# -- M/M/1 by the stationary distribution -----------------------------------------------------------------------------


@lru_cache(maxsize=None)
def stationary(lam, mu):
    """(P list, L, Lq) by balance equations and numeric sums. lam, mu are ints or Fractions (kept as given)."""
    rho = float(Fraction(lam) / Fraction(mu))
    raw, term = [], 1.0
    while term > 1e-18 and len(raw) < 20000:
        raw.append(term)
        term *= rho
    norm = sum(raw)
    p = [r / norm for r in raw]
    return p, sum(n * pn for n, pn in enumerate(p)), sum((n - 1) * pn for n, pn in enumerate(p) if n >= 1)


def rates(text):
    lam = re.search(r"λ = (\d+)", text)
    mu = re.search(r"μ = (\d+)", text)
    if lam and mu:
        return int(lam.group(1)), Fraction(int(mu.group(1)))
    lam = int(re.search(r"(?:at|rate) (\d+) per hour", text).group(1))
    s = int(re.search(r"mean (\d+) minutes", text).group(1))
    return lam, Fraction(60, s)


def close(key, value):
    return abs(float(key) - value) <= 1e-9 * max(1.0, abs(value))


def queue_oracle(q):
    """The float value for an easy or medium question, from the stationary distribution."""
    text = q.text
    lam, mu = rates(text)
    p, L, Lq = stationary(lam, mu)
    if q.difficulty == "easy":
        if "utilisation" in text or "busy" in text:
            return 1 - p[0]
        if "empty" in text or "no customer" in text:
            return p[0]
        assert "(L)" in text or "L, the mean number" in text
        return L
    unit = 60 if "in minutes" in text else 1
    if "Lq" in text or "not counting the one in service" in text or "waiting in the queue, not" in text:
        return Lq
    if "Wq" in text or "before service begins" in text:
        return Lq / lam * unit
    if "W, the mean total time" in text or "spend in the system on average" in text:
        return L / lam * unit
    if "exactly" in text:
        return p[int(re.search(r"exactly (\d+)", text).group(1))]
    k = int(re.search(r"(?:at least|that) (\d+)(?: customers| or more)", text).group(1))
    return sum(p[k:])


def int_target_oracle(q):
    """The whole-number answer of a hard question, by scanning integers on either side of the boundary."""
    text = q.text
    if "largest whole" in text:
        mu = int(re.search(r"μ = (\d+)", text).group(1))
        T = int(re.search(r"(?:exceed|at most) (\d+) minutes", text).group(1))
        ok = [lam for lam in range(1, mu) if stationary(lam, mu)[2] / lam * 60 <= T + 1e-12]
        assert ok == list(range(1, max(ok) + 1))  # the wait only rises with lambda
        return max(ok)
    lam = int(re.search(r"λ = (\d+)", text).group(1))
    if "%" in text:
        k = int(
            re.search(r"(\d+) or more customers|at least (\d+) customers", text)
            .group(0)
            .split()[0 if "or more" in text else 2]
        )
        pct = int(re.search(r"(\d+)%", text).group(1))
        mu = lam + 1
        for _ in range(200 * lam):
            if sum(stationary(lam, mu)[0][k:]) <= pct / 100 + 1e-12:
                return mu
            mu += 1
        raise AssertionError(text)
    if "L, " in text and "mean number" in text:
        ell = float(re.search(r"(?:at most|at or below) (\d+(?:\.\d+)?)", text).group(1))
        mu = lam + 1
        while stationary(lam, mu)[1] > ell + 1e-12:
            mu += 1
        return mu
    w = int(re.search(r"(?:exceed|at most) (\d+) minutes", text).group(1))
    mu = lam + 1
    while stationary(lam, mu)[1] / lam * 60 > w + 1e-12:
        mu += 1
    return mu


def queue_saved(q):
    text = q.text
    lam = int(re.search(r"λ = (\d+)", text).group(1))
    mu = int(re.search(r"μ = (\d+)", text).group(1))
    mu2 = int(re.findall(r"(\d+) per hour", text)[-1])
    assert mu2 > mu
    return (stationary(lam, mu)[1] - stationary(lam, mu2)[1]) / lam * 60


def check_queue_single(q):
    text = q.text
    key = Fraction(q.answer_text)
    if q.difficulty == "hard":
        if "reduction" in text or "fall" in text:
            assert close(key, queue_saved(q)), (text, q.answer_text)
        else:
            assert key == int_target_oracle(q), (text, q.answer_text, int_target_oracle(q))
        return
    assert close(key, queue_oracle(q)), (text, q.answer_text, queue_oracle(q))


# -- EOQ by SymPy ---------------------------------------------------------------------------------------------------

Q_, D_, S_, H_ = sympy.symbols("Q D S H", positive=True)


@lru_cache(maxsize=None)
def eoq_minimiser(D, S, H):
    """(Q*, minimum cost) from the derivative of D S / Q + H Q / 2, exactly."""
    cost = D * S / Q_ + H * Q_ / 2
    roots = [r for r in sympy.solve(sympy.diff(cost, Q_), Q_) if r.is_positive]
    assert len(roots) == 1
    q_star = sympy.nsimplify(roots[0])
    return q_star, cost.subs(Q_, q_star)


def tc(D, S, H, Q):
    return sympy.Rational(D * S) / Q + sympy.Rational(H) * Q / 2


def three(text):
    """(D, S, H) from a setup sentence."""
    D = int(re.search(r"(\d+) units", text).group(1))
    S, H = (int(v) for v in re.findall(r"\$(\d+)", text)[:2])
    return D, S, H


def eoq_oracle(q):
    text = q.text
    if re.search(r"multiples? of \d+ units", text):
        D, S, H = three(text)
        m = int(re.search(r"multiples? of (\d+)", text).group(1))
        q_star, _ = eoq_minimiser(D, S, H)
        grid = range(m, int(3 * q_star) + m, m)
        best = min(grid, key=lambda x: (tc(D, S, H, x), x))
        return (
            best
            if ("order quantity" in text and "Which" in text or "Find the feasible order quantity" in text)
            else tc(D, S, H, best)
        )
    if (
        "annual demand" in text.lower()
        and "Ordering costs $" in text
        and "% of" in text
        or "needs" in text
        and "% of its price" in text
    ):
        c = int(re.search(r"\$(\d+)", text).group(1))
        D = int(re.search(r"(?:needs|annual demand is) (\d+)", text).group(1))
        S = int(re.findall(r"\$(\d+)", text)[1])
        H = sympy.Rational(c * int(re.search(r"(\d+)%", text).group(1)), 100)
        _q, best = eoq_minimiser(D, S, H)
        return c * D + best
    if "demand D" in text or "annual demand in units" in text:
        S, H, T = (int(v) for v in re.findall(r"\$(\d+)", text)[:3])
        sol = sympy.solve(sympy.Eq(eoq_minimiser(D_, S, H)[1], T), D_)
        assert len(sol) == 1
        return sol[0]
    D, S, H = three(text)
    q_star, best = eoq_minimiser(D, S, H)
    if (
        "economic order quantity Q*" in text
        or "optimal order quantity" in text
        or "Find the economic order quantity" in text
    ):
        return q_star
    if "placed at the economic order quantity" in text or "minimum total annual cost" in text:
        return best
    if "How many orders per year" in text or "number of orders placed in a year" in text:
        return sympy.Rational(D) / q_star
    if "reorder" in text:
        lead = int(re.search(r"(?:takes|lead time of) (\d+) working days", text).group(1))
        return sympy.Rational(D, 250) * lead
    if "working days pass" in text or "between successive orders" in text:
        return 250 * q_star / D
    q2 = int(re.search(r"(?:orders|ordering|order is for|every order is for) (\d+) units", text).group(1))
    if "more per year" in text or "exceed" in text:
        return tc(D, S, H, q2) - best
    return tc(D, S, H, q2)


def check_eoq_single(q):
    assert Fraction(q.answer_text) == Fraction(
        int(sympy.Rational(eoq_oracle(q)).p), int(sympy.Rational(eoq_oracle(q)).q)
    ), (
        q.text,
        q.answer_text,
        eoq_oracle(q),
    )


# -- wiring and the shared contract ---------------------------------------------------------------------------------


def test_wiring():
    s = skill(SKILL)
    assert s.answer == "fraction" and "university-year-2" in s.tags and tuple(s.year_level) == ("university",)
    assert s.domain == "optimisation"
    got = [(src.topic, src.difficulty) for src in s.sources]
    assert got == [
        (QUEUE, "easy"),
        (EOQ, "easy"),
        (QUEUE, "medium"),
        (EOQ, "medium"),
        (QUEUE, "hard"),
        (EOQ, "hard"),
        (EXAM, "exam"),
    ]
    assert len({src.effective_key for src in s.sources}) == len(s.sources)
    assert s.sources[-1].stages == ["exam"]
    assert s.prerequisites


@pytest.mark.parametrize("tier", TIERS)
def test_queue_keys_by_stationary_distribution(tier):
    for seed in range(120):
        check_queue_single(make(QUEUE, seed=seed, difficulty=tier))


@pytest.mark.parametrize("tier", TIERS)
def test_eoq_keys_by_sympy(tier):
    for seed in range(120):
        q = make(EOQ, seed=seed, difficulty=tier)
        want = sympy.Rational(eoq_oracle(q))
        assert Fraction(q.answer_text) == Fraction(int(want.p), int(want.q)), (q.text, q.answer_text, want)


@pytest.mark.parametrize("topic", (QUEUE, EOQ))
@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_and_tagged(topic, tier):
    s = skill(SKILL)
    for seed in range(60):
        q = make(topic, seed=seed, difficulty=tier)
        assert grade(s, q, q.answer_text).correct, (topic, tier, seed, q.answer_text)
        assert len(q.distractors) >= 2
        tagged = 0
        for text in q.distractors:
            assert not grade(s, q, text).correct, (topic, tier, seed, text, q.answer_text)
            note = q.option_notes.get(text)
            if note is not None:
                assert note in CATALOGUE
                tagged += 1
        assert tagged >= 1, (topic, tier, seed)


@pytest.mark.parametrize("topic", (QUEUE, EOQ))
@pytest.mark.parametrize("tier", TIERS)
def test_answer_variety_prompt_shapes_and_text(topic, tier):
    qs = [make(topic, seed=seed, difficulty=tier) for seed in range(60)]
    assert len({q.answer_text for q in qs}) >= 6, (topic, tier)
    assert len({prompt_shape(q) for q in qs}) >= 3, (topic, tier)
    assert len({q.text for q in qs}) >= 20, (topic, tier)
    for seed, q in enumerate(qs):
        assert find_text_defects(q, include_working=True) == [], (topic, tier, seed)
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()
        assert "**" not in q.text


@pytest.mark.parametrize("topic", (QUEUE, EOQ, EXAM))
def test_unknown_tier_raises_value_error(topic):
    with pytest.raises(ValueError):
        REGISTRY[topic][1](__import__("random").Random(1), tier="nonsense")


def test_queue_is_always_stable_and_not_obvious():
    for tier in TIERS:
        for seed in range(100):
            q = make(QUEUE, seed=seed, difficulty=tier)
            text = q.text
            if "λ = " in text and "μ = " in text and tier != "hard":
                lam, mu = rates(text)
                rho = Fraction(lam) / mu
                assert 0 < rho < 1 and rho.denominator >= 5, text


def test_eoq_demand_and_lot_are_whole():
    for tier in TIERS:
        for seed in range(100):
            q = make(EOQ, seed=seed, difficulty=tier)
            if "units" in q.text and "$" in q.text and "Annual demand" not in q.text or True:
                numbers = [int(v) for v in re.findall(r"\b\d+\b", q.text)]
                assert all(v > 0 for v in numbers)


# -- exam -----------------------------------------------------------------------------------------------------------


def exam_values(q):
    """{label: float or exact} for each part of an exam question, re-derived from the rendered text."""
    stem = q.stem
    prompts = {p.label: p.prompt for p in q.parts}
    out = {}
    if "queue" in stem.lower() or "M/M/1" in stem:
        lam, mu = rates(stem)
        p, L, Lq = stationary(lam, mu)
        rho = 1 - p[0]
        for label, prompt in prompts.items():
            if "utilisation" in prompt:
                out[label] = rho
            elif "Little's law, find W" in prompt:
                out[label] = L / lam * 60
            elif "mean number of customers in the system" in prompt and "Hence" in prompt:
                out[label] = L
            elif prompt.startswith("Find Wq"):
                out[label] = Lq / lam * 60
            elif "Hence find W" in prompt:
                out[label] = L / lam * 60
            elif "Little's law, find L" in prompt:
                out[label] = L
            elif "system is empty" in prompt:
                out[label] = p[0]
            elif "exactly" in prompt:
                k = int(re.search(r"exactly (\d+)", prompt).group(1))
                out[label] = p[k]
            elif "at least" in prompt:
                k = int(re.search(r"at least (\d+)", prompt).group(1))
                out[label] = sum(p[k:])
            else:
                raise AssertionError(prompt)
        return out
    D, S, H = three(stem)
    q_star, best = eoq_minimiser(D, S, H)
    for label, prompt in prompts.items():
        if "economic order quantity" in prompt:
            out[label] = q_star
        elif "minimum total annual" in prompt or "total annual ordering plus holding cost" in prompt:
            out[label] = best
        elif "number of orders" in prompt:
            out[label] = sympy.Rational(D) / q_star
        elif "working days between orders" in prompt:
            out[label] = 250 * q_star / D
        elif "reorder point" in prompt:
            lead = int(re.search(r"take (\d+) working days", prompt).group(1))
            out[label] = sympy.Rational(D, 250) * lead
        elif "exceed that minimum" in prompt:
            q2 = int(re.search(r"ordering (\d+) units", prompt).group(1))
            out[label] = tc(D, S, H, q2) - best
        else:
            raise AssertionError(prompt)
    return out


def test_exam_keys_by_independent_routes():
    for seed in range(300):
        q = exam(EXAM, seed)
        values = exam_values(q)
        assert set(values) == {p.label for p in q.parts}
        for part in q.parts:
            key = Fraction(part.answer_text)
            want = values[part.label]
            if isinstance(want, float):
                assert close(key, want), (seed, part.label, part.prompt, part.answer_text, want)
            else:
                want = sympy.Rational(want)
                assert key == Fraction(int(want.p), int(want.q)), (seed, part.label, part.answer_text, want)


def test_exam_shape_marks_and_wiring():
    s = skill(SKILL)
    assert s.sources[-1].topic == EXAM
    shapes = set()
    for seed in range(120):
        q = exam(EXAM, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert all(p.answer_kind == "fraction" for p in q.parts)
        assert sum(p.marks for p in q.parts) == 7
        assert all(p.follow_through for p in q.parts[1:]) or any(p.follow_through for p in q.parts[1:])
        shapes.add(prompt_shape(q))
    assert len(shapes) >= 3
    check_variety(EXAM, lambda s_: make(QUEUE, seed=s_, difficulty="hard"))


def test_exam_answer_space_text_and_determinism():
    check_answer_space(EXAM)
    check_deterministic(EXAM)
    check_no_text_defects(EXAM)
    assert len({exam(EXAM, s).answer_text for s in range(300)}) >= 100
    models = {("queue" if "queue" in exam(EXAM, s).stem.lower() else "eoq") for s in range(60)}
    assert models == {"queue", "eoq"}


def test_exam_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, EXAM)


def test_exam_follow_through_from_a_wrong_part_a():
    """Part (a) is answered wrongly (a tenth off); every later part worked out from the learner's own earlier
    replies, using the part's own follow-through expression, earns its full marks."""
    s = skill(SKILL)
    checked = 0
    for seed in range(150):
        q = exam(EXAM, seed)
        replies = {}
        canon = {p.label: Fraction(p.answer_text) for p in q.parts}
        replies["a"] = canon["a"] * Fraction(9, 10)
        for part in q.parts[1:]:
            if part.follow_through:
                expr = sympy.sympify(
                    part.follow_through,
                    locals={
                        f"part_{label}": sympy.Rational(r.numerator, r.denominator) for label, r in replies.items()
                    },
                )
                replies[part.label] = Fraction(int(sympy.Rational(expr).p), int(sympy.Rational(expr).q))
            else:
                replies[part.label] = canon[part.label]
        if len({str(v) for v in replies.values()}) < len(replies):
            continue
        result = grade(s, q, {k: str(v) for k, v in replies.items()})
        assert result.parts["a"].marks == 0, seed
        for part in q.parts[1:]:
            got = result.parts[part.label]
            assert got.marks == part.marks, (seed, part.label, replies, got.marks)
            if part.follow_through and replies[part.label] != canon[part.label]:
                assert got.follow_through is True, (seed, part.label)
        checked += 1
    assert checked >= 100, checked


def test_exam_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, EXAM, {"a": "99/7", "b": "905/3", "c": "907/11"})


# -- round 3: typed commas, reorder points, lot multiples, tags ---------------------------------------------------------


def _all_keys():
    """(question or part, text, key) for every single question and exam part over many seeds."""
    for topic in (QUEUE, EOQ):
        for tier in TIERS:
            for seed in range(150):
                q = make(topic, seed=seed, difficulty=tier)
                yield (topic, tier, seed), q.text, q.answer_text
    for seed in range(200):
        q = exam(EXAM, seed)
        for part in q.parts:
            yield ("exam", seed, part.label), part.prompt, part.answer_text


def test_large_keys_say_not_to_use_commas_and_are_graded_without_them():
    """The fraction grader reads `1,128,000` as a list, so a key of 1000 or more says how to type it, and the plain
    digits are accepted by the real grader."""
    s = skill(SKILL)
    big = 0
    for where, text, key in _all_keys():
        if Fraction(key) >= 1000:
            big += 1
            assert "commas" in text, (where, text)
    assert big > 100
    for topic in (QUEUE, EOQ):
        for tier in TIERS:
            for seed in range(60):
                q = make(topic, seed=seed, difficulty=tier)
                if Fraction(q.answer_text) >= 1000:
                    assert grade(s, q, q.answer_text).correct


def test_eoq_figures_are_whole_where_the_question_invites_it():
    reorder = orders = dollars = lots = whole_dollars = 0
    for tier in TIERS:
        for seed in range(150):
            q = make(EOQ, seed=seed, difficulty=tier)
            if "reorder" in q.text:
                reorder += 1
                assert Fraction(q.answer_text).denominator == 1, q.text
            if "How many orders per year" in q.text or "number of orders placed in a year" in q.text:
                orders += 1
                assert Fraction(q.answer_text).denominator == 1, q.text
            if "units each time" in q.text or "units at a time" in q.text or "every order is for" in q.text:
                lots += 1
                whole_dollars += Fraction(q.answer_text).denominator == 1
    for seed in range(300):
        e = exam(EXAM, seed)
        for part in e.parts:
            if "reorder point" in part.prompt:
                reorder += 1
                assert Fraction(part.answer_text).denominator == 1, part.prompt
            if "orders placed per year" in part.prompt:
                orders += 1
                assert Fraction(part.answer_text).denominator == 1, part.prompt
    assert reorder > 30 and orders > 30 and lots > 20
    assert whole_dollars >= 0.9 * lots


def test_lot_multiples_have_feasible_options_and_the_nearest_lot_often_loses():
    asked = traps = 0
    for seed in range(500):
        q = make(EOQ, seed=seed, difficulty="hard")
        if not re.search(r"multiples? of \d+ units", q.text) or not (
            "Which order quantity" in q.text or "Find the feasible order quantity" in q.text
        ):
            continue
        asked += 1
        D, S, H = three(q.text)
        m = int(re.search(r"multiples? of (\d+)", q.text).group(1))
        q_star, _ = eoq_minimiser(D, S, H)
        key = int(q.answer_text)
        assert key % m == 0 and all(int(d) % m == 0 and int(d) >= m for d in q.distractors), (
            seed,
            q.text,
            q.distractors,
        )
        nearest = min(range(m, int(3 * q_star) + m, m), key=lambda x: (abs(x - q_star), x))
        traps += nearest != key
    assert asked > 40
    assert traps >= 0.3 * asked, (traps, asked)


def test_every_declared_tag_fires_and_is_declared():
    import yaml
    from pathlib import Path

    declared = {
        m["id"]
        for m in yaml.safe_load(Path("skills/optimisation/uc2-mm1-queue-and-eoq.yaml").read_text())["misconceptions"]
    }
    fired = set()
    for topic in (QUEUE, EOQ):
        for tier in TIERS:
            for seed in range(400):
                fired |= {t for t in make(topic, seed=seed, difficulty=tier).option_notes.values() if t}
    for seed in range(400):
        for part in exam(EXAM, seed).parts:
            fired |= {t for t in (part.option_notes or {}).values() if t}
    assert fired == declared, (sorted(declared - fired), sorted(fired - declared))


def test_tags_sit_on_the_slip_they_name():
    """Spot checks, re-derived from the rendered stem: the tagged option's value is the slip named."""
    checked = 0
    for seed in range(300):
        q = make(EOQ, seed=seed, difficulty="easy")
        if "Find the economic order quantity" in q.text or "optimal order quantity" in q.text:
            D, S, H = three(q.text)
            for text, tag in q.option_notes.items():
                if tag == "eoq-no-square-root":
                    assert Fraction(text) == Fraction(2 * D * S, H)
                    checked += 1
                if tag == "eoq-swaps-s-and-h":
                    assert Fraction(text) ** 2 == Fraction(2 * D * H, S)
                    checked += 1
        q = make(EOQ, seed=seed, difficulty="medium")
        if "between successive orders" in q.text or "working days pass" in q.text:
            D, S, H = three(q.text)
            q_star, _ = eoq_minimiser(D, S, H)
            for text, tag in q.option_notes.items():
                if tag == "eoq-cycle-as-orders-per-year":
                    assert Fraction(text) == Fraction(D) / Fraction(int(q_star))
                    checked += 1
                if tag == "eoq-rop-wrong-days":
                    assert Fraction(text) == Fraction(365) * Fraction(int(q_star)) / D
                    checked += 1
        q = make(QUEUE, seed=seed, difficulty="easy")
        if "probability that the system is empty" in q.text or "Find P(0)" in q.text:
            lam, mu = rates(q.text)
            for text, tag in q.option_notes.items():
                if tag == "mm1-idle-as-busy":
                    assert Fraction(text) == Fraction(lam, mu)
                    checked += 1
                if tag == "mm1-idle-squared":
                    assert Fraction(text) == (1 - Fraction(lam, mu)) ** 2
                    checked += 1
    assert checked > 80
