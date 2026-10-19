"""Tests `supply-demand-equilibrium` and `fair-game-expected-gain` (topics `w3_supply_demand`, `w3_fair_game`, exams).

The oracles read the demand and supply rules, the prizes and the counts back out of the question TEXT and re-derive each
key independently: supply and demand by solving the two rules with Fractions, fair games by listing every equally likely
outcome.
"""

from __future__ import annotations

import itertools
import re
from fractions import Fraction

import mathgen.generators.w3_models_build as mod
from tests import w3_support as w3
from tests import w7b_support as sup
from tests.w3_support import dollars

SD, SD_X = "w3_supply_demand", "exd_x_w3_supply_demand"
FG, FG_X = "w3_fair_game", "exd_x_w3_fair_game"
SK_SD = sup.skill_of("supply-demand-equilibrium")
SK_FG = sup.skill_of("fair-game-expected-gain")


def test_catalogue_has_every_module_misconception():
    w3.check_catalogue(mod)


class TestSupplyDemandShared(w3.shared_checks(SK_SD, SD, SD_X)):
    pass


class TestFairGameShared(w3.shared_checks(SK_FG, FG, FG_X)):
    pass


# -- supply and demand ------------------------------------------------------------------------------------------------


def rules(text):
    """(a, b, c, d) with demand Q = a - bP and supply Q = c + dP, read from the text."""
    a, b = re.search(r"Q = (\d+) - (\d+)P", text).groups()
    m = re.search(r"Q = (?:(\d+) \+ )?(\d+)P(?!\))", text.split("supply")[1])
    c, d = m.groups()
    return int(a), int(b), int(c or 0), int(d)


def equilibrium(a, b, c, d):
    p = Fraction(a - c, b + d)
    q = a - b * p
    assert q == c + d * p
    return p, q


def test_sd_easy_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(SD, "easy", seed)
        t, ans = q.text, int(q.answer_text)
        a, b, c, d = rules(t)
        price = int(re.search(r"price is \$(\d+)", t).group(1))
        qd, qs = a - b * price, c + d * price
        if "units are demanded" in t:
            assert ans == qd, (seed, t)
            seen.add("demand")
        elif "units are supplied" in t:
            assert ans == qs, (seed, t)
            seen.add("supply")
        else:
            word = re.search(r"What is the (shortage|surplus)", t).group(1)
            assert ans == abs(qs - qd) > 0, (seed, t)
            assert (word == "surplus") == (qs > qd)
            seen.add("gap")
    assert seen == {"demand", "supply", "gap"}


def test_sd_medium_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(SD, "medium", seed)
        t, ans = q.text, int(q.answer_text)
        p, qty = equilibrium(*rules(t))
        if "spent on the product" in t:
            assert ans == p * qty, (seed, t)
            seen.add("revenue")
        elif "Find the equilibrium quantity" in t:
            assert ans == qty, (seed, t)
            seen.add("quantity")
        else:
            assert "Find the equilibrium price" in t
            assert ans == p, (seed, t)
            seen.add("price")
    assert seen == {"price", "quantity", "revenue"}


def line_through(p1, q1, p2, q2):
    slope = Fraction(q2 - q1, p2 - p1)
    return slope, q1 - slope * p1  # Q = slope * P + intercept


def test_sd_hard_keys():
    seen = set()
    for seed in range(250):
        q = sup.single(SD, "hard", seed)
        t, ans = q.text, int(q.answer_text)
        if "At $" in t:
            m = re.findall(r"At \$(\d+) the quantity demanded is (\d+) and the quantity supplied is (\d+)", t)
            (p1, d1, s1), (p2, d2, s2) = [tuple(int(x) for x in row) for row in m]
            sd, id_ = line_through(p1, d1, p2, d2)
            ss, is_ = line_through(p1, s1, p2, s2)
            assert sd < 0 < ss
            price = (is_ - id_) / (sd - ss)
            assert price == ans, (seed, t)
            seen.add("points")
        elif "tax" in t:
            a, b = (int(x) for x in re.search(r"Q = (\d+) - (\d+)P", t).groups())
            c, d = (int(x) for x in re.search(r"supply is given by Q = (\d+) \+ (\d+)P", t).groups())
            tax = int(re.search(r"tax of \$(\d+)", t).group(1))
            # buyers pay P; sellers receive P - tax; solve a - bP = c + d(P - tax)
            price = Fraction(a - c + d * tax, b + d)
            assert a - b * price == ans, (seed, t)
            assert c + d * (price - tax) == ans
            seen.add("tax")
        else:
            a, b, c, d = rules(t)
            surplus = int(re.search(r"surplus of (\d+) units", t).group(1))
            assert (c + d * ans) - (a - b * ans) == surplus, (seed, t)
            seen.add("surplus")
    assert seen == {"points", "tax", "surplus"}


def test_sd_rules_are_sensible():
    for tier in sup.TIERS:
        for seed in range(80):
            q = sup.single(SD, tier, seed)
            if "At $" in q.text:
                continue
            a, b, c, d = rules(q.text)
            p, qty = equilibrium(a, b, c, d)
            assert p > 0 and qty > 0 and c >= 0 and b > 0 and d > 0


def test_sd_exam_keys_and_shape():
    sup.check_exam_chain(SK_SD, SD_X, [1, 2, 2, 2], ["number"] * 4)
    for seed in range(100):
        q = sup.exam(SD_X, seed)
        a, b, c, d = rules(q.stem)
        pa, pb, pc, pd = q.parts
        price = int(re.search(r"price is \$(\d+)", pa.prompt).group(1))
        assert int(pa.answer_text) == a - b * price
        assert int(pb.answer_text) == (a - b * price) - (c + d * price) > 0
        p, qty = equilibrium(a, b, c, d)
        assert int(pc.answer_text) == p and int(pd.answer_text) == qty


def test_sd_exam_follow_through():
    def wrong(q, canon):
        a, b, c, d = rules(q.stem)
        pc = int(q.parts[2].answer_text)
        bad = pc + 1
        return {"c": str(bad), "d": str(a - b * bad)}

    assert sup.check_follow_through(SK_SD, SD_X, "d", wrong) >= 30


# -- fair games -------------------------------------------------------------------------------------------------------


def device_n(text):
    m = re.search(r"(\d+) equal sectors|holding (\d+) identical|(\d+) numbered", text)
    return int(next(g for g in m.groups() if g))


def outcomes(text):
    """The prize on each of the N equally likely outcomes, read from the text."""
    n = device_n(text)
    prizes = [0] * n
    i = 0
    for prize, count in re.findall(r"\$(\d+) on (\d+) (?:sector|counter|ticket)", text):
        for _ in range(int(count)):
            prizes[i] = int(prize)
            i += 1
    return prizes


def fee_of(text):
    return Fraction(re.search(r"costs \$([\d.]+) to play", text).group(1))


def expected(prizes):
    return Fraction(sum(prizes), len(prizes))


def test_fg_outcomes_parser_is_sound():
    for seed in range(60):
        q = sup.single(FG, "easy", seed)
        prizes = outcomes(q.text)
        assert sum(1 for p in prizes if p) == sum(int(c) for _, c in re.findall(r"\$(\d+) on (\d+)", q.text))


def test_fg_easy_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(FG, "easy", seed)
        t, ans = q.text, Fraction(q.answer_text)
        ev = expected(outcomes(t))
        if "expected gain per play" in t:
            assert ans == ev - fee_of(t), (seed, t)
            seen.add("gain")
        else:
            assert ans == ev, (seed, t)
            seen.add("prize")
    assert seen == {"gain", "prize"}


def test_fg_medium_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(FG, "medium", seed)
        t, ans = q.text, Fraction(q.answer_text)
        ev = expected(outcomes(t))
        if "entry fee" in t:
            assert ans == ev, (seed, t)
            seen.add("fee")
        else:
            plays = int(re.search(r"played (\d+) times", t).group(1))
            assert ans == (fee_of(t) - ev) * plays, (seed, t)
            seen.add("organiser")
    assert seen == {"fee", "organiser"}


def test_fg_hard_keys():
    seen = set()
    for seed in range(250):
        q = sup.single(FG, "hard", seed)
        t, ans = q.text, Fraction(q.answer_text)
        if "red counters" in t:
            red, green = (int(x) for x in re.search(r"holds (\d+) red counters and (\d+) green", t).groups())
            fee = dollars(t)[0]
            prize = dollars(t)[1]
            bag = ["r"] * red + ["g"] * green
            pairs = list(itertools.permutations(range(len(bag)), 2))
            wins = sum(1 for i, j in pairs if bag[i] == bag[j] == "r")
            assert ans == Fraction(wins, len(pairs)) * prize - fee, (seed, t)
            seen.add("two-draws")
        elif "prize $x" in t or "$x is paid" in t:
            n = device_n(t)
            p1, c1, c2 = re.search(
                r"prize of \$(\d+) is paid on (\d+) \w+ and a prize of \$x is paid on (\d+)", t
            ).groups()
            fee = dollars(t)[-1]
            assert (Fraction(int(c1) * int(p1), n) + Fraction(int(c2), n) * ans) == fee, (seed, t)
            seen.add("missing")
        else:
            ev = expected(outcomes(t))
            fee, target = fee_of(t), dollars(t)[-1]
            assert ans * (fee - ev) == target, (seed, t)
            seen.add("plays")
    assert seen == {"two-draws", "missing", "plays"}


def test_fg_exam_keys_and_shape():
    sup.check_exam_chain(SK_FG, FG_X, [2, 1, 1, 2], ["number"] * 4)
    for seed in range(100):
        q = sup.exam(FG_X, seed)
        ev = expected(outcomes(q.stem))
        fee = fee_of(q.stem)
        a, b, c, d = q.parts
        assert Fraction(a.answer_text) == ev
        assert Fraction(b.answer_text) == ev - fee
        wanted = dollars(c.prompt)[0]
        assert Fraction(c.answer_text) == ev + wanted
        plays = int(re.search(r"played (\d+) times", d.prompt).group(1))
        assert Fraction(d.answer_text) == (fee - ev) * plays


def test_fg_exam_follow_through():
    def wrong(q, canon):
        fee = fee_of(q.stem)
        bad = Fraction(q.parts[0].answer_text) + 2
        return {"a": f"{float(bad):.2f}", "b": f"{float(bad - fee):.2f}"}

    assert sup.check_follow_through(SK_FG, FG_X, "b", wrong) >= 30
