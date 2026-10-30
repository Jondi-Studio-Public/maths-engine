"""`uc2-interior-closure-boundary`: keys graded right, distractors graded wrong, and every answer recomputed from the stem.

The oracle reads the RENDERED text of a question, rebuilds the set it describes, and finds the interior, closure,
boundary and limit points by looking at neighbourhoods rather than by the generator's interval algebra:

- a subset of R is cut at its critical points (endpoints, isolated points, the first terms and the limits of sequences),
  each open gap between them is tested once with a real sample point and once as a rational one, and each critical point
  is classified from the two gaps beside it;
- a point of the plane is tested by sampling a small circle of points around it against the inequality in the stem.
"""

from __future__ import annotations

import re
from fractions import Fraction as Fr

import numpy as np
import pytest
import sympy

from mathgen import make
from mathgen.engine import grade
from mathgen.grading.inequality import read_solution_set
from mathgen.text_quality import find_text_defects, prompt_shape
from tests.exd_x_sc2_support import check_all_wrong_earns_nothing, check_deterministic, check_wired, skill

SKILL = "uc2-interior-closure-boundary"
TOPIC = "w5b_interior_closure"
EXAM = "exd_x_w5b_interior_closure"
SEEDS = range(60)
NUM = r"-?\d+(?:\.\d+)?"


def tier_q(tier: str, seed: int):
    return make(EXAM if tier == "exam" else TOPIC, seed=seed, difficulty=tier)


def fr(text: str) -> Fr:
    return Fr(text)


# -- reading a subset of R off the stem ---------------------------------------------------------------------------------------


PIECE = re.compile(
    rf"(?P<q>ℚ ∩ (?P<ql>[\[(])(?P<qa>{NUM}), (?P<qb>{NUM})(?P<qr>[\])]))"
    rf"|(?P<seq>\{{(?:(?P<sc>{NUM}) (?P<ss>[+-]) )?(?P<sm>-?)1/n : n = 1, 2, 3, …\}})"
    rf"|(?P<pt>\{{(?P<pv>{NUM})\}})"
    rf"|(?P<iv>(?P<il>[\[(])(?P<ia>{NUM}), (?P<ib>{NUM})(?P<ir>[\])]))"
)


def read_set(text: str):
    """The pieces of the subset of R written in `text`, as dicts."""
    pieces = []
    for m in PIECE.finditer(text):
        if m.group("q"):
            pieces.append(
                dict(t="Q", a=fr(m.group("qa")), b=fr(m.group("qb")), l=m.group("ql") == "[", r=m.group("qr") == "]")
            )
        elif m.group("seq"):
            if m.group("sc") is None:
                c, sgn = Fr(0), -1 if m.group("sm") else 1
            else:
                c, sgn = fr(m.group("sc")), 1 if m.group("ss") == "+" else -1
            pieces.append(dict(t="S", c=c, sgn=sgn))
        elif m.group("pt"):
            pieces.append(dict(t="P", a=fr(m.group("pv"))))
        else:
            pieces.append(
                dict(t="I", a=fr(m.group("ia")), b=fr(m.group("ib")), l=m.group("il") == "[", r=m.group("ir") == "]")
            )
    return pieces


def member(pieces, x: Fr, rational: bool = True) -> bool:
    for p in pieces:
        if p["t"] == "I" and (p["a"] < x < p["b"] or (x == p["a"] and p["l"]) or (x == p["b"] and p["r"])):
            return True
        if p["t"] == "P" and x == p["a"]:
            return True
        if rational and p["t"] == "Q" and (p["a"] < x < p["b"] or (x == p["a"] and p["l"]) or (x == p["b"] and p["r"])):
            return True
        if rational and p["t"] == "S":
            d = x - p["c"]
            if d != 0 and (d > 0) == (p["sgn"] > 0) and (1 / abs(d)).denominator == 1:
                return True
    return False


def covers(pieces, kind: str, x: Fr) -> bool:
    return any(p["t"] == kind and p["a"] < x < p["b"] for p in pieces)


class Line:
    """Interior, closure, boundary and limit points of a subset of R, found from neighbourhoods."""

    def __init__(self, pieces):
        self.pieces = pieces
        crit = set()
        for p in pieces:
            if p["t"] in ("I", "Q"):
                crit |= {p["a"], p["b"]}
            elif p["t"] == "P":
                crit.add(p["a"])
            else:
                crit.add(p["c"])
                crit |= {p["c"] + p["sgn"] * Fr(1, n) for n in range(1, 41)}
        self.crit = sorted(crit)
        c = self.crit
        self.gaps = [(None, c[0])] + list(zip(c, c[1:])) + [(c[-1], None)]
        self.gap_full, self.gap_dense = [], []
        for lo, hi in self.gaps:
            mid = (lo + hi) / 2 if lo is not None and hi is not None else (hi - 1 if lo is None else lo + 1)
            self.gap_full.append(covers_real(pieces, mid))
            self.gap_dense.append(covers(pieces, "Q", mid))
        limits = {p["c"] for p in pieces if p["t"] == "S"}
        self.limits_seq = limits
        self.kind = {}
        for i, x in enumerate(c):
            lf, rf = self.gap_full[i], self.gap_full[i + 1]
            ld, rd = self.gap_dense[i], self.gap_dense[i + 1]
            inside = member(pieces, x)
            interior = inside and lf and rf
            limit = lf or rf or ld or rd or x in limits
            self.kind[x] = dict(inside=inside, interior=interior, limit=limit, closure=inside or limit)

    # sets as lists of points or runs

    def _runs(self, flag_point, flag_gap):
        """Maximal runs of consecutive atoms (points and gaps) that satisfy the flags: [(lo, hi)], None for unbounded."""
        atoms = []
        for i, (lo, hi) in enumerate(self.gaps):
            atoms.append(("gap", i, flag_gap(i)))
            if i < len(self.crit):
                atoms.append(("pt", self.crit[i], flag_point(self.crit[i])))
        runs, cur = [], None
        for kind, ref, ok in atoms:
            if ok:
                if cur is None:
                    cur = [kind, ref, kind, ref]
                else:
                    cur[2:] = [kind, ref]
            elif cur is not None:
                runs.append(cur)
                cur = None
        if cur is not None:
            runs.append(cur)
        out = []
        for k0, r0, k1, r1 in runs:
            lo = self.gaps[r0][0] if k0 == "gap" else r0
            hi = self.gaps[r1][1] if k1 == "gap" else r1
            out.append((lo, hi, k0 == "pt", k1 == "pt"))
        return out

    def interior_set(self):
        runs = self._runs(lambda x: self.kind[x]["interior"], lambda i: self.gap_full[i])
        return (
            sympy.Union(
                *[
                    sympy.Interval.open(
                        sympy.Rational(lo.numerator, lo.denominator), sympy.Rational(hi.numerator, hi.denominator)
                    )
                    for lo, hi, _a, _b in runs
                ]
            )
            if runs
            else sympy.EmptySet
        )

    def closure_set(self):
        runs = self._runs(lambda x: self.kind[x]["closure"], lambda i: self.gap_full[i] or self.gap_dense[i])
        return (
            sympy.Union(
                *[
                    sympy.Interval(
                        sympy.Rational(lo.numerator, lo.denominator), sympy.Rational(hi.numerator, hi.denominator)
                    )
                    for lo, hi, _a, _b in runs
                ]
            )
            if runs
            else sympy.EmptySet
        )

    def boundary_points(self):
        return sorted(
            x for x, k in self.kind.items() if k["closure"] and not k["interior"] and not self._in_term_tail(x)
        )

    def _in_term_tail(self, x):
        return False

    def limit_points(self):
        return sorted(x for x, k in self.kind.items() if k["limit"])

    def closure_minus_set(self):
        return sorted(x for x, k in self.kind.items() if k["closure"] and not k["inside"])

    def isolated(self):
        return sorted(x for x, k in self.kind.items() if k["inside"] and not k["limit"])

    def boundary_length(self):
        total = Fr(0)
        for i, (lo, hi) in enumerate(self.gaps):
            if lo is not None and hi is not None and self.gap_dense[i] and not self.gap_full[i]:
                total += hi - lo
        return total


def covers_real(pieces, x: Fr) -> bool:
    return any(p["t"] == "I" and (p["a"] < x < p["b"]) for p in pieces)


def sym(x):
    x = Fr(x)
    return sympy.Rational(x.numerator, x.denominator)


def endpoints(real_set):
    """Every endpoint of the intervals that make up a SymPy subset of R."""
    pieces = real_set.args if isinstance(real_set, sympy.Union) else (real_set,)
    return sorted({Fr(int(v.p), int(v.q)) for piece in pieces for v in (piece.start, piece.end)})


def asks(text: str) -> str:
    """Which question a real-line stem asks."""
    t = text
    if "isolated" in t:
        return "iso_set"
    if "are not in S" in t or "itself does not" in t or "do not belong to S" in t:
        return "missing_set"
    if "limit points" in t:
        return "limit_set"
    if "interior" in t:
        return "interior_ends"
    if "closure" in t:
        return "closure_ends"
    if "boundary" in t:
        return "bd_set"
    raise AssertionError(t)


def real_answer(q):
    """The set of numbers the stem asks for, recomputed by the neighbourhood oracle (None for the planar stems)."""
    text = q.text
    if "R²" in text or "plane" in text:
        return None
    line = Line(read_set(text))
    return {
        "interior_ends": lambda: endpoints(line.interior_set()),
        "closure_ends": lambda: endpoints(line.closure_set()),
        "bd_set": line.boundary_points,
        "iso_set": line.isolated,
        "limit_set": line.limit_points,
        "missing_set": line.closure_minus_set,
    }[asks(text)]()


def read_items(answer: str):
    return sorted(Fr(v.strip()) for v in answer.strip("{}").split(","))


# -- the plane ------------------------------------------------------------------------------------------------------------------


def planar_predicate(text: str):
    """A function (x, y) -> bool (numpy-friendly) for the set written in the stem."""
    m = re.search(r"the square \[(-?\d+), (-?\d+)\] × \[(-?\d+), (-?\d+)\] with its boundary removed", text)
    if m:
        a, b, c, d = (int(v) for v in m.groups())
        return lambda x, y: (a < x) & (x < b) & (c < y) & (y < d)
    m = re.search(rf"S = ([\[(])({NUM}), ({NUM})([\])]) × ([\[(])({NUM}), ({NUM})([\])])", text)
    if m:
        l1, a, b, r1, l2, c, d, r2 = m.groups()
        a, b, c, d = (float(v) for v in (a, b, c, d))

        def rect(x, y):
            ok_x = ((x >= a) if l1 == "[" else (x > a)) & ((x <= b) if r1 == "]" else (x < b))
            ok_y = ((y >= c) if l2 == "[" else (y > c)) & ((y <= d) if r2 == "]" else (y < d))
            return ok_x & ok_y

        return rect
    cond = re.search(r"\{\(x, y\) : (.*?)\}", text).group(1)
    cond = cond.replace("²", "**2").replace("≤", "<=").replace("≥", ">=")
    cond = re.sub(r"(\d)([xy])", r"\1*\2", cond)
    parts = re.split(r"(<=|>=|<|>)", cond)
    operands = [parts[i].strip() for i in range(0, len(parts), 2)]
    ops = [parts[i] for i in range(1, len(parts), 2)]

    def pred(x, y):
        vals = [eval(op, {"__builtins__": {}}, {"x": x, "y": y}) for op in operands]  # noqa: S307 - the oracle reads its own stem
        result = True
        for (u, v), o in zip(zip(vals, vals[1:]), ops):
            u, v = np.broadcast_arrays(u, v)
            result = result & {"<": u < v, "<=": u <= v, ">": u > v, ">=": u >= v}[o]
        return result

    return pred


def classify_point(pred, x, y, eps=1e-3):
    """(interior, boundary, closure) of a point, by sampling a small circle of points around it."""
    inside = bool(pred(np.float64(x), np.float64(y)))
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    near = np.atleast_1d(pred(x + eps * np.cos(angles), y + eps * np.sin(angles)))
    interior = inside and bool(near.all())
    closure = inside or bool(near.any())
    return interior, closure and not interior, closure


def planar_points(text: str):
    m = re.search(r"(?:points|among|from the points) ((?:\(-?\d+, -?\d+\)(?:, )?)+)", text)
    return [tuple(int(v) for v in pt.split(", ")) for pt in re.findall(r"\((-?\d+, -?\d+)\)", m.group(1))]


def planar_answer(q):
    pred = planar_predicate(q.text)
    pts = planar_points(q.text)
    flags = {p: classify_point(pred, *p) for p in pts}
    if "interior of S" in q.text:
        return {p for p, f in flags.items() if f[0]}
    if "closure of S" in q.text:
        return {p for p, f in flags.items() if f[2]}
    return {p for p, f in flags.items() if f[1]}


def read_points(answer: str):
    return {tuple(int(v) for v in pt.split(", ")) for pt in re.findall(r"\((-?\d+, -?\d+)\)", answer)}


# -- tests -----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_key_accepted_and_distractors_rejected(tier):
    sk = skill(SKILL)
    for seed in SEEDS:
        q = tier_q(tier, seed)
        assert grade(sk, q, q.answer_text).correct, (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 2, (seed, q.text)
        for d in q.distractors:
            assert not grade(sk, q, d).correct, (seed, q.text, q.answer_text, d)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_oracle_recomputes_every_answer(tier):
    real = planar = 0
    for seed in range(150):
        q = tier_q(tier, seed)
        want = real_answer(q)
        if want is None:
            assert read_points(q.answer_text) == planar_answer(q), (seed, q.text, q.answer_text)
            planar += 1
        else:
            assert read_items(q.answer_text) == sorted(want), (seed, q.text, q.answer_text, want)
            real += 1
    assert real >= 60
    assert (planar >= 20) if tier == "hard" else planar == 0


def test_forms_all_appear():
    seen = {"easy": set(), "medium": set(), "hard": set()}
    for tier in seen:
        for seed in range(200):
            q = tier_q(tier, seed)
            seen[tier].add("planar" if "R²" in q.text or "plane" in q.text else asks(q.text))
    assert seen["easy"] == {"interior_ends", "closure_ends", "bd_set"}
    assert seen["medium"] == {"interior_ends", "closure_ends", "bd_set", "iso_set"}
    assert seen["hard"] == {"planar", "limit_set", "missing_set", "interior_ends", "closure_ends"}


def test_traps_are_present():
    """Touching intervals merge or stay apart, rational pieces drop out of the interior, a sequence limit can be missing."""
    merge = apart = rational_slip = missing_limit = planar_boundary_centre = 0
    for seed in range(200):
        q = tier_q("medium", seed)
        if asks(q.text) in ("interior_ends", "closure_ends", "bd_set"):
            ends = [(p["a"], p["b"], p["l"], p["r"]) for p in read_set(q.text) if p["t"] == "I"]
            for a1, b1, l1, r1 in ends:
                for a2, b2, l2, r2 in ends:
                    if b1 == a2:
                        merge += r1 or l2
                        apart += not (r1 or l2)
    for seed in range(300):
        q = tier_q("hard", seed)
        if "R²" in q.text or "plane" in q.text:
            planar_boundary_centre += "0 <" in q.text
            continue
        pieces = read_set(q.text)
        if asks(q.text) == "interior_ends" and any(p["t"] == "Q" for p in pieces):
            rational_slip += 1
        if asks(q.text) == "missing_set" and any(p["t"] == "S" for p in pieces):
            missing_limit += 1
    assert merge >= 5 and apart >= 5, (merge, apart)
    assert rational_slip >= 5 and missing_limit >= 8 and planar_boundary_centre >= 2


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_variety_and_shapes(tier):
    qs = [tier_q(tier, s) for s in SEEDS]
    assert len({prompt_shape(q) for q in qs}) >= 3
    answers = {q.answer_text for q in qs}
    assert len(answers) >= 6, answers
    top = max(sum(1 for q in qs if q.answer_text == a) for a in answers)
    assert top <= 0.4 * len(qs), (tier, top)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_deterministic_and_no_text_defects(tier):
    for seed in (0, 1, 7, 123):
        assert tier_q(tier, seed).to_json() == tier_q(tier, seed).to_json()
    for seed in SEEDS:
        q = tier_q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)
        assert "*" not in " ".join([q.text, *(p.prompt for p in q.parts)])


def test_unknown_tier_is_rejected():
    for topic in (TOPIC, EXAM):
        with pytest.raises(Exception):
            make(topic, seed=1, tier="nonsense")


def test_wired():
    sk = skill(SKILL)
    assert sk.domain == "metric-spaces-and-topology"
    assert {s.difficulty for s in sk.sources} == {"easy", "medium", "hard", "exam"}
    assert {s.topic for s in sk.sources if s.difficulty != "exam"} == {TOPIC}
    check_wired(SKILL, EXAM)


# -- the exam chain ----------------------------------------------------------------------------------------------------------------


def exam_oracle(q):
    line = Line(read_set(q.text))
    boundary_flavour = "boundary" in q.parts[1].prompt
    pts = line.boundary_points() if boundary_flavour else line.closure_minus_set()
    return line, pts


def test_exam_chain_matches_oracle():
    flavours = set()
    for seed in range(300):
        q = tier_q("exam", seed)
        assert [p.label for p in q.parts] == list("abcde")
        line, pts = exam_oracle(q)
        flavours.add("boundary" in q.parts[1].prompt)
        parts = {p.label: p for p in q.parts}
        read = read_solution_set(parts["a"].answer_text)
        assert read[0] == line.interior_set(), (seed, q.text, parts["a"].answer_text)
        assert sorted(Fr(v.strip()) for v in parts["b"].answer_text.strip("{}").split(",")) == sorted(pts), seed
        assert int(parts["c"].answer_text) == len(pts), seed
        assert int(parts["d"].answer_text) == sum(pts), seed
        assert Fr(parts["e"].answer_text) == Fr(sum(pts), len(pts)), seed
        assert parts["e"].follow_through == "part_d/part_c"
    assert flavours == {True, False}


def test_exam_keys_accepted_distractors_rejected():
    sk = skill(SKILL)
    for seed in range(120):
        q = tier_q("exam", seed)
        result = grade(sk, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(p.marks for p in q.parts), (seed, q.text)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                part_result = grade(sk, q, {**canon, p.label: d}).parts[p.label]
                assert part_result.marks == 0, (seed, p.label, d, p.answer_text)


def test_exam_follow_through_from_wrong_count_and_sum():
    """A wrong count (c) and sum (d) earn nothing, and the mean (e) worked out from them still earns full marks."""
    sk = skill(SKILL)
    checked = 0
    for seed in range(120):
        q = tier_q("exam", seed)
        canon = {p.label: p.answer_text for p in q.parts}
        e_part = next(p for p in q.parts if p.label == "e")
        n, total = int(canon["c"]), int(canon["d"])
        # a wrong chain whose mean is a new number (not the key, not a repeat of the replies, not a listed slip)
        for wrong_n, wrong_total in ((n + 1, total + 2), (n + 2, total + 1), (n + 3, total - 2), (n + 1, total - 3)):
            mean = Fr(wrong_total, wrong_n)
            if mean != Fr(total, n) and mean != 0 and len({wrong_n, wrong_total, mean}) == 3:
                if str(mean) not in {str(d) for d in e_part.distractors}:
                    break
        else:
            raise AssertionError(f"no clean wrong chain for seed {seed}")
        reply = dict(canon, c=str(wrong_n), d=str(wrong_total), e=str(mean))
        result = grade(sk, q, reply)
        assert result.parts["c"].marks == 0 and result.parts["d"].marks == 0, (seed, reply)
        assert result.parts["e"].marks == e_part.marks, (seed, reply, result.parts["e"])
        assert result.parts["e"].follow_through is True, seed
        checked += 1
    assert checked == 120, checked


def test_exam_answer_space_and_junk():
    answers = {tier_q("exam", s).answer_text for s in range(300)}
    assert len(answers) >= 100, len(answers)
    assert len({prompt_shape(tier_q("exam", s)) for s in range(60)}) >= 3
    for label in "abcde":
        values = [p.answer_text for s in range(300) for p in tier_q("exam", s).parts if p.label == label]
        assert max(values.count(v) for v in set(values)) <= 0.5 * len(values), label
    check_deterministic(EXAM)
    check_all_wrong_earns_nothing(SKILL, EXAM, {"a": "(100, 101)", "b": "{100}", "c": "97", "d": "98", "e": "99"})


def test_interior_stems_do_not_claim_separated_intervals():
    """Intervals of the interior can share an endpoint that S misses (S = (-4,-1) ∪ (-1,2]), so a stem must not call them
    "touching"-free; the oracle's open pieces are listed and the key is every endpoint of the maximal open intervals."""
    shared = 0
    for tier in ("medium", "hard", "easy"):
        for seed in range(200):
            q = tier_q(tier, seed)
            if "R²" in q.text or "plane" in q.text or asks(q.text) != "interior_ends":
                continue
            assert "touching" not in q.text and "disjoint" not in q.text, (seed, q.text)
            assert "interval" in q.text
            interior = Line(read_set(q.text)).interior_set()
            pieces = interior.args if isinstance(interior, sympy.Union) else (interior,)
            ends = [(Fr(int(p.start.p), int(p.start.q)), Fr(int(p.end.p), int(p.end.q))) for p in pieces]
            if any(a[1] == b[0] for a, b in zip(sorted(ends), sorted(ends)[1:])):
                shared += 1
                assert read_items(q.answer_text) == endpoints(interior), (seed, q.text, q.answer_text)
    assert shared >= 5, shared


def test_every_declared_misconception_is_emitted_by_some_distractor():
    """No tag in the skill file is dead: each one is attached to a distractor somewhere across the tiers and the exam."""
    import yaml
    from pathlib import Path

    declared = {
        m["id"]
        for m in yaml.safe_load(
            Path("skills/metric-spaces-and-topology/uc2-interior-closure-boundary.yaml").read_text(encoding="utf-8")
        )["misconceptions"]
    }
    seen: set = set()

    def scan(item):
        seen.update(item.option_notes.values())
        for part in getattr(item, "parts", None) or []:
            scan(part)

    for tier in ("easy", "medium", "hard"):
        for seed in range(600):
            scan(tier_q(tier, seed))
    for seed in range(300):
        scan(tier_q("exam", seed))
    assert declared - seen == set(), sorted(declared - seen)


def test_planar_punctured_centre_slip_drops_the_centre():
    hits = 0
    for seed in range(600):
        q = tier_q("hard", seed)
        for text, note in q.option_notes.items():
            if note == "ic-punctured-centre-not-boundary":
                hits += 1
                key = set(read_points(q.answer_text))
                wrong = set(read_points(text))
                assert wrong < key and len(key - wrong) == 1, (seed, q.text, q.answer_text, text)
    assert hits >= 5, hits
