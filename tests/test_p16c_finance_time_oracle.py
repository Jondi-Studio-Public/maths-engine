"""Independent oracle for the `p16c_finance_time` questions: it re-derives each answer from the rendered question
text.

Parses the RENDERED question text only (``question.text``/``question.answer_text``)
and imports NOTHING from ``mathgen.generators.p16c_finance_time_build``. Every route
below is re-derived from the stated text with ``Fraction`` arithmetic (or, for the
GMT skill, plain integer minute-of-week arithmetic), never by reusing the generator's
own construction values.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import round_half_up

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS_1000 = range(1000)

LOAN_SKILL = "senior-use-technology-and-a-recurrence-relation-to-model"
EFFECT_SKILL = "senior-use-technology-to-investigate-the-effect-of-the"
WAGE_SKILL = "senior-use-a-spreadsheet-to-display-examples-of-the"
GMT_SKILL = "senior-solve-problems-involving-greenwich-mean-time-and-the"


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    if src.topic == "exam_parts_compound_interest":
        # This skill's exam source is now the multi-part chain (tests/test_exam_parts_compound_interest.py);
        # these checks keep covering the single-answer exam preset it replaced, which the generator still has.
        return make("p16c_compound_effect", seed=seed, difficulty="exam", variant="percent_effect")
    if src.topic == "exd_x_r2d_loan_recurrence_chain":
        # This skill's exam source is now a multi-part chain (tests/test_exd_x_r2d_loan_recurrence_chain.py);
        # these checks keep covering the single-answer exam variant it replaced, which the generator still has.
        return make("p16c_loan_recurrence_balance", seed=seed, difficulty="exam", variant="rate_change")
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _money_to_fraction(s: str) -> Fraction:
    neg = s.startswith("-")
    s = s.lstrip("-")
    whole, _, frac = s.partition(".")
    val = Fraction(int(whole)) + (Fraction(int(frac), 10 ** len(frac)) if frac else 0)
    return -val if neg else val


# =============================================================================
# Skill 1: p16c_loan_recurrence_balance
# =============================================================================

_L_MONEY_RE = re.compile(r"\$(\d+(?:\.\d+)?)")
_L_RATE_RE = re.compile(r"(\d+(?:\.\d+)?)%")
_L_EXTRA_PRESENT_RE = re.compile(r"extra one-off repayment")
_L_EXTRA_PERIOD_RE = re.compile(r"[Ii]n period (\d+),")
_L_N_PERIODS_RE = re.compile(r"after (\d+) periods")


def _loan_iterate(principal, r, R, n, extra_at=None, extra=Fraction(0)):
    b = Fraction(principal)
    for i in range(1, n + 1):
        pay = R + (extra if i == extra_at else 0)
        b = b * (1 + r) - pay
    return b


def _loan_oracle_forward_or_lump(text: str) -> Fraction:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # stems write $20,000
    # Dollar amounts always appear, in order, as: principal, regular
    # repayment, then (only when a lump sum is stated) the extra amount.
    amounts = [Fraction(a) for a in _L_MONEY_RE.findall(text)]
    principal, R = amounts[0], amounts[1]
    rate = Fraction(_L_RATE_RE.search(text).group(1)) / 100
    n = int(_L_N_PERIODS_RE.search(text).group(1))
    if _L_EXTRA_PRESENT_RE.search(text):
        extra = amounts[2]
        extra_at = int(_L_EXTRA_PERIOD_RE.search(text).group(1))
    else:
        extra, extra_at = Fraction(0), None
    return _loan_iterate(principal, rate, R, n, extra_at, extra)


def _loan_oracle_find_n(text: str) -> int:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # stems write $20,000
    # Dollar amounts always appear, in order: principal, regular repayment,
    # target balance.
    amounts = [Fraction(a) for a in _L_MONEY_RE.findall(text)]
    principal, R, target = amounts[0], amounts[1], amounts[2]
    rate = Fraction(_L_RATE_RE.search(text).group(1)) / 100
    n, b = 0, principal
    while b > target:
        b = b * (1 + rate) - R
        n += 1
    return n


_L_RC_FIRST_N_RE = re.compile(r"first (\d+) periods")
_L_RC_TOTAL_N_RE = re.compile(r"after (\d+) periods")


def _loan_oracle_rate_change(text: str) -> Fraction:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # stems write $20,000
    # Dollar amounts always appear, in order: principal, then the (repeated)
    # regular repayment -- take the first two.
    amounts = [Fraction(a) for a in _L_MONEY_RE.findall(text)]
    principal, R = amounts[0], amounts[1]
    rates = [Fraction(x) for x in _L_RATE_RE.findall(text)]
    r1, r2 = rates[0] / 100, rates[1] / 100
    n1 = int(_L_RC_FIRST_N_RE.search(text).group(1))
    ntotal = int(_L_RC_TOTAL_N_RE.search(text).group(1))
    n2 = ntotal - n1
    b = principal
    for _ in range(n1):
        b = b * (1 + r1) - R
    for _ in range(n2):
        b = b * (1 + r2) - R
    return b


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_loan_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(LOAN_SKILL, difficulty, seed)
        if difficulty in ("easy", "medium"):
            expected = round_half_up(_loan_oracle_forward_or_lump(q.text), 2)
            assert expected == _money_to_fraction(q.answer_text), (difficulty, seed, q.text, q.answer_text)
        elif difficulty == "hard":
            expected_n = _loan_oracle_find_n(q.text)
            assert str(expected_n) == q.answer_text, (difficulty, seed, q.text, q.answer_text)
        else:
            expected = round_half_up(_loan_oracle_rate_change(q.text), 2)
            assert expected == _money_to_fraction(q.answer_text), (difficulty, seed, q.text, q.answer_text)
        checked += 1
    assert checked == 1000


# =============================================================================
# Skill 2: p16c_compound_effect
# =============================================================================

_CE_MONEY_RE = re.compile(r"\$(\d+)")
_CE_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)%")
_CE_YEARS_RE = re.compile(r"(\d+) years")
_CE_FREQ_MARKERS = (("semi-annually", 2), ("quarterly", 4), ("monthly", 12), ("annually", 1))
_CE_FREQ_RATE_RE = re.compile(r"compounded (\S+) at (?:a nominal )?(\d+)% p\.a\.")


def _ce_fv(principal, rate_pct, freq, years) -> Fraction:
    r = Fraction(rate_pct) / 100
    return Fraction(principal) * (1 + r / freq) ** (freq * years)


def _ce_freq_positions(text: str):
    masked = text
    spans = []
    for name, freq in _CE_FREQ_MARKERS:
        for m in re.finditer(re.escape(name), masked):
            spans.append((m.start(), freq))
        masked = masked.replace(name, "#" * len(name))
    spans.sort()
    return [f for _, f in spans]


def _ce_oracle_freq_diff(text: str) -> Fraction:
    principal = Fraction(_CE_MONEY_RE.search(text).group(1))
    rate = Fraction(_CE_PCT_RE.search(text).group(1))
    years = int(_CE_YEARS_RE.search(text).group(1))
    freqs = _ce_freq_positions(text)
    assert len(freqs) == 2, text
    fvs = [_ce_fv(principal, rate, f, years) for f in freqs]
    return abs(fvs[0] - fvs[1])


def _ce_oracle_rate_diff(text: str) -> Fraction:
    principal = Fraction(_CE_MONEY_RE.search(text).group(1))
    years = int(_CE_YEARS_RE.search(text).group(1))
    freqs = _ce_freq_positions(text)
    assert len(freqs) == 1, text
    freq = freqs[0]
    rates = [Fraction(m) for m in re.findall(r"(\d+)%", text)]
    assert len(rates) == 2, text
    fvs = [_ce_fv(principal, r, freq, years) for r in rates]
    return abs(fvs[0] - fvs[1])


def _ce_oracle_percent_effect(text: str) -> Fraction:
    principal = Fraction(_CE_MONEY_RE.search(text).group(1))
    years = int(_CE_YEARS_RE.search(text).group(1))
    pairs = _CE_FREQ_RATE_RE.findall(text)
    assert len(pairs) == 2, text
    freq_by_name = dict(_CE_FREQ_MARKERS)
    fvs = [_ce_fv(principal, Fraction(rate), freq_by_name[name], years) for name, rate in pairs]
    larger, smaller = max(fvs), min(fvs)
    return (larger - smaller) / smaller * 100


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_compound_effect_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(EFFECT_SKILL, difficulty, seed)
        if difficulty in ("easy", "medium"):
            expected = round_half_up(_ce_oracle_freq_diff(q.text), 2)
            assert expected == _money_to_fraction(q.answer_text), (difficulty, seed, q.text, q.answer_text)
        elif difficulty == "hard":
            expected = round_half_up(_ce_oracle_rate_diff(q.text), 2)
            assert expected == _money_to_fraction(q.answer_text), (difficulty, seed, q.text, q.answer_text)
        else:
            expected = round_half_up(_ce_oracle_percent_effect(q.text), 1)
            assert expected == Fraction(q.answer_text), (difficulty, seed, q.text, q.answer_text)
        checked += 1
    assert checked == 1000


# =============================================================================
# Skill 3: p16c_worker_wage_table
# =============================================================================

_W_ROW_RE = re.compile(r"(\w+): \$(\d+)/h, (\d+) ordinary hours(?:, (\d+) overtime hours(?: at (1\.5|2)x)?)?")
_W_WHO_SINGLE_RE = re.compile(r"(?:What is|Find) (\w+)'s|calculate the total amount (\w+) earned")
_W_WHO_DIFF_RE = re.compile(r"difference between (\w+)'s wage and (\w+)'s wage|MORE did (\w+) earn than (\w+)")


def _w_parse_rows(text: str) -> dict:
    rows = {}
    for name, rate, hours, ot, mult in _W_ROW_RE.findall(text):
        ot = int(ot) if ot else 0
        multiplier = Fraction(2, 1) if mult == "2" else Fraction(3, 2)
        rows[name] = dict(rate=int(rate), hours=int(hours), ot=ot, mult=multiplier)
    return rows


def _w_wage(row) -> Fraction:
    return Fraction(row["rate"]) * row["hours"] + Fraction(row["rate"]) * row["mult"] * row["ot"]


def _wage_oracle(text: str) -> Fraction:
    rows = _w_parse_rows(text)
    diff_match = _W_WHO_DIFF_RE.search(text)
    if diff_match:
        g1, g2, g3, g4 = diff_match.groups()
        a, b = (g1, g2) if g1 else (g3, g4)
        return abs(_w_wage(rows[a]) - _w_wage(rows[b]))
    single_match = _W_WHO_SINGLE_RE.search(text)
    who = next(g for g in single_match.groups() if g)
    return _w_wage(rows[who])


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_wage_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(WAGE_SKILL, difficulty, seed)
        expected = round_half_up(_wage_oracle(q.text), 2)
        assert expected == _money_to_fraction(q.answer_text), (difficulty, seed, q.text, q.answer_text)
        checked += 1
    assert checked == 1000


# =============================================================================
# Skill 4: p16c_gmt_dateline (mcq)
# =============================================================================

_G_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_G_DEPART_RE = re.compile(r"departs \S.*?\(GMT([+-]\d+(?::\d+)?)\) at (\d{2}):(\d{2}) on (\w+)")
_G_LEG1_RE = re.compile(r"flies(?: for)? (\d+) hours to \S.*?\(GMT([+-]\d+(?::\d+)?)\)")
_G_LAYOVER_RE = re.compile(r"After a (\d+)-hour layover")
_G_LEG2_RE = re.compile(r"connecting flight takes (\d+) hours to \S.*?\(GMT([+-]\d+(?::\d+)?)\)")


def _g_parse_offset(s: str) -> Fraction:
    sign = 1 if s[0] == "+" else -1
    rest = s[1:]
    if ":" in rest:
        h, m = rest.split(":")
        return sign * (Fraction(int(h)) + Fraction(int(m), 60))
    return sign * Fraction(int(rest))


def _g_to_minutes(day_idx, hour, minute):
    return day_idx * 1440 + hour * 60 + minute


def _g_offset_minutes(offset: Fraction) -> int:
    return int(offset * 60)


def _g_fmt_time(total_minutes: int) -> str:
    day_idx = (total_minutes // 1440) % 7
    minute_of_day = total_minutes % 1440
    h, m = divmod(minute_of_day, 60)
    return f"{_G_DAYS[day_idx]} {h:02d}:{m:02d}"


def _gmt_oracle(text: str) -> str:
    off_from_s, hh, mm, day_name = _G_DEPART_RE.search(text).groups()
    day_idx = _G_DAYS.index(day_name)
    off_from = _g_parse_offset(off_from_s)
    depart_minutes = _g_to_minutes(day_idx, int(hh), int(mm))

    dur1, off_to1_s = _G_LEG1_RE.search(text).groups()
    off_to1 = _g_parse_offset(off_to1_s)
    utc1 = depart_minutes - _g_offset_minutes(off_from) + int(dur1) * 60
    arrive1 = utc1 + _g_offset_minutes(off_to1)

    layover_match = _G_LAYOVER_RE.search(text)
    if not layover_match:
        return _g_fmt_time(arrive1)

    layover = int(layover_match.group(1))
    dur2, off_to2_s = _G_LEG2_RE.search(text).groups()
    off_to2 = _g_parse_offset(off_to2_s)
    depart2_local = arrive1 + layover * 60
    depart2_utc = depart2_local - _g_offset_minutes(off_to1)
    utc2 = depart2_utc + int(dur2) * 60
    arrive2 = utc2 + _g_offset_minutes(off_to2)
    return _g_fmt_time(arrive2)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_gmt_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(GMT_SKILL, difficulty, seed)
        expected = _gmt_oracle(q.text)
        assert expected == q.answer_text, (difficulty, seed, q.text, q.answer_text)
        checked += 1
    assert checked == 1000
