"""Deliberately does NOT import anything from
``mathgen.generators.p19i_timetable_build`` -- it parses ``question.text``
with its own regexes and recomputes the answer from scratch, then compares
against ``question.answer_text``.
"""

import re
from types import SimpleNamespace

import pytest

from mathgen import make

TOPIC_A = "p19i_timetable_latest_service"
TOPIC_B = "p19i_timetable_condition_lookup"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)


# =========================================================================
# Skill A oracle: latest service meeting a deadline
# =========================================================================

_A_ROW_RE = re.compile(r"(Bus|Train|Boat) (\d+) departs (\w+) at (\d{2}:\d{2}) and arrives at (\w+) at (\d{2}:\d{2})\.")
_A_DEADLINE_RE = re.compile(r"is the latest service that still reaches \w+ by (\d{2}:\d{2})\?")
_A_CANCEL_RE = re.compile(
    r"(Bus|Train|Boat) services are (?:cancelled today|not running today)|"
    r"Only (Bus|Train|Boat) services are running today"
)


def _to_min(hh_mm: str) -> int:
    h, m = hh_mm.split(":")
    return int(h) * 60 + int(m)


def _oracle_a(text: str) -> str:
    rows = _A_ROW_RE.findall(text)
    assert rows, text
    deadline = _to_min(_A_DEADLINE_RE.search(text).group(1))

    disqualified = None
    m = _A_CANCEL_RE.search(text)
    if m:
        disqualified = m.group(1) or m.group(2)
        if "Only" in m.group(0):
            # "Only <mode> services are running today" -- every OTHER mode
            # present in the table is disqualified. With exactly two modes
            # in the table, that is the one mode that is not <mode>.
            kept = disqualified
            modes_in_table = {r[0] for r in rows}
            others = modes_in_table - {kept}
            disqualified = next(iter(others)) if others else None
        # else: "<mode> services are cancelled/not running today" -- disqualified is that mode.

    candidates = []
    for mode, _num, _a, dep, _b, arr in rows:
        if disqualified is not None and mode == disqualified:
            continue
        if _to_min(arr) <= deadline:
            candidates.append(_to_min(dep))
    assert candidates, text
    best = max(candidates)
    return f"{best // 60:02d}:{best % 60:02d}"


def _challenge_a(question) -> bool:
    return _oracle_a(question.text) != question.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_oracle_agrees_with_the_shipped_answer(difficulty):
    checked, challenged = 0, []
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        checked += 1
        if _challenge_a(q):
            challenged.append((seed, q.text, q.answer_text, _oracle_a(q.text)))
    assert checked == len(SEEDS)
    assert not challenged, (
        f"skill A {difficulty}: {len(challenged)}/{checked} disagree with the oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_oracle_fires_on_a_corrupted_answer(difficulty):
    real = make(TOPIC_A, seed=0, difficulty=difficulty)
    correct = SimpleNamespace(text=real.text, answer_text=real.answer_text)
    corrupted = SimpleNamespace(text=real.text, answer_text=real.distractors[0])
    assert not _challenge_a(correct)
    assert _challenge_a(corrupted)


def test_skill_a_oracle_disagreement_is_visible_on_a_deliberately_broken_question():
    """Corrupting the answer row's own stated arrival time to one far past
    the deadline makes the oracle stop picking it."""
    q = make(TOPIC_A, seed=1, difficulty="easy")
    original = _oracle_a(q.text)
    ans_row = re.search(
        rf"(Bus|Train|Boat) \d+ departs \w+ at {re.escape(q.answer_text)} and arrives at \w+ at (\d{{2}}:\d{{2}})\.",
        q.text,
    )
    assert ans_row is not None
    arr = ans_row.group(2)
    broken_text = q.text.replace(arr, "23:59", 1)
    assert _oracle_a(broken_text) != original


# =========================================================================
# Skill B oracle: table-condition lookup
# =========================================================================

_TIDE_ROW_RE = re.compile(r"(\w+|The \d+\w+): high tide at (\d{2}:\d{2}), height (\d+\.\d) m\.")
_SUNRISE_ROW_RE = re.compile(r"(\w+|The \d+\w+): sunrise at (\d{2}:\d{2})\.")
_MOON_ROW_RE = re.compile(
    r"(The \d+\w+): (new moon|waxing crescent|first quarter|waxing gibbous|full moon|"
    r"waning gibbous|last quarter|waning crescent)\."
)

_TIDE_SIMPLE_Q_RE = re.compile(r"Which day has the (highest|lowest) high tide\?")
_TIDE_WEEKEND_Q_RE = re.compile(r"Which day has the highest high tide that is NOT a weekend day\?")
_SUNRISE_SIMPLE_Q_RE = re.compile(r"Which day has the (latest|earliest) sunrise\?")
_SUNRISE_SECOND_Q_RE = re.compile(r"Which day has the second (latest|earliest) sunrise\?")
_MOON_AFTER_FULL_Q_RE = re.compile(r"Which day is (\d+) days? after the first day showing a full moon\?")
_MOON_BEFORE_FULL_Q_RE = re.compile(r"Which day is (\d+) days? before the first day showing a full moon\?")
_MOON_GIBBOUS_Q_RE = re.compile(r"Which is the first day after the (\d+)\w+ showing a waning gibbous moon\?")
_SUNRISE_DATE_Q_RE = re.compile(r"Which day has the latest sunrise among the days after the 3rd\?")
_MOON_Q_RE = re.compile(r"Which is the first full moon after the (\d+)\w+\?")

_WEEKEND = {"Saturday", "Sunday"}


def _oracle_b(text: str) -> str:
    # The extra moon-phase hard-tier question kinds
    mm = _MOON_AFTER_FULL_Q_RE.search(text) or _MOON_BEFORE_FULL_Q_RE.search(text) or _MOON_GIBBOUS_Q_RE.search(text)
    if mm:
        rows = _MOON_ROW_RE.findall(text)
        nums = [int(re.search(r"\d+", d).group()) for d, _ in rows]
        phases = [ph for _, ph in rows]
        label = dict(zip(nums, [d for d, _ in rows]))
        if _MOON_GIBBOUS_Q_RE.search(text):
            ref = int(mm.group(1))
            return label[min(n for n, ph in zip(nums, phases) if n > ref and ph == "waning gibbous")]
        first_full = min(n for n, ph in zip(nums, phases) if ph == "full moon")
        step = int(mm.group(1)) * (1 if _MOON_AFTER_FULL_Q_RE.search(text) else -1)
        return label[first_full + step]

    m = _SUNRISE_SECOND_Q_RE.search(text)
    if m:
        rows = _SUNRISE_ROW_RE.findall(text)
        ordered = sorted(rows, key=lambda r: _to_min(r[1]), reverse=(m.group(1) == "latest"))
        return ordered[1][0]

    m = _MOON_Q_RE.search(text)
    if m:
        cutoff = int(m.group(1))
        rows = _MOON_ROW_RE.findall(text)
        assert rows, text
        day_nums = [int(re.search(r"\d+", d).group()) for d, _ in rows]
        after = sorted((d for d, phase in zip(day_nums, [p for _, p in rows]) if phase == "full moon" and d > cutoff))
        assert after, text
        target = after[0]
        idx = day_nums.index(target)
        return rows[idx][0]

    m = _TIDE_WEEKEND_Q_RE.search(text)
    if m:
        rows = _TIDE_ROW_RE.findall(text)
        assert rows, text
        candidates = [(day, float(h)) for day, _t, h in rows if day not in _WEEKEND]
        assert candidates, text
        return max(candidates, key=lambda x: x[1])[0]

    m = _SUNRISE_DATE_Q_RE.search(text)
    if m:
        rows = _SUNRISE_ROW_RE.findall(text)
        assert rows, text
        candidates = [(day, t) for day, t in rows if int(re.search(r"\d+", day).group()) > 3]
        assert candidates, text
        return max(candidates, key=lambda x: _to_min(x[1]))[0]

    m = _TIDE_SIMPLE_Q_RE.search(text)
    if m:
        direction = m.group(1)
        rows = _TIDE_ROW_RE.findall(text)
        assert rows, text
        values = [(day, float(h)) for day, _t, h in rows]
        key = max if direction == "highest" else min
        return key(values, key=lambda x: x[1])[0]

    m = _SUNRISE_SIMPLE_Q_RE.search(text)
    if m:
        direction = m.group(1)
        rows = _SUNRISE_ROW_RE.findall(text)
        assert rows, text
        values = [(day, _to_min(t)) for day, t in rows]
        key = max if direction == "latest" else min
        return key(values, key=lambda x: x[1])[0]

    raise AssertionError(f"no recognised question pattern in: {text!r}")


def _challenge_b(question) -> bool:
    return _oracle_b(question.text) != question.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_b_oracle_agrees_with_the_shipped_answer(difficulty):
    checked, challenged = 0, []
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        checked += 1
        if _challenge_b(q):
            challenged.append((seed, q.text, q.answer_text, _oracle_b(q.text)))
    assert checked == len(SEEDS)
    assert not challenged, (
        f"skill B {difficulty}: {len(challenged)}/{checked} disagree with the oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_b_oracle_fires_on_a_corrupted_answer(difficulty):
    real = make(TOPIC_B, seed=0, difficulty=difficulty)
    correct = SimpleNamespace(text=real.text, answer_text=real.answer_text)
    corrupted = SimpleNamespace(text=real.text, answer_text=real.distractors[0])
    assert not _challenge_b(correct)
    assert _challenge_b(corrupted)


def test_skill_b_oracle_disagreement_is_visible_on_a_deliberately_broken_question():
    """Corrupt the answer day's own stated value to the OPPOSITE extreme
    (a huge height if the condition wants "highest"/"lowest", a very early
    or very late time otherwise), so the oracle stops picking it."""
    q = make(TOPIC_B, seed=1, difficulty="easy")
    original = _oracle_b(q.text)
    # The answer currently holds the extreme the question asks for; push it
    # to the OPPOSITE extreme so it can no longer hold that title.
    wants_low = "lowest" in q.text or "earliest" in q.text
    if _TIDE_ROW_RE.search(q.text):
        day, t, h = next(r for r in _TIDE_ROW_RE.findall(q.text) if r[0] == q.answer_text)
        replacement = "9.9" if wants_low else "0.1"
        broken_text = q.text.replace(f"height {h} m", f"height {replacement} m", 1)
    else:
        day, t = next(r for r in _SUNRISE_ROW_RE.findall(q.text) if r[0] == q.answer_text)
        replacement = "23:59" if wants_low else "00:01"
        broken_text = q.text.replace(f"sunrise at {t}", f"sunrise at {replacement}", 1)
    assert _oracle_b(broken_text) != original
