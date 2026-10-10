"""Calls `mathgen.make` only and reads `question.text`/`question.answer_text`/
`question.distractors` -- it never imports anything from
`mathgen.generators.p19c_independence_language_build`. Both oracles parse
the rendered question with their OWN regex/keyword logic and re-derive the
answer independently, over >= 1000 seeds per tier.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

TIERS = ["easy", "medium", "hard", "exam"]

# --- Independence oracle ----------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_fraction(text: str) -> Fraction:
    return Fraction(text).limit_denominator(10**9) if "." not in text else Fraction(text)


def _derive_independence(text: str) -> tuple[str, str]:
    """Parse P(A), P(B) and the third given value from `text` with this
    oracle's own logic (never the generator's), independently test
    P(A and B) == P(A) x P(B), and return (verdict, reason_text) where
    reason_text is used only for diagnostics."""
    # Hard-tier counts variant: "Of N people surveyed: A have property A,
    # B have property B, and AB have both A and B."
    counts_m = re.search(
        r"Of (\d+) people surveyed: (\d+) have property A, (\d+) have property B, "
        r"and (\d+) have both A and B",
        text,
    )
    if counts_m:
        n, a, b, ab = (int(g) for g in counts_m.groups())
        pa, pb = Fraction(a, n), Fraction(b, n)
        true_product = pa * pb
        and_value = Fraction(ab, n)
        verdict = "Independent" if and_value == true_product else "not_independent"
        return verdict, f"and={and_value} product={true_product}"

    # Hard-tier percentage variant: "P(A) = 30%. P(B) = 40%. P(A and B) = 12%."
    pct_m = re.search(r"P\(A\) = (\d+)%\. P\(B\) = (\d+)%\. P\(A and B\) = (\d+)%\.", text)
    if pct_m:
        pa_pct, pb_pct, and_pct = (int(g) for g in pct_m.groups())
        pa, pb = Fraction(pa_pct, 100), Fraction(pb_pct, 100)
        true_product = pa * pb
        and_value = Fraction(and_pct, 100)
        verdict = "Independent" if and_value == true_product else "not_independent"
        return verdict, f"and={and_value} product={true_product}"

    pa_match = re.search(r"P\(A\) = (-?\d+(?:\.\d+)?)\.", text)
    pb_match = re.search(r"P\(B\) = (-?\d+(?:\.\d+)?)\.", text)
    assert pa_match and pb_match, f"could not parse P(A)/P(B) from: {text!r}"
    pa, pb = Fraction(pa_match.group(1)), Fraction(pb_match.group(1))
    true_product = pa * pb

    # Direct P(A and B) given (easy / exam).
    direct = re.search(r"P\(A and B\) = (-?\d+(?:\.\d+)?)\.", text)
    if direct:
        and_value = Fraction(direct.group(1))
        verdict = "Independent" if and_value == true_product else "not_independent"
        return verdict, f"and={and_value} product={true_product}"

    # Conditional-probability variant (medium): recover P(A and B) = P(A|B) x P(B).
    cond = re.search(r"P\(A\|B\) = (-?\d+(?:\.\d+)?) ", text)
    assert cond, f"could not parse any given third value from: {text!r}"
    p_cond = Fraction(cond.group(1))
    and_value = p_cond * pb
    verdict = "Independent" if and_value == true_product else "not_independent"
    return verdict, f"and={and_value} product={true_product}"


def test_independence_oracle_matches_over_1000_seeds_per_tier():
    total = 0
    verdict_counts = {tier: {"Independent": 0, "not_independent": 0} for tier in TIERS}
    for tier in TIERS:
        for seed in range(1000):
            q = make("p19c_independence_verdict", seed=seed, difficulty=tier)
            verdict, note = _derive_independence(q.text)
            is_independent = verdict == "Independent"
            answer_is_independent = q.answer_text == "Independent"
            assert is_independent == answer_is_independent, (
                f"tier={tier} seed={seed}: oracle derived independent={is_independent} "
                f"({note}) but answer_text={q.answer_text!r}\ntext: {q.text}"
            )
            verdict_counts[tier][verdict] += 1
            total += 1
    assert total == 4000
    print("independence verdict counts per tier (1000 seeds each):", verdict_counts)


def test_independence_wrong_options_self_consistent_except_deliberate_miscomputation():
    """Every 'not independent' option's stated P(A and B) must equal the
    question's own given third value (recovered, for the P(A|B) convention) --
    only the stated P(A) x P(B) may differ, and only by construction."""
    opt_re = re.compile(
        r"Not independent -- P\(A and B\) = (-?\d+(?:\.\d+)?) but "
        r"P\(A\) x P\(B\) = (-?\d+(?:\.\d+)?)"
    )
    for tier in TIERS:
        for seed in range(200):
            q = make("p19c_independence_verdict", seed=seed, difficulty=tier)
            _, note = _derive_independence(q.text)
            stated_and = Fraction(note.split("and=")[1].split(" ")[0])
            for option in [q.answer_text] + q.distractors:
                m = opt_re.match(option)
                if not m:
                    assert option == "Independent"
                    continue
                option_and = Fraction(m.group(1))
                assert option_and == stated_and, (
                    f"tier={tier} seed={seed}: option {option!r} states a P(A and B) "
                    f"that does not match the question's own given value {stated_and}"
                )


# --- Probability-language oracle --------------------------------------------

_RANGE_RE = re.compile(
    r"'possible' for 0 < P <= (0\.\d+), 'probable' for \1 < P <= (0\.\d+), "
    r"'likely' for \2 < P < (0\.\d+), and 'certain' for P >= \3"
)


def _derive_language(text: str) -> str:
    m = _RANGE_RE.search(text)
    assert m, f"could not parse the stated four-range convention from: {text!r}"
    b1, b2, b3 = (Fraction(g) for g in m.groups())

    direct = re.search(r"P = (-?\d+(?:\.\d+)?)", text)
    if direct:
        value = Fraction(direct.group(1))
    else:
        pct = re.search(r"a probability of (\d+)%", text)
        if pct:
            value = Fraction(int(pct.group(1)), 100)
        else:
            rate = re.search(r"happens (\d+) times out of (\d+)", text)
            if rate:
                k, m_ = (int(g) for g in rate.groups())
                value = Fraction(k, m_)
            else:
                comp = re.search(r"does NOT occur is (-?\d+(?:\.\d+)?)", text)
                assert comp, f"could not parse a probability value from: {text!r}"
                value = Fraction(1) - Fraction(comp.group(1))

    if value <= b1:
        return "possible"
    if value <= b2:
        return "probable"
    if value < b3:
        return "likely"
    return "certain"


def test_language_oracle_matches_over_1000_seeds_per_tier():
    total = 0
    counts = {tier: {} for tier in TIERS}
    for tier in TIERS:
        for seed in range(1000):
            q = make("p19c_probability_language_classify", seed=seed, difficulty=tier)
            derived = _derive_language(q.text)
            assert derived == q.answer_text, (
                f"tier={tier} seed={seed}: oracle derived {derived!r} but answer_text is "
                f"{q.answer_text!r}\ntext: {q.text}"
            )
            counts[tier][derived] = counts[tier].get(derived, 0) + 1
            total += 1
    assert total == 4000
    print("probability-language classification counts per tier (1000 seeds each):", counts)
