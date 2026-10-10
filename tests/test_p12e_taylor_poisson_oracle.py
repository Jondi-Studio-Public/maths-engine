"""Both skills are computational (the answer is fully determined by the rendered text), so this parses the text
with regexes and recomputes from scratch -- a from-scratch factorial/power loop and `fractions.Fraction`,
never `mathgen.core`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)

TOPIC_1 = "p12e_taylor_linear_approx"
TOPIC_2 = "p12e_poisson_count"


# ---------------------------------------------------------------------------
# Taylor linear approximation oracle


def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def _coef_value(coef_str: str) -> int:
    if coef_str in ("", "+"):
        return 1
    if coef_str == "-":
        return -1
    return int(coef_str)


def _parse_quadratic(expr_text: str) -> tuple[int, int, int]:
    """A quadratic p*x^2 + q*x + r (q/r terms optional), rendered like
    "3x^2 + 5x - 5", "x^2 - 5", "2 - x^2". Parse term by term with a plain
    regex over signed monomials -- never sympify (that would share
    machinery with the generator's own independent check, not be a second
    independent route). Each matched token's own trailing "x^2"/"x" (if
    any) decides which coefficient it sets, rather than relying on which
    alternation branch fired (an empty-vs-non-participating capture group
    are indistinguishable via `re.findall`, which was this parser's first,
    wrong attempt)."""
    p = q = r = 0
    for m in re.finditer(r"[+-]?\s*\d*x\^2|[+-]?\s*\d*x(?!\^)|[+-]?\s*\d+", expr_text):
        token = m.group(0).replace(" ", "")
        if token.endswith("x^2"):
            p = _coef_value(token[:-3])
        elif token.endswith("x"):
            q = _coef_value(token[:-1])
        else:
            r = int(token)
    return p, q, r


def _oracle_taylor_easy_medium_hard(text: str) -> Fraction:
    """easy/medium: f a quadratic; hard: f = k/x. Both share the same
    "f(x) = {expr}. ... a = {a} ... f({x0}) ..." surface shape."""
    m = re.match(r"^f\(x\) = ([^.]+)\.", text)
    assert m, f"could not find f(x) = ... in {text!r}"
    expr_text = m.group(1).strip()

    a_match = re.search(r"a = (-?\d+)\b", text)
    assert a_match, f"could not find 'a = <int>' in {text!r}"
    a = int(a_match.group(1))

    x0_match = re.search(r"f\((-?\d+(?:\.\d+)?)\)(?:\s*(?:\.|,| to))", text)
    assert x0_match, f"could not find the target f(x0) in {text!r}"
    x0 = Fraction(x0_match.group(1))

    rational_match = re.match(r"^(-?\d+)/x$", expr_text)
    if rational_match:
        k = int(rational_match.group(1))
        f_a = Fraction(k, a)
        fprime_a = Fraction(-k, a * a)
    else:
        p, q, r = _parse_quadratic(expr_text)
        f_a = p * a * a + q * a + r
        fprime_a = 2 * p * a + q

    delta = x0 - a
    return f_a + fprime_a * delta


def _oracle_taylor_exam(text: str) -> Fraction:
    target_match = re.search(r"[Ff]ind f'\((-?\d+)\)", text)
    assert target_match, f"could not find target f'(a) in {text!r}"
    a = int(target_match.group(1))

    fa_match = re.search(rf"f\({a}\) = (-?\d+)", text)
    assert fa_match, f"could not find f({a}) = ... in {text!r}"
    fa = int(fa_match.group(1))

    all_f_args = [int(v) for v in re.findall(r"f\((-?\d+)\)", text)]
    x0_candidates = [v for v in all_f_args if v != a]
    assert x0_candidates, f"could not find x0 in {text!r}"
    x0 = x0_candidates[0]

    l_match = re.search(rf"f\({x0}\)[^0-9\-]*(-?\d+)", text)
    assert l_match, f"could not find L(x0) near f({x0}) in {text!r}"
    l_x0 = int(l_match.group(1))

    return Fraction(l_x0 - fa, x0 - a)


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_taylor_oracle_recomputes_easy_medium_hard(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        recomputed = _oracle_taylor_easy_medium_hard(q.text)
        if tier in ("medium", "hard"):
            recomputed_str = f"{float(recomputed):.2f}"
            assert recomputed_str == q.answer_text, (
                f"{tier} seed={seed}: oracle {recomputed_str!r} != answer_text {q.answer_text!r} (text={q.text!r})"
            )
        else:
            assert recomputed == int(q.answer_text), (
                f"{tier} seed={seed}: oracle {recomputed} != answer_text {q.answer_text!r} (text={q.text!r})"
            )
        checked += 1
    assert checked == len(SEEDS)


def test_taylor_oracle_recomputes_exam():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty="exam")
        recomputed = _oracle_taylor_exam(q.text)
        assert recomputed == int(q.answer_text), (
            f"exam seed={seed}: oracle {recomputed} != answer_text {q.answer_text!r} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_taylor_oracle_is_red_on_a_corrupted_answer():
    """Proves the oracle actually checks something: mutate `answer_text` and
    confirm the oracle-vs-canonical comparison now fails."""
    q = make(TOPIC_1, seed=0, difficulty="easy")
    recomputed = _oracle_taylor_easy_medium_hard(q.text)
    corrupted = str(int(q.answer_text) + 7)
    assert recomputed != int(corrupted)


# ---------------------------------------------------------------------------
# Poisson count-probability oracle


def _poisson_p(e_val: Fraction, m: int, k: int) -> Fraction:
    power = 1
    for _ in range(k):
        power *= m
    return e_val * Fraction(power, _factorial(k))


def _oracle_poisson_easy(text: str) -> Fraction:
    m_match = re.search(r"lambda\*t = (\d+)", text)
    assert m_match, f"could not find lambda*t in {text!r}"
    m = int(m_match.group(1))

    e_match = re.search(r"e\^\(-lambda t\)\D*?(\d+/\d+|\d+)", text)
    assert e_match, f"could not find the given e^(-lambda t) value in {text!r}"
    e_val = Fraction(e_match.group(1))

    k_match = re.search(r"P\(N\(t\) = (\d+)\)", text)
    assert k_match, f"could not find k in {text!r}"
    k = int(k_match.group(1))

    return _poisson_p(e_val, m, k)


def _oracle_poisson_medium(text: str) -> Fraction:
    lam_match = re.search(r"lambda = (\d+)", text)
    t_match = re.search(r"\bt = (\d+)\b", text)
    e_match = re.search(r"e\^\(-lambda t\)\D*?(\d+/\d+|\d+)", text)
    k_match = re.search(r"P\(N\(t\) = (\d+)\)", text)
    assert lam_match and t_match and e_match and k_match, f"could not parse {text!r}"
    lam = int(lam_match.group(1))
    t = int(t_match.group(1))
    e_val = Fraction(e_match.group(1))
    k = int(k_match.group(1))
    return _poisson_p(e_val, lam * t, k)


def _oracle_poisson_hard(text: str) -> Fraction:
    m_match = re.search(r"lambda\*t = (\d+)", text)
    assert m_match, f"could not find lambda*t in {text!r}"
    m = int(m_match.group(1))
    ks = [int(v) for v in re.findall(r"P\(N\(t\) = (\d+)\)", text)]
    assert len(ks) == 2, f"expected exactly two P(N(t)=k) mentions, got {ks} in {text!r}"
    k1, k2 = ks
    p1 = _poisson_p(Fraction(1), m, k1)
    p2 = _poisson_p(Fraction(1), m, k2)
    return p1 / p2


def _oracle_poisson_exam(text: str) -> int:
    m_match = re.search(r"lambda\*t = (\d+)", text)
    assert m_match, f"could not find lambda*t in {text!r}"
    m = int(m_match.group(1))
    k1_match = re.search(r"P\(N\(t\) = (\d+)\)", text)
    assert k1_match, f"could not find k1 in {text!r}"
    k1 = int(k1_match.group(1))
    ratio_match = re.search(r"(?:=|\bequals\b)\s*(\d+/\d+|\d+)\s*exactly", text)
    assert ratio_match, f"could not find the stated ratio in {text!r}"
    ratio = Fraction(ratio_match.group(1))

    matches = []
    for k2_candidate in range(0, 9):
        p1 = _poisson_p(Fraction(1), m, k1)
        p2 = _poisson_p(Fraction(1), m, k2_candidate)
        if p1 / p2 == ratio:
            matches.append(k2_candidate)
    assert len(matches) == 1, f"ratio {ratio} does not pin down a unique k2 in 0..8 for {text!r}: {matches}"
    return matches[0]


@pytest.mark.parametrize(
    "tier,fn",
    [
        ("easy", _oracle_poisson_easy),
        ("medium", _oracle_poisson_medium),
        ("hard", _oracle_poisson_hard),
    ],
)
def test_poisson_oracle_recomputes(tier, fn):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        recomputed = fn(q.text)
        expected = Fraction(q.answer["p"], q.answer["q"])
        assert recomputed == expected, (
            f"{tier} seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_poisson_oracle_recomputes_exam():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty="exam")
        recomputed = _oracle_poisson_exam(q.text)
        assert recomputed == q.answer["p"], (
            f"exam seed={seed}: oracle {recomputed} != canonical {q.answer['p']} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_poisson_oracle_is_red_on_a_corrupted_answer():
    """Proves the oracle actually checks something: mutate the canonical p
    and confirm the oracle-vs-canonical comparison now fails."""
    q = make(TOPIC_2, seed=0, difficulty="easy")
    recomputed = _oracle_poisson_easy(q.text)
    corrupted = Fraction(q.answer["p"] + 1, q.answer["q"])
    assert recomputed != corrupted
