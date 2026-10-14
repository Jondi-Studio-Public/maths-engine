"""Tests for the multi-part exam source `exd_x_r3g_rsa_keypair` (RSA: n, phi(n), d, then decrypt)."""

from __future__ import annotations

import math
import re

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_rsa_keypair"
SKILL = "ucv1-rsa-modular-step"


def _facts(q):
    p, qq, e, c = (int(re.search(rf"\b{v} = (\d+)", q.text).group(1)) for v in "pqec")
    return p, qq, e, c


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(
        TOPIC,
        "p12f_rsa_modular_step",
        {"mode": "decrypt", "primes": [13, 17, 19, 23, 29, 31], "exponents": [3, 5, 7]},
    )


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        p, qq, e, c = _facts(q)
        n, phi = p * qq, (p - 1) * (qq - 1)
        d = pow(e, -1, phi)
        want = {"a": str(n), "b": str(phi), "c": str(d), "d": str(pow(c, d, n))}
        # the decryption really inverts the encryption: the message re-encrypts to the ciphertext
        if sup.canonical(q) != want or pow(int(want["d"]), e, n) != c:
            bad.append((q.text, sup.canonical(q), want))
    assert bad == [], bad[:2]


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def test_follow_through_from_a_wrong_modulus():
    def replies(q):
        p, qq, e, c = _facts(q)
        n = p * qq + 10
        return {**sup.canonical(q), "a": str(n), "b": str(n - p - qq + 1)}, {"b"}, {"a"}

    sup.check_follow_through(SKILL, TOPIC, replies)


def test_follow_through_from_a_wrong_totient_into_the_exponent():
    def replies(q):
        p, qq, e, c = _facts(q)
        phi = (p - 1) * (qq - 1)
        wrong = next(phi + k for k in range(2, 40, 2) if math.gcd(e, phi + k) == 1)
        d = pow(e, -1, wrong)
        return {**sup.canonical(q), "b": str(wrong), "c": str(d)}, {"c"}, {"b"}

    sup.check_follow_through(SKILL, TOPIC, replies)


def test_follow_through_from_a_wrong_exponent_into_the_message():
    def replies(q):
        p, qq, e, c = _facts(q)
        n, phi = p * qq, (p - 1) * (qq - 1)
        d = pow(e, -1, phi)
        wrong = d - 1 if d > 3 else d + 1
        return {**sup.canonical(q), "c": str(wrong), "d": str(pow(c, wrong, n))}, {"d"}, {"c"}

    sup.check_follow_through(SKILL, TOPIC, replies)
