"""Exam source for rectangular-prism volume: a fish tank taken from volume to the time needed to fill it.

Part (a) finds the volume in cm³, part (b) converts it to litres, part (c) finds how many more litres fill a tank
that already holds water to a stated depth, and part (d) turns that into minutes at a stated flow rate. Parts (b),
(c) and (d) carry `follow_through`. Dimensions are multiples of 10 cm so every litre figure is a whole number, and
the flow rate is drawn from the divisors of the litres still needed so the time is a whole number of minutes.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator

from ._exd_pr3_support import leaks, part

_TANKS = [
    "A rectangular fish tank at an aquarium shop in Adelaide",
    "A rectangular water tank on a farm near Wagga Wagga",
    "A rectangular tank at a marine centre in Cairns",
    "A rectangular rainwater tank behind a house in Hobart",
]
_FILLERS = ["A hose", "A pump", "A tap"]


@generator("exd_x_pr3_prism_volume", level=8, presets={"exam": dict(tier="exam")})
def exd_x_pr3_prism_volume(rng, tier="exam"):
    check(tier == "exam", "only the exam tier exists")
    for _ in attempts(2000, "a tank whose answers are not in the prompt"):
        L = 10 * rng.randint(4, 12)
        W = 10 * rng.randint(3, 10)
        H = 10 * rng.randint(3, 10)
        if W > L:
            L, W = W, L
        d = 10 * rng.randint(1, H // 10 - 1)
        vol = L * W * H
        litres = vol // 1000
        need = L * W * (H - d) // 1000
        rates = [r for r in range(2, 61) if need % r == 0 and 3 <= need // r <= 90]
        if not rates:
            continue
        r = rng.choice(rates)
        minutes = need // r
        check(vol % 1000 == 0 and L * W * (H - d) % 1000 == 0 and need == litres * (H - d) // H, "litres are not exact")
        stem = (
            f"{rng.choice(_TANKS)} is {L} cm long, {W} cm wide and {H} cm high. "
            f"It already holds water to a depth of {d} cm. One litre is 1000 cm³."
        )
        stem2 = f"{rng.choice(_FILLERS)} adds water at {r} litres per minute."
        if not leaks(stem + " " + stem2, [vol, litres, need, minutes]):
            break
    stem = f"{stem} {stem2} Give every answer as a whole number."
    parts = [
        part(
            "Find the volume of the whole tank, in cm³.",
            vol,
            marks=1,
            sensible_range=(0, None),
            working=[f"{L} × {W} × {H} = {vol} cm³."],
            candidates=[
                (2 * (L * W + L * H + W * H), "uses-surface-area"),
                (L + W + H, "adds-dimensions"),
                (L * W, "base-area-for-volume"),
            ],
        ),
        part(
            "Using your answer to (a), find the capacity of the whole tank, in litres.",
            litres,
            marks=1,
            follow_through="part_a/1000",
            working=[f"{vol} ÷ 1000 = {litres} litres."],
            candidates=[
                (Fraction(vol, 100), "wrong-power-of-ten-conversion"),
                (Fraction(vol, 10), "wrong-power-of-ten-conversion"),
                (vol, "unit-not-converted"),
            ],
        ),
        part(
            "How many more litres are needed to fill the tank to the top? Use your answer to (b).",
            need,
            marks=2,
            follow_through=f"part_b*({H}-{d})/{H}",
            working=[
                f"Empty depth = {H} − {d} = {H - d} cm, which is {H - d}/{H} of the tank.",
                f"{litres} × {H - d}/{H} = {need} litres.",
            ],
            candidates=[
                (Fraction(litres * d, H), "gave-part-not-remainder"),
                (litres, "ignored-water-already-in"),
                (litres - d, "subtracted-unlike-quantities"),
            ],
        ),
        part(
            "How many minutes will it take to fill the tank to the top? Use your answer to (c).",
            minutes,
            marks=1,
            follow_through=f"part_c/{r}",
            working=[f"{need} ÷ {r} = {minutes} minutes."],
            candidates=[
                (need * r, "multiplied-instead-of-divided-rate"),
                (Fraction(litres, r), "ignored-water-already-in"),
                (need - r, "subtracted-unlike-quantities"),
                (need + r, "rate-added-not-multiplied"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
