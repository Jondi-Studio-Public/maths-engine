"""Generates `exd_x_r3f_meridian_voyage` questions: an exam chain on a voyage along a meridian.

A vessel sails due north or south along a line of longitude between two latitudes, at a steady speed in knots,
burning fuel at a steady rate. Along a meridian one minute of latitude is one nautical mile. The chain:

(a) the distance sailed, in nautical miles                              number, 2 marks   (the latitude difference)
(b) the time taken, in hours to 1 decimal place                         number, 1 mark    part_a / speed
(c) the distance in kilometres, to the nearest kilometre                number, 1 mark    1.852 part_a
(d) the fuel used, in litres                                            number, 2 marks   part_b x rate

Parts (b), (c) and (d) follow through. The questions are built backwards from the time (a multiple of 0.1 h
chosen so the distance is a whole number of nautical miles) and the fuel rate (a multiple of 10 L/h), so the
canonical time and fuel are exact, and a learner who rounds the time as asked reaches the same fuel. About half
the voyages cross the equator, which is where the two latitudes are added rather than subtracted.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._exd_r3f_parts import clear_of_boundary, number_part
from ._exd_distinct_parts import distinct_part_answers

#: vessel -> (steady speeds in knots, fuel burn rates in litres per hour), plausible for the kind of vessel
_VESSELS = {
    "A cargo ship": ((15, 16, 20, 25), (1800, 2000, 2400, 3000)),
    "A supply ship": ((10, 12, 15), (900, 1200, 1500)),
    "A research vessel": ((10, 12), (500, 600, 800)),
    "A fishing trawler": ((8, 10), (150, 200, 250)),
    "A survey vessel": ((10, 12), (300, 400, 500)),
    "A patrol boat": ((15, 16, 20, 25), (400, 500, 600)),
}
_LONGITUDES = (
    "113\u00b0E",
    "151\u00b0E",
    "155\u00b0E",
    "160\u00b0E",
    "170\u00b0E",
    "178\u00b0E",
    "37\u00b0W",
    "62\u00b0W",
    "75\u00b0W",
)
_NM_TO_KM = Fraction(463, 250)  # 1.852 exactly

_STEMS = (
    "{vessel} leaves port at latitude {lat1} and sails due {direction} along the {lon} meridian to latitude "
    "{lat2}. It travels at a steady {speed} knots and burns fuel at {rate} litres per hour. One knot is one "
    "nautical mile per hour, one minute of latitude is one nautical mile, and 1 nautical mile = 1.852 km.",
    "{vessel} sails due {direction} along the {lon} line of longitude, from latitude {lat1} to latitude {lat2}. "
    "Its speed is a steady {speed} knots and it uses {rate} litres of fuel every hour. One knot is one nautical "
    "mile per hour, one minute of latitude is one nautical mile, and 1 nautical mile = 1.852 km.",
)


def _latitude_text(arcminutes: int) -> str:
    degrees, minutes = divmod(abs(arcminutes), 60)
    return f"{degrees}\u00b0{minutes:02d}\u2032 {'N' if arcminutes > 0 else 'S'}"


@generator("exd_x_r3f_meridian_voyage", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_meridian_voyage(rng: random.Random, tier: str = "exam") -> dict:
    """Distance in nautical miles, then the time, the distance in kilometres, and the fuel used."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    want_cross = rng.random() < 0.5
    vessel = rng.choice(sorted(_VESSELS))
    speeds, rates = _VESSELS[vessel]
    for _ in attempts(5000, "a voyage between two non-zero latitudes below 80 degrees"):
        speed = rng.choice(speeds)
        tenths = rng.randint(150, 1200)
        if (speed * tenths) % 10:
            continue
        nm = speed * tenths // 10
        lat1 = rng.randint(-4300, 4300)
        north = rng.random() < 0.5
        lat2 = lat1 + nm if north else lat1 - nm
        if lat1 == 0 or lat2 == 0 or abs(lat2) > 4800 or ((lat1 > 0) != (lat2 > 0)) != want_cross:
            continue
        km = Fraction(nm) * _NM_TO_KM
        if clear_of_boundary(km, 0):
            break
    hours = Fraction(nm, speed)
    rate = rng.choice(rates)
    fuel = hours * rate

    # Independent checks: the latitude strings read back to the distance, and the time and fuel are exact.
    check(abs(lat2 - lat1) == nm and (north == (lat2 > lat1)), "latitude difference disagrees with the distance")
    check(hours * 10 == tenths and (hours * rate).denominator == 1, "time or fuel is not exact")

    deg1, deg2 = abs(lat1) // 60 + (abs(lat1) % 60) / 100, abs(lat2) // 60 + (abs(lat2) % 60) / 100
    same_side = (lat1 > 0) == (lat2 > 0)
    nm_candidates = [
        (abs(abs(lat1) - abs(lat2)) if not same_side else abs(lat1) + abs(lat2), "wrong-hemisphere-rule"),
        (round(60 * abs(deg1 - deg2) if same_side else 60 * (deg1 + deg2)), "misreads-minutes-as-hundredths"),
        (nm / 2, "halves-the-constant"),
        (nm / 60, "forgot-arcminute-conversion"),
    ]
    hours_candidates = [
        (km / speed, "used-kilometres-with-knots"),
        (nm / (Fraction(463, 250) * speed), "converted-the-speed-instead-of-the-distance"),
        (nm * speed, "speed-time-inverted"),
    ]
    km_candidates = [
        (nm / _NM_TO_KM, "divided-by-nautical-mile-factor"),
        (Fraction(1609, 1000) * nm, "confuses-nm-with-statute-mile"),
        (nm, "forgot-unit-conversion"),
        (Fraction(1151, 1000) * nm, "converted-to-statute-miles-instead-of-km"),
    ]
    fuel_candidates = [
        (nm * rate, "distance-used-in-place-of-time"),
        (km / speed * rate, "used-kilometres-with-knots"),
        (hours + rate, "added-instead-of-multiplied"),
        (nm / (_NM_TO_KM * speed) * rate, "converted-the-speed-instead-of-the-distance"),
    ]

    stem = phrase(
        rng,
        _STEMS,
        vessel=vessel,
        direction="north" if north else "south",
        lon=rng.choice(_LONGITUDES),
        lat1=_latitude_text(lat1),
        lat2=_latitude_text(lat2),
        speed=speed,
        rate=rate,
    )
    exact_nm = lambda items: [(None if v is None else Fraction(v).limit_denominator(10**6), t) for v, t in items]  # noqa: E731
    return dict(
        text=stem,
        parts=[
            number_part(
                "Find the distance sailed, in nautical miles.",
                nm,
                exact_nm(nm_candidates),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The latitudes are {_latitude_text(lat1)} and {_latitude_text(lat2)}, so the voyage covers "
                    f"{nm} minutes of latitude, which is {nm} nautical miles."
                ],
            ),
            number_part(
                "Hence find the time taken, in hours, correct to 1 decimal place.",
                hours,
                exact_nm(hours_candidates),
                places=1,
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a/{speed}",
                working=[f"Time = {nm} \u00f7 {speed} = {float(hours):.1f} hours."],
            ),
            number_part(
                "Find the distance sailed, in kilometres, correct to the nearest kilometre.",
                km,
                exact_nm(km_candidates),
                marks=1,
                sensible_range=(0, None),
                follow_through="463*part_a/250",
                working=[f"{nm} \u00d7 1.852 = {float(km):.1f}, which is {round(km)} km to the nearest kilometre."],
            ),
            number_part(
                "Hence find the fuel used on the voyage, in litres.",
                fuel,
                exact_nm(fuel_candidates),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b*{rate}",
                working=[f"Fuel = {float(hours):.1f} \u00d7 {rate} = {int(fuel)} litres."],
            ),
        ],
    )
