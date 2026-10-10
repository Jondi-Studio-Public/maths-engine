"""Great-circle distance between two places on Earth with DIFFERENT longitudes
(senior-find-distances-between-two-places-on-earth-using / ACMEM161).

This skill's own contract is narrowed to different-longitude pairs precisely
so it can never degenerate into its sibling
`senior-find-distances-between-two-places-on-earth-on` (ACMEM160,
`mathgen/generators/p9b_earth_distance_time_build.py`), which assesses the
SAME-longitude meridian-arc case. That module is read here only as a
structure/quality template -- nothing is imported from it, and no constant
(`NM_PER_KM` etc.) is shared; this module defines its own copy of every
constant it needs.

Kernel: the spherical law of cosines,
    cos(central_angle) = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon1-lon2)
with 1 minute of arc = 1 nautical mile = 1.852 km exactly (matching the
sibling's own stated convention, same words, independently defined here).

Every value in this module is a `float`, not a `Fraction` -- unlike most
`mathgen` generators. The central angle is inherently irrational (it comes
out of `acos`/`asin` of a real trigonometric expression), so there is no
exact rational arithmetic to carry through: `Fraction`-based helpers
(`round_half_up`, `stated_precision_string`, ...) all require a terminating
rational value and would raise on a typical seed here. This is a documented,
deliberate deviation from the project's usual Fraction-first convention, not
an oversight -- see `_round_half_up` and `_verify_central_angle` below.
"""

from __future__ import annotations

import math

from ..core import (
    check,
    check_distinct,
    generator,
    latex_answer,
    pick_distractors,
    stated_precision_tolerance,
)

# 1 nautical mile = 1.852 km, exactly. Own copy -- NOT imported from
# `p9b_earth_distance_time_build.py`.
KM_PER_NM = 1.852

# Floating-point trig tolerance for the independent haversine check below.
# Both the law-of-cosines and haversine formulas are mathematically exact for
# a central angle on a sphere; any disagreement beyond ordinary double-
# precision noise at these magnitudes (~1e-12) indicates a real construction
# defect, not a legitimate rounding difference between two rational routes
# (there is no rational route here -- see the module docstring). 1e-6 degrees
# is roughly 0.06 nautical miles, far tighter than anything this generator
# ever displays (whole nm/km, or 1 d.p. of longitude), so it can never mask a
# real defect while still being loose enough to absorb ordinary floating
# point noise from two independently-coded trig expressions.
_ANGLE_TOLERANCE_DEG = 1e-6

SPHERICITY_NOTE = (
    "Assume the Earth is a perfect sphere and 1 minute of arc equals 1 "
    "nautical mile (1 nautical mile = 1.852 km exactly)."
)


def _central_angle_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Spherical law of cosines -- construction's own route."""
    lat1r, lat2r, lon1r, lon2r = map(math.radians, (lat1, lat2, lon1, lon2))
    cos_c = math.sin(lat1r) * math.sin(lat2r) + math.cos(lat1r) * math.cos(lat2r) * math.cos(lon1r - lon2r)
    cos_c = max(-1.0, min(1.0, cos_c))
    return math.degrees(math.acos(cos_c))


def _central_angle_deg_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """The haversine formula -- a genuinely different route (structurally,
    not merely a second call to the same expression) used ONLY to
    independently check `_central_angle_deg` above, never for construction
    itself."""
    lat1r, lat2r, lon1r, lon2r = map(math.radians, (lat1, lat2, lon1, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    a = max(0.0, min(1.0, a))
    return math.degrees(2 * math.asin(math.sqrt(a)))


def _verify_central_angle(lat1: float, lon1: float, lat2: float, lon2: float, angle: float) -> None:
    """Independent check: the haversine formula must agree with the
    spherical-law-of-cosines `angle` construction actually used, within
    `_ANGLE_TOLERANCE_DEG` (see that constant's docstring for why an exact
    comparison is not available here)."""
    check_angle = _central_angle_deg_haversine(lat1, lon1, lat2, lon2)
    check(
        abs(check_angle - angle) < _ANGLE_TOLERANCE_DEG,
        f"law-of-cosines angle {angle} != haversine angle {check_angle}",
    )


def _round_half_up(value: float, places: int) -> float:
    """Ties-away-from-zero rounding for a `float` (mirrors
    `mathgen.core.round_half_up`'s tie-break rule exactly, but that helper
    is `Fraction`-typed only and cannot accept an irrational trig result --
    see the module docstring)."""
    scale = 10**places
    magnitude = math.floor(abs(value) * scale + 0.5)
    result = magnitude / scale
    return -result if value < 0 else result


def _fmt_int(value: float) -> str:
    return str(int(value))


def _fmt_1dp(value: float) -> str:
    return f"{value:.1f}"


def _lat_display(lat: float) -> str:
    mag = abs(lat)
    hemi = "N" if lat >= 0 else "S"
    return f"{mag:g}°{hemi}"


def _lon_display(lon: float) -> str:
    mag = abs(lon)
    hemi = "E" if lon >= 0 else "W"
    return f"{mag:g}°{hemi}"


def _same_hemisphere(lat1: float, lat2: float) -> bool:
    return (lat1 >= 0) == (lat2 >= 0)


# ============================================================================
# Data tables -- real, well-known cities, always at DIFFERENT longitudes.
# Each row is its own (place, lat, lon, place, lat, lon) tuple, not a shared
# lookup table, so a row's coordinates can be nudged by a few hundredths of a
# degree (still an "approximately" real location) to keep its rounded
# nm/km/longitude-difference figure clear of a x.5 rounding boundary --
# exactly the per-row-tuning discipline `p9b_earth_distance_time_build.py`
# already uses for its own tables.
# ============================================================================

# (place_a, lat_a, lon_a, place_b, lat_b, lon_b) -- both places in the SAME hemisphere (both N or
# both S). Every row selected programmatically from a real-city candidate pool for BOTH a plausible
# central angle (5, 150) AND its own rounded nm/km value (and every forward-tier misconception
# distractor) clearing a x.5 rounding boundary / distinctness requirement by a safety margin, so no
# row here needs an artificial coordinate nudge -- the margin requirement is satisfied by selection,
# the same discipline the original 12-row table used via manual per-row nudging.
SAME_HEMISPHERE = [
    ("Sydney", -33.87, 151.21, "Perth", -31.94, 115.86),
    ("Sydney", -33.87, 151.21, "Auckland", -36.85, 174.76),
    ("Sydney", -33.87, 151.21, "Cape Town", -33.93, 18.42),
    ("Sydney", -33.87, 151.21, "Buenos Aires", -34.61, -58.38),
    ("Sydney", -33.87, 151.21, "Lima", -12.05, -77.04),
    ("Sydney", -33.87, 151.21, "Santiago", -33.46, -70.67),
    ("Sydney", -33.87, 151.21, "Melbourne", -37.8, 144.96),
    ("Sydney", -33.87, 151.21, "Johannesburg", -26.2, 28.05),
    ("Perth", -31.94, 115.86, "Auckland", -36.85, 174.76),
    ("Perth", -31.94, 115.86, "Cape Town", -33.93, 18.42),
    ("Perth", -31.94, 115.86, "Buenos Aires", -34.61, -58.38),
    ("Perth", -31.94, 115.86, "Santiago", -33.46, -70.67),
    ("Perth", -31.94, 115.86, "Melbourne", -37.8, 144.96),
    ("Perth", -31.94, 115.86, "Johannesburg", -26.2, 28.05),
    ("Perth", -31.94, 115.86, "Nairobi", -1.29, 36.82),
    ("Auckland", -36.85, 174.76, "Cape Town", -33.93, 18.42),
    ("Auckland", -36.85, 174.76, "Buenos Aires", -34.61, -58.38),
    ("Auckland", -36.85, 174.76, "Lima", -12.05, -77.04),
    ("Auckland", -36.85, 174.76, "Santiago", -33.46, -70.67),
    ("Auckland", -36.85, 174.76, "Melbourne", -37.8, 144.96),
    ("Auckland", -36.85, 174.76, "Johannesburg", -26.2, 28.05),
    ("Rome", 41.9, 12.5, "Moscow", 55.76, 37.62),
    ("Rome", 41.9, 12.5, "London", 51.51, -0.13),
    ("Rome", 41.9, 12.5, "Tokyo", 35.68, 139.65),
    ("Rome", 41.9, 12.5, "Beijing", 39.89, 116.41),
    ("Rome", 41.9, 12.5, "Cairo", 30.04, 31.24),
    ("Rome", 41.9, 12.5, "Paris", 48.85, 2.35),
    ("Rome", 41.9, 12.5, "New York", 40.71, -74.01),
    ("Rome", 41.9, 12.5, "Mumbai", 19.08, 72.88),
    ("Moscow", 55.76, 37.62, "London", 51.51, -0.13),
    ("Moscow", 55.76, 37.62, "Tokyo", 35.68, 139.65),
    ("Moscow", 55.76, 37.62, "Beijing", 39.89, 116.41),
    ("Moscow", 55.76, 37.62, "Cairo", 30.04, 31.24),
    ("Moscow", 55.76, 37.62, "Paris", 48.85, 2.35),
    ("Moscow", 55.76, 37.62, "Mumbai", 19.08, 72.88),
    ("Moscow", 55.76, 37.62, "Toronto", 43.65, -79.38),
    ("London", 51.51, -0.13, "Tokyo", 35.68, 139.65),
    ("London", 51.51, -0.13, "Beijing", 39.89, 116.41),
    ("London", 51.51, -0.13, "Cairo", 30.04, 31.24),
    ("London", 51.51, -0.13, "New York", 40.71, -74.01),
    ("London", 51.51, -0.13, "Mumbai", 19.08, 72.88),
    ("London", 51.51, -0.13, "Toronto", 43.65, -79.38),
    ("Tokyo", 35.68, 139.65, "Beijing", 39.89, 116.41),
    ("Tokyo", 35.68, 139.65, "New York", 40.71, -74.01),
    ("Tokyo", 35.68, 139.65, "Mumbai", 19.08, 72.88),
    ("Tokyo", 35.68, 139.65, "Toronto", 43.65, -79.38),
    ("Tokyo", 35.68, 139.65, "Bogota", 4.71, -74.07),
    ("Beijing", 39.89, 116.41, "Cairo", 30.04, 31.24),
    ("Beijing", 39.89, 116.41, "Paris", 48.85, 2.35),
    ("Beijing", 39.89, 116.41, "Mumbai", 19.08, 72.88),
    ("Beijing", 39.89, 116.41, "Toronto", 43.65, -79.38),
    ("Cairo", 30.04, 31.24, "Paris", 48.85, 2.35),
    ("Cairo", 30.04, 31.24, "New York", 40.71, -74.01),
    ("Cairo", 30.04, 31.24, "Mumbai", 19.08, 72.88),
    ("Cairo", 30.04, 31.24, "Toronto", 43.65, -79.38),
    ("Cape Town", -33.93, 18.42, "Buenos Aires", -34.61, -58.38),
    ("Cape Town", -33.93, 18.42, "Lima", -12.05, -77.04),
    ("Cape Town", -33.93, 18.42, "Santiago", -33.46, -70.67),
    ("Cape Town", -33.93, 18.42, "Melbourne", -37.8, 144.96),
    ("Cape Town", -33.93, 18.42, "Johannesburg", -26.2, 28.05),
    ("Buenos Aires", -34.61, -58.38, "Lima", -12.05, -77.04),
    ("Buenos Aires", -34.61, -58.38, "Santiago", -33.46, -70.67),
    ("Buenos Aires", -34.61, -58.38, "Melbourne", -37.8, 144.96),
    ("Buenos Aires", -34.61, -58.38, "Johannesburg", -26.2, 28.05),
    ("Lima", -12.05, -77.04, "Santiago", -33.46, -70.67),
    ("Lima", -12.05, -77.04, "Melbourne", -37.8, 144.96),
    ("Lima", -12.05, -77.04, "Johannesburg", -26.2, 28.05),
    ("Lima", -12.05, -77.04, "Nairobi", -1.29, 36.82),
    ("Santiago", -33.46, -70.67, "Melbourne", -37.8, 144.96),
    ("Santiago", -33.46, -70.67, "Johannesburg", -26.2, 28.05),
    ("Melbourne", -37.8, 144.96, "Johannesburg", -26.2, 28.05),
    ("Paris", 48.85, 2.35, "New York", 40.71, -74.01),
    ("Paris", 48.85, 2.35, "Mumbai", 19.08, 72.88),
    ("Paris", 48.85, 2.35, "Toronto", 43.65, -79.38),
    ("Paris", 48.85, 2.35, "Bogota", 4.71, -74.07),
    ("Nairobi", -1.29, 36.82, "Sao Paulo", -23.55, -46.63),
    ("Nairobi", -1.29, 36.82, "Rio de Janeiro", -22.91, -43.17),
    ("Nairobi", -1.29, 36.82, "Montevideo", -34.9, -56.16),
    ("Nairobi", -1.29, 36.82, "Wellington", -41.29, 174.78),
    ("Nairobi", -1.29, 36.82, "Brisbane", -27.47, 153.03),
    ("Nairobi", -1.29, 36.82, "Adelaide", -34.93, 138.6),
    ("New York", 40.71, -74.01, "Manila", 14.6, 120.98),
    ("New York", 40.71, -74.01, "Madrid", 40.42, -3.7),
    ("New York", 40.71, -74.01, "Berlin", 52.52, 13.4),
    ("Mumbai", 19.08, 72.88, "Bogota", 4.71, -74.07),
    ("Toronto", 43.65, -79.38, "Bogota", 4.71, -74.07),
    ("Toronto", 43.65, -79.38, "Manila", 14.6, 120.98),
    ("Bogota", 4.71, -74.07, "Madrid", 40.42, -3.7),
    ("Bogota", 4.71, -74.07, "Athens", 37.98, 23.73),
    ("Bogota", 4.71, -74.07, "Delhi", 28.61, 77.21),
    ("Bogota", 4.71, -74.07, "Seoul", 37.57, 126.98),
    ("Manila", 14.6, 120.98, "Madrid", 40.42, -3.7),
    ("Manila", 14.6, 120.98, "Berlin", 52.52, 13.4),
    ("Manila", 14.6, 120.98, "Athens", 37.98, 23.73),
    ("Manila", 14.6, 120.98, "Delhi", 28.61, 77.21),
    ("Manila", 14.6, 120.98, "Bangkok", 13.76, 100.5),
    ("Manila", 14.6, 120.98, "Singapore", 1.35, 103.82),
    ("Jakarta", -6.2, 106.85, "Sao Paulo", -23.55, -46.63),
    ("Jakarta", -6.2, 106.85, "Montevideo", -34.9, -56.16),
    ("Jakarta", -6.2, 106.85, "Wellington", -41.29, 174.78),
    ("Jakarta", -6.2, 106.85, "Brisbane", -27.47, 153.03),
    ("Jakarta", -6.2, 106.85, "Adelaide", -34.93, 138.6),
    ("Madrid", 40.42, -3.7, "Berlin", 52.52, 13.4),
    ("Madrid", 40.42, -3.7, "Athens", 37.98, 23.73),
    ("Madrid", 40.42, -3.7, "Delhi", 28.61, 77.21),
    ("Madrid", 40.42, -3.7, "Bangkok", 13.76, 100.5),
    ("Madrid", 40.42, -3.7, "Singapore", 1.35, 103.82),
    ("Berlin", 52.52, 13.4, "Athens", 37.98, 23.73),
    ("Berlin", 52.52, 13.4, "Delhi", 28.61, 77.21),
    ("Berlin", 52.52, 13.4, "Bangkok", 13.76, 100.5),
    ("Berlin", 52.52, 13.4, "Singapore", 1.35, 103.82),
    ("Berlin", 52.52, 13.4, "Seoul", 37.57, 126.98),
    ("Athens", 37.98, 23.73, "Delhi", 28.61, 77.21),
    ("Athens", 37.98, 23.73, "Bangkok", 13.76, 100.5),
    ("Athens", 37.98, 23.73, "Singapore", 1.35, 103.82),
    ("Athens", 37.98, 23.73, "Seoul", 37.57, 126.98),
    ("Delhi", 28.61, 77.21, "Bangkok", 13.76, 100.5),
    ("Delhi", 28.61, 77.21, "Singapore", 1.35, 103.82),
    ("Delhi", 28.61, 77.21, "Seoul", 37.57, 126.98),
    ("Bangkok", 13.76, 100.5, "Shanghai", 31.23, 121.47),
    ("Bangkok", 13.76, 100.5, "Hong Kong", 22.32, 114.17),
    ("Bangkok", 13.76, 100.5, "Los Angeles", 34.05, -118.24),
    ("Singapore", 1.35, 103.82, "Seoul", 37.57, 126.98),
    ("Singapore", 1.35, 103.82, "Shanghai", 31.23, 121.47),
    ("Singapore", 1.35, 103.82, "Hong Kong", 22.32, 114.17),
    ("Seoul", 37.57, 126.98, "Hong Kong", 22.32, 114.17),
    ("Seoul", 37.57, 126.98, "Los Angeles", 34.05, -118.24),
    ("Seoul", 37.57, 126.98, "Chicago", 41.88, -87.63),
    ("Shanghai", 31.23, 121.47, "Hong Kong", 22.32, 114.17),
    ("Shanghai", 31.23, 121.47, "Los Angeles", 34.05, -118.24),
]

# Both places in OPPOSITE hemispheres (one N, one S). Same widening/selection
# discipline as `SAME_HEMISPHERE` above (130 rows, was 12).
OPPOSITE_HEMISPHERE = [
    ("Sydney", -33.87, 151.21, "Rome", 41.9, 12.5),
    ("Sydney", -33.87, 151.21, "Moscow", 55.76, 37.62),
    ("Sydney", -33.87, 151.21, "Tokyo", 35.68, 139.65),
    ("Sydney", -33.87, 151.21, "Beijing", 39.89, 116.41),
    ("Sydney", -33.87, 151.21, "Cairo", 30.04, 31.24),
    ("Sydney", -33.87, 151.21, "New York", 40.71, -74.01),
    ("Sydney", -33.87, 151.21, "Mumbai", 19.08, 72.88),
    ("Sydney", -33.87, 151.21, "Toronto", 43.65, -79.38),
    ("Perth", -31.94, 115.86, "Rome", 41.9, 12.5),
    ("Perth", -31.94, 115.86, "Moscow", 55.76, 37.62),
    ("Perth", -31.94, 115.86, "London", 51.51, -0.13),
    ("Perth", -31.94, 115.86, "Tokyo", 35.68, 139.65),
    ("Perth", -31.94, 115.86, "Mumbai", 19.08, 72.88),
    ("Perth", -31.94, 115.86, "Manila", 14.6, 120.98),
    ("Perth", -31.94, 115.86, "Madrid", 40.42, -3.7),
    ("Perth", -31.94, 115.86, "Athens", 37.98, 23.73),
    ("Auckland", -36.85, 174.76, "Moscow", 55.76, 37.62),
    ("Auckland", -36.85, 174.76, "Tokyo", 35.68, 139.65),
    ("Auckland", -36.85, 174.76, "Beijing", 39.89, 116.41),
    ("Auckland", -36.85, 174.76, "New York", 40.71, -74.01),
    ("Auckland", -36.85, 174.76, "Mumbai", 19.08, 72.88),
    ("Auckland", -36.85, 174.76, "Toronto", 43.65, -79.38),
    ("Auckland", -36.85, 174.76, "Bogota", 4.71, -74.07),
    ("Auckland", -36.85, 174.76, "Delhi", 28.61, 77.21),
    ("Rome", 41.9, 12.5, "Lima", -12.05, -77.04),
    ("Rome", 41.9, 12.5, "Santiago", -33.46, -70.67),
    ("Rome", 41.9, 12.5, "Melbourne", -37.8, 144.96),
    ("Rome", 41.9, 12.5, "Johannesburg", -26.2, 28.05),
    ("Rome", 41.9, 12.5, "Nairobi", -1.29, 36.82),
    ("Rome", 41.9, 12.5, "Jakarta", -6.2, 106.85),
    ("Moscow", 55.76, 37.62, "Cape Town", -33.93, 18.42),
    ("Moscow", 55.76, 37.62, "Buenos Aires", -34.61, -58.38),
    ("Moscow", 55.76, 37.62, "Lima", -12.05, -77.04),
    ("Moscow", 55.76, 37.62, "Santiago", -33.46, -70.67),
    ("Moscow", 55.76, 37.62, "Melbourne", -37.8, 144.96),
    ("London", 51.51, -0.13, "Cape Town", -33.93, 18.42),
    ("London", 51.51, -0.13, "Buenos Aires", -34.61, -58.38),
    ("London", 51.51, -0.13, "Santiago", -33.46, -70.67),
    ("London", 51.51, -0.13, "Johannesburg", -26.2, 28.05),
    ("London", 51.51, -0.13, "Nairobi", -1.29, 36.82),
    ("London", 51.51, -0.13, "Jakarta", -6.2, 106.85),
    ("London", 51.51, -0.13, "Sao Paulo", -23.55, -46.63),
    ("Tokyo", 35.68, 139.65, "Cape Town", -33.93, 18.42),
    ("Tokyo", 35.68, 139.65, "Lima", -12.05, -77.04),
    ("Tokyo", 35.68, 139.65, "Melbourne", -37.8, 144.96),
    ("Tokyo", 35.68, 139.65, "Johannesburg", -26.2, 28.05),
    ("Tokyo", 35.68, 139.65, "Nairobi", -1.29, 36.82),
    ("Beijing", 39.89, 116.41, "Cape Town", -33.93, 18.42),
    ("Beijing", 39.89, 116.41, "Lima", -12.05, -77.04),
    ("Beijing", 39.89, 116.41, "Melbourne", -37.8, 144.96),
    ("Beijing", 39.89, 116.41, "Johannesburg", -26.2, 28.05),
    ("Beijing", 39.89, 116.41, "Nairobi", -1.29, 36.82),
    ("Beijing", 39.89, 116.41, "Wellington", -41.29, 174.78),
    ("Cairo", 30.04, 31.24, "Cape Town", -33.93, 18.42),
    ("Cairo", 30.04, 31.24, "Buenos Aires", -34.61, -58.38),
    ("Cairo", 30.04, 31.24, "Lima", -12.05, -77.04),
    ("Cairo", 30.04, 31.24, "Santiago", -33.46, -70.67),
    ("Cairo", 30.04, 31.24, "Melbourne", -37.8, 144.96),
    ("Cairo", 30.04, 31.24, "Johannesburg", -26.2, 28.05),
    ("Cairo", 30.04, 31.24, "Nairobi", -1.29, 36.82),
    ("Cape Town", -33.93, 18.42, "Paris", 48.85, 2.35),
    ("Cape Town", -33.93, 18.42, "New York", 40.71, -74.01),
    ("Cape Town", -33.93, 18.42, "Mumbai", 19.08, 72.88),
    ("Buenos Aires", -34.61, -58.38, "New York", 40.71, -74.01),
    ("Buenos Aires", -34.61, -58.38, "Mumbai", 19.08, 72.88),
    ("Buenos Aires", -34.61, -58.38, "Toronto", 43.65, -79.38),
    ("Buenos Aires", -34.61, -58.38, "Bogota", 4.71, -74.07),
    ("Buenos Aires", -34.61, -58.38, "Madrid", 40.42, -3.7),
    ("Lima", -12.05, -77.04, "Paris", 48.85, 2.35),
    ("Lima", -12.05, -77.04, "New York", 40.71, -74.01),
    ("Lima", -12.05, -77.04, "Toronto", 43.65, -79.38),
    ("Santiago", -33.46, -70.67, "Paris", 48.85, 2.35),
    ("Santiago", -33.46, -70.67, "New York", 40.71, -74.01),
    ("Santiago", -33.46, -70.67, "Mumbai", 19.08, 72.88),
    ("Santiago", -33.46, -70.67, "Toronto", 43.65, -79.38),
    ("Melbourne", -37.8, 144.96, "New York", 40.71, -74.01),
    ("Melbourne", -37.8, 144.96, "Toronto", 43.65, -79.38),
    ("Melbourne", -37.8, 144.96, "Manila", 14.6, 120.98),
    ("Johannesburg", -26.2, 28.05, "Paris", 48.85, 2.35),
    ("Johannesburg", -26.2, 28.05, "New York", 40.71, -74.01),
    ("Johannesburg", -26.2, 28.05, "Mumbai", 19.08, 72.88),
    ("Paris", 48.85, 2.35, "Nairobi", -1.29, 36.82),
    ("Paris", 48.85, 2.35, "Jakarta", -6.2, 106.85),
    ("Paris", 48.85, 2.35, "Sao Paulo", -23.55, -46.63),
    ("Paris", 48.85, 2.35, "Rio de Janeiro", -22.91, -43.17),
    ("Nairobi", -1.29, 36.82, "Mumbai", 19.08, 72.88),
    ("Nairobi", -1.29, 36.82, "Toronto", 43.65, -79.38),
    ("Toronto", 43.65, -79.38, "Jakarta", -6.2, 106.85),
    ("Bogota", 4.71, -74.07, "Sao Paulo", -23.55, -46.63),
    ("Bogota", 4.71, -74.07, "Rio de Janeiro", -22.91, -43.17),
    ("Bogota", 4.71, -74.07, "Montevideo", -34.9, -56.16),
    ("Bogota", 4.71, -74.07, "Wellington", -41.29, 174.78),
    ("Bogota", 4.71, -74.07, "Brisbane", -27.47, 153.03),
    ("Bogota", 4.71, -74.07, "Adelaide", -34.93, 138.6),
    ("Manila", 14.6, 120.98, "Jakarta", -6.2, 106.85),
    ("Manila", 14.6, 120.98, "Wellington", -41.29, 174.78),
    ("Manila", 14.6, 120.98, "Brisbane", -27.47, 153.03),
    ("Manila", 14.6, 120.98, "Adelaide", -34.93, 138.6),
    ("Jakarta", -6.2, 106.85, "Madrid", 40.42, -3.7),
    ("Jakarta", -6.2, 106.85, "Berlin", 52.52, 13.4),
    ("Jakarta", -6.2, 106.85, "Athens", 37.98, 23.73),
    ("Madrid", 40.42, -3.7, "Rio de Janeiro", -22.91, -43.17),
    ("Madrid", 40.42, -3.7, "Montevideo", -34.9, -56.16),
    ("Madrid", 40.42, -3.7, "Adelaide", -34.93, 138.6),
    ("Madrid", 40.42, -3.7, "Quito", -0.18, -78.47),
    ("Berlin", 52.52, 13.4, "Sao Paulo", -23.55, -46.63),
    ("Berlin", 52.52, 13.4, "Montevideo", -34.9, -56.16),
    ("Berlin", 52.52, 13.4, "Brisbane", -27.47, 153.03),
    ("Berlin", 52.52, 13.4, "Adelaide", -34.93, 138.6),
    ("Berlin", 52.52, 13.4, "Quito", -0.18, -78.47),
    ("Athens", 37.98, 23.73, "Sao Paulo", -23.55, -46.63),
    ("Athens", 37.98, 23.73, "Rio de Janeiro", -22.91, -43.17),
    ("Athens", 37.98, 23.73, "Montevideo", -34.9, -56.16),
    ("Athens", 37.98, 23.73, "Brisbane", -27.47, 153.03),
    ("Athens", 37.98, 23.73, "Adelaide", -34.93, 138.6),
    ("Athens", 37.98, 23.73, "Quito", -0.18, -78.47),
    ("Delhi", 28.61, 77.21, "Sao Paulo", -23.55, -46.63),
    ("Delhi", 28.61, 77.21, "Rio de Janeiro", -22.91, -43.17),
    ("Delhi", 28.61, 77.21, "Montevideo", -34.9, -56.16),
    ("Delhi", 28.61, 77.21, "Brisbane", -27.47, 153.03),
    ("Delhi", 28.61, 77.21, "Adelaide", -34.93, 138.6),
    ("Bangkok", 13.76, 100.5, "Sao Paulo", -23.55, -46.63),
    ("Bangkok", 13.76, 100.5, "Rio de Janeiro", -22.91, -43.17),
    ("Bangkok", 13.76, 100.5, "Wellington", -41.29, 174.78),
    ("Bangkok", 13.76, 100.5, "Brisbane", -27.47, 153.03),
    ("Bangkok", 13.76, 100.5, "Adelaide", -34.93, 138.6),
    ("Singapore", 1.35, 103.82, "Sao Paulo", -23.55, -46.63),
    ("Singapore", 1.35, 103.82, "Rio de Janeiro", -22.91, -43.17),
    ("Singapore", 1.35, 103.82, "Montevideo", -34.9, -56.16),
    ("Singapore", 1.35, 103.82, "Wellington", -41.29, 174.78),
]

# The exam (inverse) tier's own table: widened to 140 rows (was 24) AND, filtered through
# `_well_conditioned` (defined below, after the shared `_invert_delta_lon_deg`/`_delta_lon_for_km`
# helpers) so every surviving row's answer is stable against a +/-1 km change in the given distance.
# Selected programmatically from a real-city candidate pool for: plausible central angle, nm/km AND
# 1-d.p.-longitude rounding margin, distractor distinctness, and conditioning -- so, as with the two
# tables above, no row here needs a hand nudge. This table is NOT reused for the forward tiers and
# is not merged with them (see the original block comment on why).
INVERSE_LONGITUDE_ROWS = [
    ("Sydney", -33.87, 151.21, "Rome", 41.9, 12.5),
    ("Sydney", -33.87, 151.21, "Moscow", 55.76, 37.62),
    ("Sydney", -33.87, 151.21, "Cairo", 30.04, 31.24),
    ("Sydney", -33.87, 151.21, "Cape Town", -33.93, 18.42),
    ("Sydney", -33.87, 151.21, "Santiago", -33.46, -70.67),
    ("Sydney", -33.87, 151.21, "Melbourne", -37.8, 144.96),
    ("Sydney", -33.87, 151.21, "Johannesburg", -26.2, 28.05),
    ("Sydney", -33.87, 151.21, "Nairobi", -1.29, 36.82),
    ("Perth", -31.94, 115.86, "Auckland", -36.85, 174.76),
    ("Perth", -31.94, 115.86, "Rome", 41.9, 12.5),
    ("Perth", -31.94, 115.86, "London", 51.51, -0.13),
    ("Perth", -31.94, 115.86, "Cape Town", -33.93, 18.42),
    ("Perth", -31.94, 115.86, "Melbourne", -37.8, 144.96),
    ("Perth", -31.94, 115.86, "Johannesburg", -26.2, 28.05),
    ("Perth", -31.94, 115.86, "Nairobi", -1.29, 36.82),
    ("Perth", -31.94, 115.86, "Mumbai", 19.08, 72.88),
    ("Auckland", -36.85, 174.76, "Moscow", 55.76, 37.62),
    ("Auckland", -36.85, 174.76, "Beijing", 39.89, 116.41),
    ("Auckland", -36.85, 174.76, "Buenos Aires", -34.61, -58.38),
    ("Auckland", -36.85, 174.76, "Lima", -12.05, -77.04),
    ("Auckland", -36.85, 174.76, "Santiago", -33.46, -70.67),
    ("Auckland", -36.85, 174.76, "Melbourne", -37.8, 144.96),
    ("Auckland", -36.85, 174.76, "Nairobi", -1.29, 36.82),
    ("Rome", 41.9, 12.5, "Moscow", 55.76, 37.62),
    ("Rome", 41.9, 12.5, "London", 51.51, -0.13),
    ("Rome", 41.9, 12.5, "Tokyo", 35.68, 139.65),
    ("Rome", 41.9, 12.5, "Beijing", 39.89, 116.41),
    ("Rome", 41.9, 12.5, "Cairo", 30.04, 31.24),
    ("Rome", 41.9, 12.5, "Buenos Aires", -34.61, -58.38),
    ("Moscow", 55.76, 37.62, "Tokyo", 35.68, 139.65),
    ("Moscow", 55.76, 37.62, "Beijing", 39.89, 116.41),
    ("Moscow", 55.76, 37.62, "Buenos Aires", -34.61, -58.38),
    ("Moscow", 55.76, 37.62, "Santiago", -33.46, -70.67),
    ("Moscow", 55.76, 37.62, "Melbourne", -37.8, 144.96),
    ("London", 51.51, -0.13, "Cairo", 30.04, 31.24),
    ("London", 51.51, -0.13, "New York", 40.71, -74.01),
    ("London", 51.51, -0.13, "Mumbai", 19.08, 72.88),
    ("London", 51.51, -0.13, "Toronto", 43.65, -79.38),
    ("London", 51.51, -0.13, "Bogota", 4.71, -74.07),
    ("London", 51.51, -0.13, "Jakarta", -6.2, 106.85),
    ("Tokyo", 35.68, 139.65, "Beijing", 39.89, 116.41),
    ("Tokyo", 35.68, 139.65, "Cairo", 30.04, 31.24),
    ("Tokyo", 35.68, 139.65, "Cape Town", -33.93, 18.42),
    ("Tokyo", 35.68, 139.65, "Lima", -12.05, -77.04),
    ("Tokyo", 35.68, 139.65, "Johannesburg", -26.2, 28.05),
    ("Tokyo", 35.68, 139.65, "Nairobi", -1.29, 36.82),
    ("Beijing", 39.89, 116.41, "Cairo", 30.04, 31.24),
    ("Beijing", 39.89, 116.41, "Cape Town", -33.93, 18.42),
    ("Beijing", 39.89, 116.41, "Johannesburg", -26.2, 28.05),
    ("Beijing", 39.89, 116.41, "Nairobi", -1.29, 36.82),
    ("Cairo", 30.04, 31.24, "Buenos Aires", -34.61, -58.38),
    ("Cairo", 30.04, 31.24, "Lima", -12.05, -77.04),
    ("Cairo", 30.04, 31.24, "Santiago", -33.46, -70.67),
    ("Cape Town", -33.93, 18.42, "Buenos Aires", -34.61, -58.38),
    ("Cape Town", -33.93, 18.42, "Lima", -12.05, -77.04),
    ("Cape Town", -33.93, 18.42, "Santiago", -33.46, -70.67),
    ("Cape Town", -33.93, 18.42, "Melbourne", -37.8, 144.96),
    ("Buenos Aires", -34.61, -58.38, "Lima", -12.05, -77.04),
    ("Buenos Aires", -34.61, -58.38, "Santiago", -33.46, -70.67),
    ("Buenos Aires", -34.61, -58.38, "Johannesburg", -26.2, 28.05),
    ("Lima", -12.05, -77.04, "Melbourne", -37.8, 144.96),
    ("Lima", -12.05, -77.04, "Johannesburg", -26.2, 28.05),
    ("Lima", -12.05, -77.04, "Paris", 48.85, 2.35),
    ("Santiago", -33.46, -70.67, "Johannesburg", -26.2, 28.05),
    ("Santiago", -33.46, -70.67, "Nairobi", -1.29, 36.82),
    ("Melbourne", -37.8, 144.96, "Johannesburg", -26.2, 28.05),
    ("Melbourne", -37.8, 144.96, "Nairobi", -1.29, 36.82),
    ("Paris", 48.85, 2.35, "New York", 40.71, -74.01),
    ("Paris", 48.85, 2.35, "Mumbai", 19.08, 72.88),
    ("Paris", 48.85, 2.35, "Toronto", 43.65, -79.38),
    ("Paris", 48.85, 2.35, "Bogota", 4.71, -74.07),
    ("Paris", 48.85, 2.35, "Manila", 14.6, 120.98),
    ("Paris", 48.85, 2.35, "Jakarta", -6.2, 106.85),
    ("Paris", 48.85, 2.35, "Athens", 37.98, 23.73),
    ("Nairobi", -1.29, 36.82, "New York", 40.71, -74.01),
    ("New York", 40.71, -74.01, "Madrid", 40.42, -3.7),
    ("New York", 40.71, -74.01, "Berlin", 52.52, 13.4),
    ("New York", 40.71, -74.01, "Athens", 37.98, 23.73),
    ("New York", 40.71, -74.01, "Los Angeles", 34.05, -118.24),
    ("New York", 40.71, -74.01, "Chicago", 41.88, -87.63),
    ("Mumbai", 19.08, 72.88, "Manila", 14.6, 120.98),
    ("Mumbai", 19.08, 72.88, "Jakarta", -6.2, 106.85),
    ("Mumbai", 19.08, 72.88, "Madrid", 40.42, -3.7),
    ("Mumbai", 19.08, 72.88, "Berlin", 52.52, 13.4),
    ("Mumbai", 19.08, 72.88, "Athens", 37.98, 23.73),
    ("Toronto", 43.65, -79.38, "Madrid", 40.42, -3.7),
    ("Toronto", 43.65, -79.38, "Berlin", 52.52, 13.4),
    ("Toronto", 43.65, -79.38, "Athens", 37.98, 23.73),
    ("Toronto", 43.65, -79.38, "Los Angeles", 34.05, -118.24),
    ("Toronto", 43.65, -79.38, "Mexico City", 19.43, -99.13),
    ("Toronto", 43.65, -79.38, "Wellington", -41.29, 174.78),
    ("Bogota", 4.71, -74.07, "Madrid", 40.42, -3.7),
    ("Bogota", 4.71, -74.07, "Berlin", 52.52, 13.4),
    ("Bogota", 4.71, -74.07, "Athens", 37.98, 23.73),
    ("Bogota", 4.71, -74.07, "Delhi", 28.61, 77.21),
    ("Bogota", 4.71, -74.07, "Mexico City", 19.43, -99.13),
    ("Bogota", 4.71, -74.07, "Sao Paulo", -23.55, -46.63),
    ("Manila", 14.6, 120.98, "Jakarta", -6.2, 106.85),
    ("Manila", 14.6, 120.98, "Madrid", 40.42, -3.7),
    ("Manila", 14.6, 120.98, "Berlin", 52.52, 13.4),
    ("Manila", 14.6, 120.98, "Delhi", 28.61, 77.21),
    ("Manila", 14.6, 120.98, "Bangkok", 13.76, 100.5),
    ("Manila", 14.6, 120.98, "Singapore", 1.35, 103.82),
    ("Jakarta", -6.2, 106.85, "Madrid", 40.42, -3.7),
    ("Jakarta", -6.2, 106.85, "Athens", 37.98, 23.73),
    ("Jakarta", -6.2, 106.85, "Delhi", 28.61, 77.21),
    ("Jakarta", -6.2, 106.85, "Los Angeles", 34.05, -118.24),
    ("Madrid", 40.42, -3.7, "Berlin", 52.52, 13.4),
    ("Madrid", 40.42, -3.7, "Athens", 37.98, 23.73),
    ("Berlin", 52.52, 13.4, "Delhi", 28.61, 77.21),
    ("Berlin", 52.52, 13.4, "Bangkok", 13.76, 100.5),
    ("Athens", 37.98, 23.73, "Delhi", 28.61, 77.21),
    ("Delhi", 28.61, 77.21, "Bangkok", 13.76, 100.5),
    ("Delhi", 28.61, 77.21, "Singapore", 1.35, 103.82),
    ("Delhi", 28.61, 77.21, "Seoul", 37.57, 126.98),
    ("Bangkok", 13.76, 100.5, "Seoul", 37.57, 126.98),
    ("Bangkok", 13.76, 100.5, "Shanghai", 31.23, 121.47),
    ("Bangkok", 13.76, 100.5, "Hong Kong", 22.32, 114.17),
    ("Bangkok", 13.76, 100.5, "Los Angeles", 34.05, -118.24),
    ("Bangkok", 13.76, 100.5, "Mexico City", 19.43, -99.13),
    ("Singapore", 1.35, 103.82, "Los Angeles", 34.05, -118.24),
    ("Singapore", 1.35, 103.82, "Rio de Janeiro", -22.91, -43.17),
    ("Singapore", 1.35, 103.82, "Montevideo", -34.9, -56.16),
    ("Singapore", 1.35, 103.82, "Wellington", -41.29, 174.78),
    ("Singapore", 1.35, 103.82, "Brisbane", -27.47, 153.03),
    ("Singapore", 1.35, 103.82, "Adelaide", -34.93, 138.6),
    ("Seoul", 37.57, 126.98, "Hong Kong", 22.32, 114.17),
    ("Seoul", 37.57, 126.98, "Los Angeles", 34.05, -118.24),
    ("Seoul", 37.57, 126.98, "Mexico City", 19.43, -99.13),
    ("Seoul", 37.57, 126.98, "Dubai", 25.2, 55.27),
    ("Seoul", 37.57, 126.98, "Istanbul", 41.01, 28.98),
    ("Seoul", 37.57, 126.98, "Vienna", 48.21, 16.37),
    ("Shanghai", 31.23, 121.47, "Hong Kong", 22.32, 114.17),
    ("Shanghai", 31.23, 121.47, "Los Angeles", 34.05, -118.24),
    ("Shanghai", 31.23, 121.47, "Mexico City", 19.43, -99.13),
    ("Shanghai", 31.23, 121.47, "Wellington", -41.29, 174.78),
    ("Shanghai", 31.23, 121.47, "Dubai", 25.2, 55.27),
    ("Shanghai", 31.23, 121.47, "Istanbul", 41.01, 28.98),
    ("Shanghai", 31.23, 121.47, "Vienna", 48.21, 16.37),
    ("Hong Kong", 22.32, 114.17, "Los Angeles", 34.05, -118.24),
]


def _forward_question(rng, rows, *, unit: str):
    """Shared body for the easy/medium/hard (forward) tiers -- draw a row,
    compute the great-circle distance, and offer it in nautical miles
    (`unit="nm"`, easy/medium) or kilometres (`unit="km"`, hard)."""
    place_a, lat_a, lon_a, place_b, lat_b, lon_b = rng.choice(rows)
    check(lon_a != lon_b, f"{place_a}/{place_b}: longitudes must differ (ACMEM161)")

    same_hemi = _same_hemisphere(lat_a, lat_b)
    angle = _central_angle_deg(lat_a, lon_a, lat_b, lon_b)
    _verify_central_angle(lat_a, lon_a, lat_b, lon_b, angle)
    check(5.0 < angle < 150.0, f"central angle {angle} outside the plausible (5, 150) range")

    nm_raw = angle * 60.0
    nm = int(_round_half_up(nm_raw, 0))

    # Misconceptions shared by every forward tier.
    delta_lat = abs(lat_a - lat_b) if same_hemi else abs(lat_a) + abs(lat_b)
    meridian_nm = int(_round_half_up(delta_lat * 60.0, 0))  # ignores longitude entirely
    wrong_hemi_angle = _central_angle_deg(lat_a, lon_a, -lat_b, lon_b)
    wrong_hemi_nm = int(_round_half_up(wrong_hemi_angle * 60.0, 0))
    forgot_arcmin_nm = int(_round_half_up(angle, 0))  # forgot the x60 step

    lat_a_disp, lat_b_disp = _lat_display(lat_a), _lat_display(lat_b)
    lon_a_disp, lon_b_disp = _lon_display(lon_a), _lon_display(lon_b)

    if unit == "nm":
        answer_val = nm
        answer_text = _fmt_int(nm)
        text = (
            f"{place_a} is located at latitude {lat_a_disp} and longitude {lon_a_disp}. "
            f"{place_b} is located at latitude {lat_b_disp} and longitude {lon_b_disp}. "
            f"{SPHERICITY_NOTE} Using the great-circle (spherical law of cosines) method, "
            f"what is the distance between {place_a} and {place_b}, in nautical miles, "
            f"to the nearest nautical mile?"
        )
        working = [
            f"cos(central angle) = sin({lat_a_disp}) x sin({lat_b_disp}) + "
            f"cos({lat_a_disp}) x cos({lat_b_disp}) x cos({lon_a_disp} - {lon_b_disp})",
            f"central angle = {angle:.4f} degrees",
            f"distance = {angle:.4f} x 60 = {nm_raw:.2f} nm -> {nm} nm to the nearest nm",
        ]
        check_distinct(answer_val, meridian_nm, wrong_hemi_nm, forgot_arcmin_nm)
        candidates = [_fmt_int(meridian_nm), _fmt_int(wrong_hemi_nm), _fmt_int(forgot_arcmin_nm)]
        distractors = pick_distractors(answer_text, candidates)
    else:  # unit == "km"
        km_raw = nm_raw * KM_PER_NM
        km = int(_round_half_up(km_raw, 0))
        answer_val = km
        answer_text = _fmt_int(km)
        text = (
            f"{place_a} is located at latitude {lat_a_disp} and longitude {lon_a_disp}. "
            f"{place_b} is located at latitude {lat_b_disp} and longitude {lon_b_disp}. "
            f"{SPHERICITY_NOTE} Using the great-circle (spherical law of cosines) method, "
            f"what is the distance between {place_a} and {place_b}, in kilometres, "
            f"to the nearest kilometre?"
        )
        working = [
            f"cos(central angle) = sin({lat_a_disp}) x sin({lat_b_disp}) + "
            f"cos({lat_a_disp}) x cos({lat_b_disp}) x cos({lon_a_disp} - {lon_b_disp})",
            f"central angle = {angle:.4f} degrees, distance = {angle:.4f} x 60 = {nm_raw:.2f} nm",
            f"{nm_raw:.2f} nm x 1.852 = {km_raw:.2f} km -> {km} km to the nearest km",
        ]
        forgot_unit_km = nm  # reports the nm figure as if it were already km
        meridian_km = int(_round_half_up(meridian_nm * KM_PER_NM, 0))
        wrong_hemi_km = int(_round_half_up(wrong_hemi_nm * KM_PER_NM, 0))
        forgot_arcmin_km = int(_round_half_up(forgot_arcmin_nm * KM_PER_NM, 0))
        check_distinct(answer_val, forgot_unit_km, meridian_km, wrong_hemi_km, forgot_arcmin_km)
        candidates = [
            _fmt_int(forgot_unit_km),
            _fmt_int(meridian_km),
            _fmt_int(wrong_hemi_km),
            _fmt_int(forgot_arcmin_km),
        ]
        distractors = pick_distractors(answer_text, candidates)

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer_val,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _invert_delta_lon_deg(lat_a: float, lat_b: float, angle_from_given_deg: float) -> float:
    """Invert the spherical law of cosines for the longitude difference,
    given both latitudes and the central angle (in degrees) implied by the
    GIVEN great-circle distance. The SAME routine both `_inverse_question`
    (construction) and `_well_conditioned` (the conditioning filter below)
    call, refactored out of `_inverse_question`'s former local closure so the
    two can never silently diverge into two different formulas."""
    lat_ar, lat_br = math.radians(lat_a), math.radians(lat_b)
    denom = math.cos(lat_ar) * math.cos(lat_br)
    angle_from_given_rad = math.radians(angle_from_given_deg)
    cos_arg = (math.cos(angle_from_given_rad) - math.sin(lat_ar) * math.sin(lat_br)) / denom
    cos_arg = max(-1.0, min(1.0, cos_arg))
    return math.degrees(math.acos(cos_arg))


def _angle_deg_from_km(km: float) -> float:
    """km (given great-circle distance) -> the central angle, in degrees, it
    implies -- the SAME km -> nm -> degrees conversion `_inverse_question`
    uses for its own `angle_from_given`, factored out so the conditioning
    predicate can never silently diverge from it."""
    return (km / KM_PER_NM) / 60.0


def _delta_lon_for_km(lat_a: float, lat_b: float, km: float) -> float:
    """km (given great-circle distance) -> unrounded longitude-difference
    answer, via `_angle_deg_from_km` and `_invert_delta_lon_deg`. Shared by
    `_inverse_question` and `_well_conditioned`."""
    return _invert_delta_lon_deg(lat_a, lat_b, _angle_deg_from_km(km))


def _well_conditioned(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> bool:
    """A +/-1 km change in the GIVEN great-circle distance must move the unrounded longitude-difference answer by
    less than 0.02 degrees -- a learner who rounds the intermediate central angle to the displayed precision
    must never be able to cross into a different rounded answer. Central difference over +/-1 km around the
    row's own true (rounded) given distance, using the SAME `_delta_lon_for_km` inversion the real question
    construction calls."""
    angle = _central_angle_deg(lat_a, lon_a, lat_b, lon_b)
    km_given = _round_half_up(angle * 60.0 * KM_PER_NM, 0)
    lo = _delta_lon_for_km(lat_a, lat_b, km_given - 1)
    hi = _delta_lon_for_km(lat_a, lat_b, km_given + 1)
    sensitivity = abs(hi - lo) / 2.0
    return sensitivity < 0.02


def _inverse_question(rng):
    """The exam tier: invert the spherical law of cosines. The learner is
    given both latitudes and the great-circle distance (in km) and must find
    the longitude difference -- a genuinely different decision (solving for
    the longitude term), never a bigger number."""
    place_a, lat_a, lon_a, place_b, lat_b, lon_b = rng.choice(INVERSE_LONGITUDE_ROWS)
    check(lon_a != lon_b, f"{place_a}/{place_b}: longitudes must differ (ACMEM161)")

    angle = _central_angle_deg(lat_a, lon_a, lat_b, lon_b)
    _verify_central_angle(lat_a, lon_a, lat_b, lon_b, angle)
    check(5.0 < angle < 150.0, f"central angle {angle} outside the plausible (5, 150) range")

    nm_raw = angle * 60.0
    km_raw = nm_raw * KM_PER_NM
    km_given = int(_round_half_up(km_raw, 0))  # this is the number the learner is GIVEN

    lat_ar, lat_br = math.radians(lat_a), math.radians(lat_b)

    # The correct route: km -> nm -> degrees of central angle (shared
    # `_angle_deg_from_km`), THEN invert (shared `_invert_delta_lon_deg`) --
    # the SAME two routines `_well_conditioned` calls via `_delta_lon_for_km`,
    # so construction and the conditioning filter can never silently diverge.
    angle_from_given = _angle_deg_from_km(km_given)
    delta_lon_raw = _invert_delta_lon_deg(lat_a, lat_b, angle_from_given)
    delta_lon = _round_half_up(delta_lon_raw, 1)

    # Independent check: forward-simulate from the ANSWER back to a central
    # angle via the same law-of-cosines formula construction itself used,
    # and confirm it recovers `angle_from_given` -- a route independent of
    # the inversion algebra just performed (mirrors
    # `p9b_earth_distance_time_build.py`'s own `_verify_inverse` idiom).
    recovered_angle = _central_angle_deg(lat_a, 0.0, lat_b, delta_lon_raw)
    check(
        abs(recovered_angle - angle_from_given) < 1e-6,
        f"forward re-derivation {recovered_angle} != given angle {angle_from_given}",
    )

    lat_a_disp, lat_b_disp = _lat_display(lat_a), _lat_display(lat_b)
    answer_text = _fmt_1dp(delta_lon)
    text = (
        f"{place_a} is at latitude {lat_a_disp} and {place_b} is at latitude {lat_b_disp}. "
        f"{SPHERICITY_NOTE} The great-circle distance between {place_a} and {place_b} is "
        f"{km_given} km. What is the difference in longitude between {place_a} and "
        f"{place_b}, to 1 decimal place, in degrees?"
    )
    working = [
        f"{km_given} km / 1.852 = {km_given / KM_PER_NM:.2f} nm -> central angle = {angle_from_given:.4f} degrees",
        f"cos(Δlon) = (cos(central angle) - sin({lat_a_disp})sin({lat_b_disp})) / (cos({lat_a_disp})cos({lat_b_disp}))",
        f"Δlon = {delta_lon_raw:.4f} -> {answer_text} degrees to 1 decimal place",
    ]

    # Misconceptions -- each a genuinely different reasoning slip, checked
    # directly every seed.
    # 1. forgot the km -> nm step: uses km directly as if it were nm.
    angle_wrong_units = km_given / 60.0
    delta_lon_wrong_units = _round_half_up(_invert_delta_lon_deg(lat_a, lat_b, angle_wrong_units), 1)
    # 2. wrong hemisphere sign: flips lat_b's sign in the inversion.
    lat_br_wrong = math.radians(-lat_b)
    cos_arg_wrong_hemi = (math.cos(math.radians(angle_from_given)) - math.sin(lat_ar) * math.sin(lat_br_wrong)) / (
        math.cos(lat_ar) * math.cos(lat_br_wrong)
    )
    cos_arg_wrong_hemi = max(-1.0, min(1.0, cos_arg_wrong_hemi))
    delta_lon_wrong_hemi = _round_half_up(math.degrees(math.acos(cos_arg_wrong_hemi)), 1)
    # 3. reports the central angle itself as the longitude difference
    # (never inverts the formula at all).
    delta_lon_no_invert = _round_half_up(angle_from_given, 1)

    check_distinct(delta_lon, delta_lon_wrong_units, delta_lon_wrong_hemi, delta_lon_no_invert)
    candidates = [
        _fmt_1dp(delta_lon_wrong_units),
        _fmt_1dp(delta_lon_wrong_hemi),
        _fmt_1dp(delta_lon_no_invert),
    ]
    distractors = pick_distractors(answer_text, candidates, tolerance=stated_precision_tolerance(1))

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=delta_lon,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


@generator(
    "p13d_great_circle_distance",
    level=12,
    presets={
        "easy": dict(mode="same_hemisphere"),
        "medium": dict(mode="opposite_hemisphere"),
        "hard": dict(mode="combined_km"),
        "exam": dict(mode="inverse_longitude"),
    },
)
def p13d_great_circle_distance(rng, mode="same_hemisphere"):
    """Great-circle distance between two places on Earth at DIFFERENT
    longitudes, via the spherical law of cosines -- never the same-longitude
    meridian-arc case, which is `earth_meridian_distance`
    (`senior-find-distances-between-two-places-on-earth-on`, ACMEM160). A
    learner cannot answer any tier of THIS question using only that
    sibling's meridian-arc method, because the two places here are never on
    the same meridian -- the meridian formula does not apply at all; this is
    a different formula, not a stylistic variant of it."""
    if mode not in ("same_hemisphere", "opposite_hemisphere", "combined_km", "inverse_longitude"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "same_hemisphere":
        return _forward_question(rng, SAME_HEMISPHERE, unit="nm")
    if mode == "opposite_hemisphere":
        return _forward_question(rng, OPPOSITE_HEMISPHERE, unit="nm")
    if mode == "combined_km":
        return _forward_question(rng, SAME_HEMISPHERE + OPPOSITE_HEMISPHERE, unit="km")
    return _inverse_question(rng)
