"""Generates `r4a_homomorphism_kernel_image` questions (easy, medium, hard) and the exam chain
`exd_x_r4a_homomorphism_kernel_image`.

``uc2-homomorphism-kernel-image`` (``abstract-algebra``, answer ``number``): the order of a kernel or an image of an
explicit homomorphism between small finite abelian groups.

* easy: the order of the kernel of φ: Z_m → Z_n, φ(x) = kx mod n (a homomorphism because n divides mk).
* medium: the order of the image of the power map x → x^k on the group of units of Z_n under multiplication.
* hard: the order of the kernel of φ: Z_a × Z_b → Z_c, φ(x, y) = ux + vy mod c, a homomorphism that is not onto, so
  dividing |Z_a × Z_b| by c is not enough.
* exam: φ: Z_a → Z_b, φ(x) = kx mod b, a homomorphism with a non-trivial kernel: (a) the order of the image, (b) the order of
  the kernel (follows through from (a), by the first isomorphism theorem), (c) how many elements map to a given element
  of the image (follows through from (b)), (d) the least non-negative element that maps to it.

Every key is found by listing the kernel (or the image) element by element and checked against the formula that the first
isomorphism theorem gives: |domain| = |kernel||image|.
"""

from __future__ import annotations

from math import gcd

from ..core import attempts, check, generator, phrase
from ._r4a_alg_kit import number_part, number_question, with_latex


def _is_hom(domain_mod, codomain_mod, k):
    """Whether x -> kx is well defined and additive from Z_domain_mod to Z_codomain_mod."""
    return (domain_mod * k) % codomain_mod == 0


_EASY_STEMS = (
    "The map φ: Z_{m} → Z_{n}, φ(x) = {k}x mod {n}, is a group homomorphism under addition. Find the order of its kernel.",
    "Let φ: Z_{m} → Z_{n} be the homomorphism of additive groups given by φ(x) = {k}x mod {n}. How many elements does "
    "ker φ have?",
    "Multiplication by {k} defines a group homomorphism φ: Z_{m} → Z_{n} under addition, with φ(x) = {k}x mod {n}. "
    "What is the size of its kernel?",
)


def _easy(rng):
    for _ in attempts(1000, "a homomorphism with a proper non-trivial kernel and image"):
        n = rng.choice([12, 15, 18, 20, 24, 28, 30, 36, 40, 42])
        k = rng.randrange(2, n)
        g = gcd(n, k)
        if g == 1 or g == n:
            continue
        step = n // g
        m = step * rng.randint(2, 9)
        if not 2 <= m <= 90 or m == n or gcd(m, n) == 1:
            continue
        ker = [x for x in range(m) if (k * x) % n == 0]
        img = {(k * x) % n for x in range(m)}
        if len(ker) not in (1, m) and len(img) not in (m, n) and len(ker) != len(img):
            break
    check(_is_hom(m, n, k), "the map is not well defined on Z_m")
    check(len(ker) * len(img) == m, "|kernel| × |image| differs from |domain|")
    slips = [
        (len(img), "kernel-order-confused-with-image-order"),
        (m, "kernel-order-whole-domain"),
        (gcd(k, n), "kernel-order-gcd-with-codomain-only"),
        (m // n if m % n == 0 else None, None),
        (1, "kernel-order-assumes-one-to-one"),
    ]
    return number_question(
        phrase(rng, _EASY_STEMS, m=m, n=n, k=k),
        len(ker),
        [
            f"φ(x) = 0 when {n} divides {k}x, that is when {n // g} divides x.",
            f"The multiples of {n // g} in Z_{m} are {', '.join(map(str, ker))}: {len(ker)} elements.",
        ],
        slips,
        lo=1,
    )


_MED_IMAGE = (
    "Let G be the group of units of Z_{n} under multiplication modulo {n}, and let φ: G → G be the homomorphism "
    "φ(x) = x^{k}. Find the order of the image of φ.",
    "The units of Z_{n} form a group G under multiplication modulo {n}. The map φ(x) = x^{k} is a homomorphism G → G. "
    "How many elements does the image of φ have?",
)
_MED_KERNEL = (
    "Let G be the group of units of Z_{n} under multiplication modulo {n}, and let φ: G → G be the homomorphism "
    "φ(x) = x^{k}. Find the order of the kernel of φ.",
    "The units of Z_{n} form a group G under multiplication modulo {n}. The map φ(x) = x^{k} is a homomorphism G → G. "
    "How many elements does the kernel of φ have?",
)


def _medium(rng):
    for _ in attempts(1000, "a power map that is neither one-to-one nor trivial"):
        n = rng.choice(
            [15, 16, 20, 21, 24, 28, 30, 35, 36, 40, 45, 48, 52, 55, 56, 60, 63, 65, 72, 75, 77, 84, 91, 99, 105, 120]
        )
        units = [x for x in range(1, n) if gcd(x, n) == 1]
        k = rng.randint(2, 9)
        image = {pow(x, k, n) for x in units}
        kernel = [x for x in units if pow(x, k, n) == 1]
        if (
            1 < len(image) < len(units)
            and len(kernel) > 1
            and len(image) != len(kernel)
            and len(image) != len(units) // k
        ):
            break
    check(len(image) * len(kernel) == len(units), "|image| × |kernel| differs from |G|")
    ask_image = rng.random() < 0.5
    if ask_image:
        key, other = len(image), len(kernel)
        slips = [
            (other, "image-order-confused-with-kernel-order"),
            (len(units), "image-order-whole-domain"),
            (len(units) // k if len(units) % k == 0 else None, "image-order-divides-by-exponent"),
            (len(units) - other, None),
        ]
        stems = _MED_IMAGE
    else:
        key, other = len(kernel), len(image)
        slips = [
            (other, "kernel-order-confused-with-image-order"),
            (len(units), "kernel-order-whole-domain"),
            (1, "kernel-order-assumes-one-to-one"),
            (k if len(units) % k == 0 else None, None),
            (len(units) - other, None),
        ]
        stems = _MED_KERNEL
    return number_question(
        phrase(rng, stems, n=n, k=k),
        key,
        [
            f"G has {len(units)} elements.",
            f"The kernel {{x : x^{k} = 1}} is {', '.join(map(str, kernel))}, with {len(kernel)} elements, "
            f"so the image has {len(units)}/{len(kernel)} = {len(image)}.",
        ],
        slips,
        lo=1,
    )


_HARD_STEMS = (
    "The map φ: Z_{a} × Z_{b} → Z_{c}, φ(x, y) = {u}x + {v}y mod {c}, is a group homomorphism under addition. Find the order "
    "of its kernel.",
    "Let φ(x, y) = {u}x + {v}y mod {c} define a homomorphism of additive groups Z_{a} × Z_{b} → Z_{c}. How many elements are "
    "in ker φ?",
    "With componentwise addition on Z_{a} × Z_{b}, the map φ(x, y) = {u}x + {v}y mod {c} is a homomorphism Z_{a} × Z_{b} → "
    "Z_{c}. What is the order of its kernel?",
)


def _hard(rng):
    for _ in attempts(3000, "a homomorphism from a product that is not onto"):
        c = rng.choice([12, 15, 18, 20, 24, 30, 36, 42, 60])
        u = rng.randrange(1, c)
        v = rng.randrange(1, c)
        a = c // gcd(c, u) * rng.randint(1, 4)
        b = c // gcd(c, v) * rng.randint(1, 4)
        if not (3 <= a <= 40 and 3 <= b <= 40) or (a * u) % c or (b * v) % c:
            continue
        ker = sum(1 for x in range(a) for y in range(b) if (u * x + v * y) % c == 0)
        img = len({(u * x + v * y) % c for x in range(a) for y in range(b)})
        if (
            img not in (1, c)
            and ker not in (1, a * b)
            and a * b <= 700
            and ker != a * b // c
            and c // img > 1
            and len({ker, img}) == 2
        ):
            break
    check(ker * img == a * b, "|kernel| × |image| differs from |domain|")
    check(img == c // gcd(c, gcd(u, v)), "the image differs from the subgroup generated by u and v")
    slips = [
        (a * b // c if (a * b) % c == 0 else None, "kernel-order-assumes-surjective"),
        (img, "kernel-order-confused-with-image-order"),
        (a * b, "kernel-order-whole-domain"),
        (gcd(a, b), None),
    ]
    return number_question(
        phrase(rng, _HARD_STEMS, a=a, b=b, c=c, u=u, v=v),
        ker,
        [
            f"The image is the subgroup of Z_{c} generated by {u} and {v}, which has {c}/gcd({c}, {u}, {v}) = {img} elements.",
            f"By the first isomorphism theorem |ker φ| = {a}·{b}/{img} = {ker}.",
        ],
        slips,
        lo=1,
    )


@generator(
    "r4a_homomorphism_kernel_image",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_latex
def r4a_homomorphism_kernel_image(rng, tier="easy"):
    """Kernel and image orders of a homomorphism; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r4a_homomorphism_kernel_image", level=19, presets={"exam": dict(tier="exam")})
@with_latex
def exd_x_r4a_homomorphism_kernel_image(rng, tier="exam"):
    """Exam chain: image order, kernel order, fibre size and least preimage for x -> kx from Z_a to Z_b."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a homomorphism with kernel of order 2 to 6 and image of order 4 to 12"):
        b = rng.choice([24, 30, 36, 40, 42, 45, 48, 54, 56, 60, 72])
        k = rng.randrange(2, b)
        g = gcd(b, k)
        img_order = b // g
        t = rng.randint(2, 6)
        a = img_order * t
        if g == 1 or not 4 <= img_order <= 12 or a > 90 or a == b:
            continue
        image = sorted({(k * x) % b for x in range(a)})
        y1, y2 = rng.sample([v for v in image if v != 0], 2) if len(image) > 3 else (None, None)
        if y1 is None:
            continue
        pre = [x for x in range(a) if (k * x) % b == y1]
        pre2 = [x for x in range(a) if (k * x) % b == y2]
        if len(pre) == t and len(pre2) == t and pre[0] not in (y1, 0) and a != img_order:
            break
    y = y1
    check(_is_hom(a, b, k), "the map is not well defined on Z_a")
    check(
        len(image) == img_order and len([x for x in range(a) if (k * x) % b == 0]) == t,
        "image or kernel order differs from the formula",
    )
    check(len(pre) * len(image) == a, "|fibre| × |image| differs from |domain|")
    least = pre[0]
    return dict(
        text=rng.choice(
            [
                f"The map φ: Z_{a} → Z_{b}, φ(x) = {k}x mod {b}, is a group homomorphism under addition.",
                f"Let φ: Z_{a} → Z_{b} be the homomorphism of additive groups given by φ(x) = {k}x mod {b}.",
                f"Multiplication by {k} defines a group homomorphism φ: Z_{a} → Z_{b} under addition, with φ(x) = {k}x mod {b}.",
            ]
        ),
        parts=[
            number_part(
                "Find the order of the image of φ.",
                img_order,
                2,
                [
                    f"The image is the subgroup of Z_{b} generated by {k}, of order {b}/gcd({b}, {k}) = {b}/{g} = {img_order}."
                ],
                [
                    (g, "image-order-read-as-gcd"),
                    (a, "image-order-whole-domain"),
                    (b, "image-order-whole-codomain"),
                    (a // g if a % g == 0 else None, None),
                ],
                sensible_range=(0, None),
            ),
            number_part(
                "Find the order of the kernel of φ.",
                t,
                2,
                [f"By the first isomorphism theorem |ker φ| = |Z_{a}|/|im φ| = {a}/{img_order} = {t}."],
                [
                    (img_order, "kernel-order-confused-with-image-order"),
                    (a, "kernel-order-whole-domain"),
                    (1, "kernel-order-assumes-one-to-one"),
                    (a - img_order, None),
                ],
                follow_through=f"{a}/part_a",
                sensible_range=(0, None),
            ),
            number_part(
                f"Exactly how many elements x of Z_{a} satisfy φ(x) = {y1} or φ(x) = {y2}?",
                2 * t,
                2,
                [
                    f"{y1} and {y2} are both in the image, so each has a preimage that is a coset of the kernel, with |ker φ| = {t} elements."
                ],
                [
                    (2, "preimage-count-assumes-one-to-one"),
                    (t, "preimage-count-one-fibre-only"),
                    (2 * img_order, "kernel-order-confused-with-image-order"),
                    (t + 2, None),
                ],
                follow_through="2*part_b",
                sensible_range=(0, None),
            ),
            number_part(
                f"Find the least non-negative x in Z_{a} with φ(x) = {y}.",
                least,
                3,
                [f"Solve {k}x ≡ {y} (mod {b}); the solutions are {', '.join(map(str, pre))}, the least being {least}."],
                [
                    (y, "preimage-taken-as-the-image"),
                    (pre[-1], "preimage-greatest-not-least"),
                    (least + img_order, None),
                    ((y // g) % img_order, None),
                    (pre[1] if len(pre) > 1 else None, None),
                    (least + 1, None),
                ],
                sensible_range=(0, None),
            ),
        ],
    )
