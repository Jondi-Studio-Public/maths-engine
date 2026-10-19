"""Generates `pv0c_euler_planar` questions: find the missing count of vertices, edges or faces from Euler's
formula.

Euler's formula for planar graphs.

``pv0c_euler_planar`` (``senior-apply-euler-s-formula-v-f-e-2``, ACMGM082,
domain financial-discrete, answer: number) -- given the numbers of any two of
the vertices, edges and faces of a stated connected planar graph (or a
convex polyhedron, whose skeleton is one), find the third using Euler's
formula v + f - e = 2.

Tiers:
- easy: v and e stated, find f (the outer/unbounded face is counted).
- medium: v or e is the unknown (the other two, including f, are given).
- hard: a degree list is stated; find f (count v from the list length, get e
  from the handshake lemma, then apply Euler).
- exam: a face-length list is stated (with the "each edge borders exactly
  two different faces" convention); find v.

The hard and exam tiers read their degree/face-length lists off a REAL plane
graph built in pure Python: a fan-triangulated m-gon (outer face = the
m-gon, inner faces triangles), vertices stellated into triangular faces, and
some edges deleted where the two faces they border share exactly that
edge's two endpoints (so the merge keeps every face a simple cycle and the
graph bridgeless). A pool of such graphs is built once, on the first hard or
exam question: never per `make()` call, to stay well under the 15ms speed
budget, and not at import, where it cost every `import mathgen` about two
seconds. Every
invariant is asserted on every pool entry: the face
lengths sum to 2e, the degrees sum to 2e, v - e + f = 2, and every edge
borders exactly two DIFFERENT faces.
"""

from __future__ import annotations

import functools
import random

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

# ---------------------------------------------------------------------------
# Real-graph construction (hard/exam tiers)
# ---------------------------------------------------------------------------

_FACE_NAME = {3: "triangle", 4: "quadrilateral", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}


def _has_edge(face: tuple, u, v) -> bool:
    """True when u and v are neighbours around the cycle `face` (a simple cycle: each vertex appears once)."""
    if u not in face:
        return False
    i = face.index(u)
    return face[(i + 1) % len(face)] == v or face[i - 1] == v


def _rotate_to_edge(face: tuple, a, b):
    """Rotate `face` (trying it and its reversal) so it starts ``a, b, ...``.
    Returns None if the edge {a, b} is not adjacent anywhere in `face`."""
    n = len(face)
    for seq in (face, tuple(reversed(face))):
        for i in range(n):
            if seq[i] == a and seq[(i + 1) % n] == b:
                return seq[i:] + seq[:i]
    return None


def _build_plane_graph(rng: random.Random, m: int, insertions: int, max_deletions: int):
    """Fan-triangulate an m-gon, stellate `insertions` vertices into
    triangular faces, then delete up to `max_deletions` edges whose two
    bordering faces share exactly that edge's two endpoints (merging them
    into one simple face). Returns (vertices, edges, faces)."""
    vertices = list(range(m))
    edges = set()
    for i in range(m):
        edges.add(frozenset((i, (i + 1) % m)))
    faces = [tuple(range(m))]  # the m-gon boundary -- one designated face
    for i in range(1, m - 1):
        faces.append((0, i, i + 1))
        edges.add(frozenset((0, i)))

    next_vertex = m
    for _ in range(insertions):
        tri_indices = [idx for idx, f in enumerate(faces) if len(f) == 3]
        if not tri_indices:
            break
        idx = rng.choice(tri_indices)
        a, b, c = faces[idx]
        w = next_vertex
        next_vertex += 1
        vertices.append(w)
        edges.update({frozenset((a, w)), frozenset((b, w)), frozenset((c, w))})
        faces.pop(idx)
        faces.extend([(a, b, w), (b, c, w), (c, a, w)])

    deleted = 0
    attempts = 0
    while deleted < max_deletions and attempts < 300:
        attempts += 1
        u, v = tuple(rng.choice(list(edges)))
        face_idxs = [idx for idx, f in enumerate(faces) if _has_edge(f, u, v)]
        if len(face_idxs) != 2:
            continue
        i1, i2 = face_idxs
        f1, f2 = faces[i1], faces[i2]
        if set(f1) & set(f2) != {u, v}:
            continue
        r1 = _rotate_to_edge(f1, u, v)
        r2 = _rotate_to_edge(f2, v, u)
        if r1 is None or r2 is None:
            continue
        merged = (u, *r2[2:], v, *r1[2:])
        for idx in sorted((i1, i2), reverse=True):
            faces.pop(idx)
        faces.append(merged)
        edges.discard(frozenset((u, v)))
        deleted += 1

    return vertices, edges, faces


def _verify_plane_graph(vertices: list, edges: set, faces: list) -> dict:
    """Assert every invariant and return the degree map."""
    v, e, f = len(vertices), len(edges), len(faces)
    check(v - e + f == 2, f"euler_planar/pool: v - e + f = {v - e + f} != 2")

    edge_face_idx: dict = {}
    for idx, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            edge_face_idx.setdefault(frozenset((a, b)), []).append(idx)
    for edge in edges:
        idxs = edge_face_idx.get(edge, [])
        check(
            len(idxs) == 2 and idxs[0] != idxs[1],
            f"euler_planar/pool: edge {tuple(edge)} does not border exactly two different faces",
        )

    check(sum(len(fc) for fc in faces) == 2 * e, "euler_planar/pool: face lengths do not sum to 2e")

    degree = {x: 0 for x in vertices}
    for edge in edges:
        for x in edge:
            degree[x] += 1
    check(sum(degree.values()) == 2 * e, "euler_planar/pool: degree sum != 2e")
    check(all(d >= 1 for d in degree.values()), "euler_planar/pool: an isolated vertex was produced")
    return degree


def _build_euler_pool(n_graphs: int = 2400) -> list[dict]:
    """A fixed, deterministic pool of real graphs, built once by `_euler_pool()`
    (never per `make()` call) -- see the module docstring.

    The exam tier's answer is this pool entry's own `v`, so `v` is what must
    spread widely: each draw fixes a TARGET `v` first,
    uniformly over 6..30, then picks `m` (the outer m-gon) and `insertions`
    so that `m + insertions == target_v` exactly (every insertion adds
    exactly one vertex; deletions never remove one, so this is exact, not
    approximate). `m` is kept small (<=8) because a bigger outer m-gon is a
    bigger face, and the `face_lengths[-1] > 8` plausibility filter below
    discards it anyway -- so most of the spread from 6..30 comes from
    `insertions`, not `m`. `max_deletions` still varies independently so
    `e` and `f` (and hence the hard tier's degree-list route to `f`) keep
    spreading even though `v` is now fixed for a given draw.
    """
    rng = random.Random("pv0c_euler_planar_pool_v2")
    pool = []
    for _ in range(n_graphs):
        target_v = rng.randint(6, 30)
        m = rng.randint(4, min(8, target_v))
        insertions = target_v - m
        max_deletions = rng.randint(0, 10 + insertions)
        vertices, edges, faces = _build_plane_graph(rng, m, insertions, max_deletions)
        face_lengths = sorted(len(fc) for fc in faces)
        if face_lengths[-1] > 8:
            continue
        degree = _verify_plane_graph(vertices, edges, faces)
        v, e, f = len(vertices), len(edges), len(faces)
        check(v == target_v, f"euler_planar/pool: built v={v} != target_v={target_v}")
        if not (6 <= v <= 30):
            continue
        if f < 3:
            continue
        degrees = sorted(degree.values())
        # Discard degenerate results (a bare polygon, every vertex degree 2,
        # or every face the same length) -- real but uninteresting, and
        # over-represented enough in the raw construction to push a single
        # answer value over the 25%-of-draws concentration bound.
        if len(set(degrees)) < 2 or len(set(face_lengths)) < 2:
            continue
        pool.append(dict(v=v, e=e, f=f, degrees=degrees, face_lengths=face_lengths))
    check(len(pool) >= 30, "euler_planar: pool build produced too few usable graphs")
    return pool


@functools.cache
def _euler_pool() -> tuple[dict, ...]:
    """The pool, built on first use rather than at import: building it takes about two seconds, which every
    `import mathgen` would otherwise pay. Its own fixed seed makes it the same whenever it is built."""
    return tuple(_build_euler_pool())


# ---------------------------------------------------------------------------
# Wording banks
# ---------------------------------------------------------------------------

_EASY_TEMPLATES = (
    "A road network connecting several towns forms a connected planar graph with {v} vertices "
    "(towns) and {e} edges (roads); no two roads cross. Using Euler's formula v + f - e = 2, where "
    "the count of faces includes the outer, unbounded region, find the number of faces f.",
    "A printed circuit board's connections form a connected planar graph, drawn without any wires "
    "crossing, with {v} vertices and {e} edges. Counting the outer, unbounded region as one of the "
    "faces, use v + f - e = 2 to find the number of faces f.",
    "A flat mesh of connected line segments has {v} vertices and {e} edges, drawn in the plane with "
    "no two edges crossing. The number of faces f (which includes the single outer, unbounded face) "
    "satisfies v + f - e = 2. Find f.",
    "A convex polyhedron has {v} vertices and {e} edges. Using Euler's formula v + f - e = 2 (here f "
    "is simply the number of faces of the polyhedron, with no outer face to add), find the number of "
    "faces f.",
)

_MEDIUM_TEMPLATES_E = (
    "A connected planar graph, drawn without crossing edges, has {v} vertices and {f} faces "
    "(including the outer, unbounded face). Using v + f - e = 2, find the number of edges e.",
    "A map divided into regions forms a connected planar graph with {v} vertices where the regions "
    "plus the outer, unbounded region give {f} faces in total. No two edges cross. Use Euler's "
    "formula v + f - e = 2 to find the number of edges e.",
    "A connected network of paths, drawn in the plane with no two paths crossing, has {v} vertices "
    "and {f} faces (the outer, unbounded region counted as one face). Find the number of edges e "
    "using v + f - e = 2.",
)

_MEDIUM_TEMPLATES_V = (
    "A connected planar graph, drawn without crossing edges, has {e} edges and {f} faces "
    "(including the outer, unbounded face). Using v + f - e = 2, find the number of vertices v.",
    "A mesh of connected line segments, with no two edges crossing, has {e} edges and {f} faces in "
    "total (the outer, unbounded region counted as one face). Find the number of vertices v using "
    "Euler's formula v + f - e = 2.",
    "A convex polyhedron has {e} edges and {f} faces. Using v + f - e = 2 (with no outer face to "
    "add), find the number of vertices v.",
)

_HARD_TEMPLATES = (
    "A connected network is drawn in the plane with no two edges crossing. Every vertex's number of "
    "connections (its degree) is stated: {degree_text}. Using Euler's formula v + f - e = 2 (the "
    "number of faces f includes the outer, unbounded region), find f.",
    "A circuit board's connected layout, drawn without crossing wires, has these vertex degrees: "
    "{degree_text}. Find the number of faces f (including the outer, unbounded region) using "
    "v + f - e = 2.",
    "A connected mesh, with no two edges crossing, has vertices of the following degrees: "
    "{degree_text}. Using Euler's formula v + f - e = 2 (the number of faces f includes the outer, "
    "unbounded region), find f.",
)

_EXAM_TEMPLATES = (
    "A connected planar graph is drawn in the plane with no two edges crossing, and every edge lies "
    "on the boundary of exactly two different faces. The outer face is bounded by {outer} edges; of "
    "the other faces, {face_text}. Find the number of vertices v, given that v + f - e = 2.",
    "A road network divides the plane into regions; every edge borders exactly two different faces. "
    "The outer, unbounded region is bounded by {outer} edges, and among the remaining regions, "
    "{face_text}. Find the number of vertices v, given that v + f - e = 2.",
    "A connected mesh has no two edges crossing, and each edge lies on the boundary of exactly two "
    "different faces. The outer face has {outer} sides; of the other faces, {face_text}. Find the "
    "number of vertices v, where v + f - e = 2.",
)


def _degree_text(degrees: list[int]) -> str:
    counts: dict = {}
    for d in degrees:
        counts[d] = counts.get(d, 0) + 1
    parts = [f"{n} {'vertex' if n == 1 else 'vertices'} of degree {d}" for d, n in sorted(counts.items())]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _face_text(lengths: list[int]) -> str:
    counts: dict = {}
    for length in lengths:
        counts[length] = counts.get(length, 0) + 1
    parts = []
    for length, n in sorted(counts.items()):
        name = _FACE_NAME[length]
        if n == 1:
            article = "an" if name[0] in "aeiou" else "a"
            parts.append(f"1 is {article} {name}")
        else:
            parts.append(f"{n} are {name}s")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# ---------------------------------------------------------------------------
# Distractor formulas
# ---------------------------------------------------------------------------


def _euler_variants(unknown: str, v: int, e: int, f: int):
    """The three shared misconception values for a stated (v, e, f) triple
    -- forgot the outer face, dropped the "+2" constant, and a sign error in
    the rearrangement -- each solved for `unknown` (never sharing the true
    formula's own arithmetic)."""
    if unknown == "f":
        forgot_outer_face = f - 1
        forgot_constant = e - v
        sign_error = v + e - 2
    elif unknown == "e":
        forgot_outer_face = e - 1
        forgot_constant = v + f
        sign_error = 2 - v + f
    elif unknown == "v":
        forgot_outer_face = v + 1
        forgot_constant = e - f
        sign_error = 2 + f - e
    else:
        raise ValueError(unknown)
    return forgot_outer_face, forgot_constant, sign_error


def _positive_candidates(*values) -> list[str]:
    return [str(x) for x in values if isinstance(x, int) and x > 0]


@generator(
    "pv0c_euler_planar",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def pv0c_euler_planar(rng: random.Random, tier: str = "easy"):
    """Apply Euler's formula v + f - e = 2 to a connected planar graph (or a
    convex polyhedron), solving for whichever of v, e, f is not given."""

    if tier == "easy":
        # Draw the ANSWER (f) first, uniformly wide, then build a valid
        # (v, e) pair that realises it (construct to target).
        f = rng.randint(2, 30)
        v_lo = max(4, -(-(f + 4) // 2))  # ceil((f + 4) / 2)
        v_hi = f + 12
        v = rng.randint(v_lo, v_hi)
        e = v + f - 2
        check(v - 1 <= e <= 3 * v - 6, f"euler_planar/easy: (v={v}, e={e}) not realisable for f={f}")

        text = phrase(rng, _EASY_TEMPLATES, v=v, e=e)
        answer = f
        unknown = "f"

    elif tier == "medium":
        unknown = rng.choice(["v", "e"])
        if unknown == "e":
            e = rng.randint(6, 60)
            v_lo = max(4, -(-(e + 6) // 3))  # ceil((e + 6) / 3)
            v_hi = min(e, 60)  # e >= v (never e = v - 1) so f = 2 - v + e >= 2, never "1 face"
            check(v_lo <= v_hi, "euler_planar/medium: no valid v for drawn e")
            v = rng.randint(v_lo, v_hi)
            f = 2 - v + e
            check(v - 1 <= e <= 3 * v - 6, f"euler_planar/medium: (v={v}, e={e}) not realisable")
            check(f >= 2, f"euler_planar/medium: f={f} degenerate (a single-face tree)")
            text = phrase(rng, _MEDIUM_TEMPLATES_E, v=v, f=f)
            answer = e
        else:
            v = rng.randint(4, 40)
            e = rng.randint(v, 3 * v - 6)  # e >= v (never e = v - 1), same reasoning
            f = 2 - v + e
            check(f >= 2, f"euler_planar/medium: f={f} degenerate (a single-face tree)")
            text = phrase(rng, _MEDIUM_TEMPLATES_V, e=e, f=f)
            answer = v

    elif tier == "hard":
        entry = rng.choice(_euler_pool())
        v, e, f = entry["v"], entry["e"], entry["f"]
        degree_text = _degree_text(entry["degrees"])
        check(sum(entry["degrees"]) == 2 * e, "euler_planar/hard: degree list does not match e")
        text = phrase(rng, _HARD_TEMPLATES, degree_text=degree_text)
        answer = f
        unknown = "f"

    elif tier == "exam":
        entry = rng.choice(_euler_pool())
        v, e, f = entry["v"], entry["e"], entry["f"]
        lengths = entry["face_lengths"]
        outer = lengths[-1]
        others = lengths[:-1]
        check(len(others) == f - 1, "euler_planar/exam: face-length list length mismatch")
        check(outer + sum(others) == 2 * e, "euler_planar/exam: face-length list does not match e")
        face_text = _face_text(others)
        text = phrase(rng, _EXAM_TEMPLATES, outer=outer, face_text=face_text)
        answer = v
        unknown = "v"

    else:
        raise ValueError(f"unknown tier {tier!r}")

    forgot_outer_face, forgot_constant, sign_error = _euler_variants(unknown, v, e, f)
    extra = []
    if tier == "hard":
        e_wrong = sum(entry["degrees"])
        extra.append(2 - v + e_wrong)  # degree-sum-as-edges
    if tier == "exam":
        e_wrong = sum(lengths)  # face-sum-as-edges: forgets each face-length sum is 2e, not e
        extra.append(2 + e_wrong - f)

    candidate_ints = [forgot_outer_face, forgot_constant, sign_error, *extra]
    candidates = _positive_candidates(*candidate_ints)
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, candidates)
    check_pairwise_distinct(
        answer_text, *distractors, msg=f"euler_planar/{tier}: final options collided for answer={answer}"
    )
    working = [
        f"v = {v}, e = {e}, f = {f} (the outer, unbounded face is one of the {f}).",
        "By Euler's formula, v + f - e = 2.",
        f"So the requested value is {answer}.",
    ]

    return {
        "text": text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }
