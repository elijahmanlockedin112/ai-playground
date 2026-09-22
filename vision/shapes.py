"""Polyhedra and the ways a twisty puzzle cuts them up.

A Rubik's cube is a cube whose faces are a 3x3 grid. Change the grid to 2x2
or 4x4 and you have a Pocket Cube or a Rubik's Revenge. Change the solid to
a tetrahedron with triangular pieces and you have a Pyraminx; to a
dodecahedron with pentagonal faces and you have a Megaminx; keep the cube
but cut it corner-to-corner and you have a Skewb.

So this file is two things: the solids (vertices, faces, face normals) and
the subdivisions (how to carve a projected face into pieces). Everything is
done in 2D on the already-projected face, which is an approximation --
strictly the subdivision should happen in 3D and be projected -- but at
sixty-odd pixels across, nobody can tell.
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2


# ----------------------------------------------------------------- solids

def cube():
    verts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    faces = [[0, 1, 2, 3], [5, 4, 7, 6], [4, 0, 3, 7],
             [1, 5, 6, 2], [4, 5, 1, 0], [3, 2, 6, 7]]
    return verts, faces


def tetrahedron():
    verts = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                     dtype=np.float64)
    # wound so that the normal by the right-hand rule points outward
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    return verts, faces


def hull_faces(verts, tol=1e-6):
    """Find the faces of a convex polyhedron from its vertices alone.

    For every triple of vertices, take the plane through them; if all the
    other vertices lie on one side, that plane is a face, and the face is
    every vertex lying on it. Brute force -- 1,140 triples for twenty
    vertices -- but it cannot be wrong, which matters more here than speed,
    and it runs once.

    (The tempting shortcut is to assume the face normals point along some
    known set of directions. For a dodecahedron those directions point at
    its *vertices*, not its faces, and the resulting "faces" are five
    non-coplanar points that render as coloured shrapnel.)
    """
    verts = np.asarray(verts, dtype=np.float64)
    n = len(verts)
    found = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                nrm = np.cross(verts[j] - verts[i], verts[k] - verts[i])
                norm = np.linalg.norm(nrm)
                if norm < tol:
                    continue                      # the three are collinear
                nrm = nrm / norm
                d = verts @ nrm
                if not (d <= d[i] + tol).all():   # something is outside
                    continue
                on = tuple(np.where(np.abs(d - d[i]) < tol)[0].tolist())
                found[on] = nrm

    faces = []
    for idx, nrm in found.items():
        pts = verts[list(idx)]
        centre = pts.mean(0)
        u = pts[0] - centre
        u /= np.linalg.norm(u)
        v = np.cross(nrm, u)
        ang = np.arctan2((pts - centre) @ v, (pts - centre) @ u)
        faces.append([idx[t] for t in np.argsort(ang)])
    return faces


def dodecahedron():
    """Twenty vertices, twelve pentagons."""
    b = 1.0 / PHI
    verts = [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
             for sz in (-1, 1)]
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            verts += [[0, s1 * b, s2 * PHI],
                      [s1 * b, s2 * PHI, 0],
                      [s1 * PHI, 0, s2 * b]]
    verts = np.array(verts, dtype=np.float64)
    return verts, hull_faces(verts)


# ----------------------------------------------------------- subdivisions

def shrink(pts, margin):
    """Pull a polygon in towards its own centre, to leave a gap where the
    puzzle's body shows through. The gaps are most of what makes a twisty
    puzzle look like one."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(0)
    return c + (pts - c) * (1.0 - margin)


def grid_quads(quad, n, margin):
    """An n x n grid of stickers on a quadrilateral face. n=2 pocket cube,
    n=3 Rubik's, n=4 revenge, n=5 professor."""
    a, b, c, d = [np.asarray(p, dtype=np.float64) for p in quad]

    def at(u, v):
        return (1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * d + u * c)

    out = []
    for i in range(n):
        for j in range(n):
            u0, u1 = i / n, (i + 1) / n
            v0, v1 = j / n, (j + 1) / n
            out.append(shrink([at(u0, v0), at(u1, v0), at(u1, v1),
                               at(u0, v1)], margin))
    return out


def grid_triangles(tri, n, margin):
    """A triangular face cut into n^2 small triangles -- the Pyraminx
    pattern. n=3 gives the usual nine per face."""
    a, b, c = [np.asarray(p, dtype=np.float64) for p in tri]

    def at(i, j):
        return a * (1 - i / n) + b * ((i - j) / n) + c * (j / n)

    out = []
    for i in range(n):
        for j in range(i + 1):
            out.append(shrink([at(i, j), at(i + 1, j), at(i + 1, j + 1)],
                              margin))
            if j < i:
                out.append(shrink([at(i, j), at(i + 1, j + 1), at(i, j + 1)],
                                  margin))
    return out


def skewb_face(quad, margin):
    """A Skewb is cut corner to corner, so each face shows a diamond in the
    middle and a triangle at each corner. Same solid as a Rubik's cube,
    completely different pattern -- which makes it a useful thing to have in
    the mix."""
    p = [np.asarray(q, dtype=np.float64) for q in quad]
    mid = [(p[i] + p[(i + 1) % 4]) / 2 for i in range(4)]
    out = [shrink(mid, margin)]
    for i in range(4):
        out.append(shrink([p[i], mid[i], mid[(i - 1) % 4]], margin))
    return out


def megaminx_face(pent, margin):
    """Eleven pieces on a pentagon: one centre, then a ring of ten.

    The ring is the region between the centre pentagon and the outer edge,
    cut at each vertex and each edge midpoint. Splitting it that way keeps
    every piece a convex quad that tiles the ring without overlapping --
    which running the cuts from the face centre does not.
    """
    p = [np.asarray(q, dtype=np.float64) for q in pent]
    c = np.mean(p, axis=0)
    q = [c + (v - c) * 0.42 for v in p]                  # centre pentagon
    out = [shrink(q, margin)]
    for i in range(5):
        j = (i + 1) % 5
        m = (p[i] + p[j]) / 2                            # outer edge midpoint
        mq = (q[i] + q[j]) / 2                           # inner edge midpoint
        out.append(shrink([q[i], p[i], m, mq], margin))
        out.append(shrink([mq, m, p[j], q[j]], margin))
    return out
