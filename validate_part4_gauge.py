# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Part 4: horizontal directions after boundary gauge fixing. Gauge action U_e -> g_s U_e g_t^{-1}; in the left-trivialised
frame (D_U xi)_e = Ad_{U_e^{-1}} xi_{s(e)} - xi_{t(e)}, with xi = 0 at frozen (boundary) vertices. H_B = ker D_U^*.
Checks: (i) a single link with both ends frozen has every direction horizontal (dim 3), while for the full gauge group the
horizontal space is 0; (ii) a three-link path with two interior vertices has dim H_B = 9 - 6 = 3; interior gauge leaves the
ordered path product unchanged, a released endpoint changes it, and H_B maps onto su(2) through the path product."""
import sys
import numpy as np
from common import Report
from su import su_basis, haar, ip, expm


def ad_matrix(U, B):
    """Matrix of Ad_{U^{-1}} in the orthonormal basis B."""
    Ui = U.conj().T
    return np.array([[ip(B[i], Ui @ B[j] @ U) for j in range(len(B))] for i in range(len(B))])


def D_matrix(Us, edges, free_vertices, B):
    k = len(B)
    col = {v: i for i, v in enumerate(free_vertices)}
    D = np.zeros((k * len(edges), k * len(free_vertices)))
    for a, ((s, t), U) in enumerate(zip(edges, Us)):
        if s in col:
            D[k * a:k * a + k, k * col[s]:k * col[s] + k] += ad_matrix(U, B)
        if t in col:
            D[k * a:k * a + k, k * col[t]:k * col[t] + k] -= np.eye(k)
    return D


def main():
    R = Report("Part 4 (boundary gauge fixing)")
    rng = np.random.default_rng(7); B = su_basis(2)
    U = [haar(2, rng)]
    D_int = D_matrix(U, [(0, 1)], [], B)
    D_full = D_matrix(U, [(0, 1)], [0, 1], B)
    hb_int = 3 - (np.linalg.matrix_rank(D_int) if D_int.size else 0)
    hb_full = 3 - np.linalg.matrix_rank(D_full)
    R.check("single link, both ends frozen: dim H_B = 3 (every direction horizontal)", hb_int == 3)
    R.check("single link, full gauge group: horizontal dimension 0", hb_full == 0)
    # three-link path v0 - v1 - v2 - v3, endpoints frozen
    Us = [haar(2, rng) for _ in range(3)]
    edges = [(0, 1), (1, 2), (2, 3)]
    D = D_matrix(Us, edges, [1, 2], B)
    r = np.linalg.matrix_rank(D)
    R.check("three-link path: D_U injective on interior xi, dim H_B = 9 - 6 = 3", r == 6, f"rank {r}")
    # finite gauge transformations and the ordered product
    P = lambda V: V[0] @ V[1] @ V[2]
    def gauge(g):
        return [g[s] @ V @ g[t].conj().T for (s, t), V in zip(edges, Us)]
    I = np.eye(2)
    g_int = {0: I, 1: haar(2, rng), 2: haar(2, rng), 3: I}
    g_end = {0: I, 1: I, 2: I, 3: haar(2, rng)}
    R.check("interior gauge transformation leaves the path product unchanged", np.allclose(P(gauge(g_int)), P(Us), atol=1e-12))
    R.check("released endpoint transformation changes the path product", not np.allclose(P(gauge(g_end)), P(Us), atol=1e-6))
    # differential of the path product on H_B (left-trivialised): rank 3 means H_B carries the endpoint-frame modes
    _, s, Vt = np.linalg.svd(D.T)
    HB = Vt[r:].T                              # orthonormal basis of ker D^T (9 x 3)
    h = 1e-6; P0 = P(Us); J = np.zeros((3, 3))
    for c in range(3):
        v = HB[:, c]
        Vp = [Us[a] @ expm(h * sum(v[3 * a + i] * B[i] for i in range(3))) for a in range(3)]
        dP = P0.conj().T @ (P(Vp) - P0) / h
        J[:, c] = [ip(B[i], dP) for i in range(3)]
    R.check("path product restricted to H_B has full rank 3 (endpoint-frame degrees of freedom survive)",
            np.linalg.matrix_rank(J, tol=1e-4) == 3)
    return R.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
