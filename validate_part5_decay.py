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
"""Part 5: the ingredients of the spatial decay estimate. Lattice counts (6 plaquettes per link, 18 neighbours, n_ej <= 1),
plaquette Hessian norms <= 1 per unit N beta in the metric -Tr, the Schur row bound J_alpha = N beta (6 + 18 e^alpha) for the
weighted link Hessian on the actual 4D lattice, the choice of the split time s, and an exact Gaussian toy in which the
covariance bound of Part 5 is compared with the true covariance."""
import math
import sys
import numpy as np
from common import Report
from lattice import adjacency_counts, bfs_dist
from su import su_basis, haar, ip


def main():
    R = Report("Part 5 (decay estimate)")
    rng = np.random.default_rng(11)
    idx, plaqs, per_link, n = adjacency_counts((3, 3, 3, 3))
    R.check("4D, L = 3: every link lies in 6 plaquettes", np.all(per_link == 6))
    R.check("every link has 18 neighbours, each sharing exactly one plaquette", np.all(n.sum(1) == 18) and n.max() == 1)
    # Hessian norms of ReTr(A e^{sX} B e^{uY}) and ReTr(A e^{sX}) at 0
    worst_mix, worst_diag = 0.0, 0.0
    for N in (2, 3):
        Bs = su_basis(N)
        for _ in range(300):
            A, Bm = haar(N, rng), haar(N, rng)
            x = rng.normal(size=len(Bs)); x /= np.linalg.norm(x)
            y = rng.normal(size=len(Bs)); y /= np.linalg.norm(y)
            X = sum(c * b for c, b in zip(x, Bs)); Y = sum(c * b for c, b in zip(y, Bs))
            worst_mix = max(worst_mix, abs(np.real(np.trace(A @ X @ Bm @ Y))))
            worst_diag = max(worst_diag, abs(np.real(np.trace(A @ X @ X))))
    R.check("|mixed Hessian of ReTr U_p| <= 1 for unit link directions (SU(2), SU(3))", worst_mix <= 1 + 1e-12,
            f"max {worst_mix:.3f}")
    R.check("|diagonal Hessian of ReTr U_p| <= 1 for unit link directions", worst_diag <= 1 + 1e-12, f"max {worst_diag:.3f}")
    # weighted Schur row bound on the lattice
    src = [0, 5]
    dist = bfs_dist(n, src)
    ok = True
    for alpha in (0.3, 1.0, 2.0):
        w = np.exp(alpha * dist)
        rows = 6 + (n * (w[:, None] / w[None, :])).sum(1)
        ok &= bool(rows.max() <= 6 + 18 * math.exp(alpha) + 1e-9)
    R.check("weighted row sums <= 6 + 18 e^alpha (units of N beta) for weights e^{alpha d(., A)}", ok)
    # split time
    worst = 0.0
    for _ in range(1000):
        a, C, J, d = rng.uniform(0.1, 3), rng.uniform(0.1, 50), rng.uniform(0.1, 30), rng.integers(1, 40)
        s = a * d / (J + 1 / C); m = a / (1 + C * J)
        worst = max(worst, abs((J * s - a * d) + m * d), abs(s / C - m * d))
    R.check("s = alpha d/(J_alpha + 1/C) equalises both exponents at m_alpha = alpha/(1 + C J_alpha)", worst < 1e-9,
            f"max err {worst:.1e}")
    # Gaussian toy: mu ~ exp(-x^T M x/2) on a ring, generator Laplacian - Mx.grad, C = 1/lambda_min(M)
    nn = 80; a0 = 2.5
    M = a0 * np.eye(nn) - np.eye(nn, k=1) - np.eye(nn, k=-1); M[0, -1] = M[-1, 0] = -1
    Cov = np.linalg.inv(M); C = 1 / np.linalg.eigvalsh(M)[0]
    okg, ratio = True, 0.0
    for alpha in (0.5, 1.0):
        J = a0 + 2 * math.exp(alpha); m = alpha / (1 + C * J)
        for d in range(1, 30):
            bound = (1 / J + math.sqrt(Cov[0, 0] * Cov[d, d])) * math.exp(-m * d)
            okg &= abs(Cov[0, d]) <= bound
            ratio = max(ratio, abs(Cov[0, d]) / bound)
    R.check("Gaussian ring: |Cov(x_0, x_d)| <= [1/J_alpha + sqrt(Var Var)] e^{-m_alpha d}, d = 1..29", okg,
            f"max ratio {ratio:.3f}")
    print(f"     SU(2) explicit rate m_1 = 1/(1 + C N beta (6 + 18 e)) at C = 1, beta = 0.01: "
          f"{1 / (1 + 1 * 2 * 0.01 * (6 + 18 * math.e)):.4f}")
    return R.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
