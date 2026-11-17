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
"""Part 2: the one-link SU(2) problem. Reduction to the Challenge A one-loop weight with beta_eff = 2 beta |a| <= 12 beta,
the radial/tangential Hessian eigenvalues, the Challenge A inequalities Q_x >= 0 >= Q_xx, equality of full-link and radial
constants (both attained at the antipode), and monotonicity of the one-loop constant in beta_eff (numerical)."""
import math
import sys
import numpy as np
from common import Loop, logC_accurate, Report
from su import su_basis, haar, su2_to_quat, angle, expm


def main():
    R = Report("Part 2 (one link)")
    rng = np.random.default_rng(20261117)
    beta = 0.37
    # (a) reduction: N beta ReTr(UA) = 2 beta_eff cos(theta(US)), beta_eff = 2 beta |a|
    err = 0.0; amax = 0.0
    for _ in range(200):
        A = sum(haar(2, rng) for _ in range(6))
        a = su2_to_quat(A); na = np.linalg.norm(a); amax = max(amax, na)
        S = A / na
        U = haar(2, rng)
        lhs = 2 * beta * np.real(np.trace(U @ A))
        rhs = 2 * (2 * beta * na) * math.cos(angle(U @ S))
        err = max(err, abs(lhs - rhs))
        err = max(err, abs(np.linalg.det(S) - 1))
    R.check("N beta ReTr(UA) = 2 beta_eff cos theta(US), beta_eff = 2 beta |a|, S in SU(2)", err < 1e-12, f"max err {err:.1e}")
    A = 6 * np.eye(2)
    R.check("all exterior links = identity: |a| = 6, beta_eff = 12 beta (factor 12 attained)",
            abs(np.linalg.norm(su2_to_quat(A)) - 6) < 1e-15 and amax <= 6, f"random max |a| = {amax:.3f}")
    # (b) twice the Hessian of Q(x), x = ReTr(U)/2, in the metric -Tr has eigenvalues (1-x^2)Q'' - xQ' (once) and -xQ' (twice)
    Q = lambda x: math.sin(1.3 * x) + 0.7 * x * x
    Q1 = lambda x: 1.3 * math.cos(1.3 * x) + 1.4 * x
    Q2 = lambda x: -1.69 * math.sin(1.3 * x) + 1.4
    B = su_basis(2); h = 1e-4; worst = 0.0
    for _ in range(25):
        U = haar(2, rng); x = np.real(np.trace(U)) / 2
        f = lambda Y: Q(np.real(np.trace(U @ expm(Y))) / 2)
        H = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Y1, Y2 = B[i] * h, B[j] * h
                H[i, j] = (f(Y1 + Y2) - f(Y1 - Y2) - f(-Y1 + Y2) + f(-Y1 - Y2)) / (4 * h * h)
        ev = np.sort(np.linalg.eigvalsh(2 * 0.5 * (H + H.T)))
        pred = np.sort([(1 - x * x) * Q2(x) - x * Q1(x), -x * Q1(x), -x * Q1(x)])
        worst = max(worst, np.max(np.abs(ev - pred)))
    R.check("2 Hess Q(x): radial (1-x^2)Q_xx - xQ_x, tangential -xQ_x (twice)", worst < 1e-5, f"max err {worst:.1e}")
    # (c) Challenge A inequalities and attainment at the antipode
    th = np.linspace(0.04, math.pi - 0.04, 241)
    okx, okxx, okmax, wst = True, True, True, 0.0
    for be in (0.1, 0.5, 1.2, 3.0):
        L = Loop(be)
        for t in (0.01, 0.1, 0.5, 2.0):
            _, q1, q2 = L.derivs(th, t)
            s, c = np.sin(th), np.cos(th)
            Qx = -q1 / s
            Qxx = (q2 + c * Qx) / s ** 2
            okx &= bool(np.all(Qx >= -1e-9)); okxx &= bool(np.all(Qxx <= 1e-7 * (1 + np.abs(Qx))))
            x = c
            rad = (1 - x * x) * Qxx - x * Qx; tan = -x * Qx
            qa = L.derivs(np.array([math.pi]), t)[2][0]
            okmax &= bool(rad.max() <= qa + 1e-7 and tan.max() <= qa + 1e-7)
            wst = max(wst, rad.max() - qa, tan.max() - qa)
    R.check("Q_x >= 0 along the flow (beta_eff in {0.1,0.5,1.2,3}, 4 heat times)", okx)
    R.check("Q_xx <= 0 along the flow", okxx)
    R.check("radial and tangential maxima both equal Q_x(t,-1): full-link constant = radial constant", okmax,
            f"max excess {wst:.1e}")
    # (d) monotonicity of the one-loop constant in beta_eff [numerical]
    grid = [0.0, 0.06, 0.12, 0.1668, 0.24, 0.5, 1.0, 2.0]
    logs = [math.log(2 / 2) if b == 0 else logC_accurate(b) for b in grid]
    mono = all(y2 > y1 for y1, y2 in zip(logs, logs[1:]))
    R.check("C_kappa,loop(beta_eff) increasing on the tested grid [numerical]", mono,
            ", ".join(f"{b:g}:{math.exp(y):.4g}" for b, y in zip(grid, logs)))
    for b in (0.005, 0.01, artanh := math.atanh(1 / 18) / 4, 0.02):
        print(f"     beta = {b:.5f}: necessary bound K C_kappa >= C_kappa,loop(12 beta) = {math.exp(logC_accurate(12 * b)):.5f}")
    return R.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
