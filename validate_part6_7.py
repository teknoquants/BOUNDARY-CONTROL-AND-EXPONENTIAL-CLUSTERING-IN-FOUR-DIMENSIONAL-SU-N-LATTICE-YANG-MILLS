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
"""Parts 6 and 7. (B-1) implies (A): if sup_T int_0^T [b(s) - rho + delta] ds <= H then C^{B,eta} <= e^{2H}/delta, tested on
random signed profiles. Negative control: two plaquette variables X_p = ReTr U_p / 2 under product Haar measure have mean 0,
variance 1/4 and covariance 0; the density 1 + eps X_p X_q is positive and normalised and gives covariance eps/16 at every
separation."""
import math
import sys
import numpy as np
from scipy.integrate import quad
from common import Report


def main():
    R = Report("Parts 6-7 (reduction and control)")
    rng = np.random.default_rng(3)
    rho = 1.0; ok = True; worst = 0.0
    for _ in range(200):
        t = np.linspace(0, 60, 60001); dt = t[1] - t[0]
        delta = rng.uniform(0.05, 0.9)
        # adverse bump plus favourable stretch; b - rho + delta has positive and negative parts
        b = rho - delta + rng.uniform(0, 3) * np.exp(-((t - rng.uniform(0, 5)) / rng.uniform(0.1, 1)) ** 2) \
            - rng.uniform(0, 1) * np.exp(-((t - rng.uniform(5, 10)) / rng.uniform(0.2, 2)) ** 2)
        g = b - rho + delta
        G = np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * dt)])
        H = max(G.max(), 0.0)
        kap = rho - b
        Lam = np.concatenate([[0], np.cumsum(0.5 * (kap[1:] + kap[:-1]) * dt)])
        Cb = 2 * np.sum(0.5 * (np.exp(-2 * Lam[1:]) + np.exp(-2 * Lam[:-1])) * dt) + np.exp(-2 * Lam[-1]) / delta
        ok &= Cb <= math.exp(2 * H) / delta * (1 + 1e-6)
        worst = max(worst, Cb * delta / math.exp(2 * H))
    R.check("(B-1) with constants (delta, H) gives C^{B,eta} <= e^{2H}/delta (200 random signed profiles)", ok,
            f"max C delta e^(-2H) = {worst:.3f}")
    # Haar marginal of x = cos(theta) on SU(2): density (2/pi) sqrt(1 - x^2) on [-1, 1]
    w = lambda x: (2 / math.pi) * math.sqrt(1 - x * x)
    m0 = quad(w, -1, 1)[0]; m1 = quad(lambda x: x * w(x), -1, 1)[0]; m2 = quad(lambda x: x * x * w(x), -1, 1)[0]
    R.check("Haar: E X = 0, E X^2 = 1/4", abs(m0 - 1) < 1e-10 and abs(m1) < 1e-10 and abs(m2 - 0.25) < 1e-10,
            f"{m0:.10f}, {m1:.1e}, {m2:.10f}")
    for eps in (0.2, 0.9):
        cov = eps * m2 * m2
        norm = 1 + eps * m1 * m1
        R.check(f"eps = {eps}: density normalised, positive (|X_p X_q| <= 1), Cov = eps/16 at every separation",
                abs(norm - 1) < 1e-10 and abs(cov - eps / 16) < 1e-10 and 1 - eps > 0, f"Cov {cov:.6f}")
    # Monte Carlo check with independent SU(2) Haar angles, reweighted
    N = 400000
    def haar_x(k):
        q = rng.normal(size=(k, 4)); q /= np.linalg.norm(q, axis=1)[:, None]
        return q[:, 0]
    xp, xq = haar_x(N), haar_x(N)
    wgt = 1 + 0.9 * xp * xq
    mc = np.sum(wgt * xp * xq) / np.sum(wgt)
    R.check("Monte Carlo (Haar quaternions, reweighted): Cov = 0.9/16 within 4 standard errors",
            abs(mc - 0.9 / 16) < 4 * 0.25 / math.sqrt(N) * 1.2, f"{mc:.5f} vs {0.9 / 16:.5f}")
    return R.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
