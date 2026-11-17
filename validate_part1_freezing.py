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
"""Part 1: freezing and heat smoothing do not commute. Two SU(2) links e (interior) and j (exterior) with weight
phi(U_e U_j) = exp(2 beta cos theta(U_e U_j)). Smoothing both links gives (P_{2t} phi)(U_e U_j); smoothing only e with j frozen
gives (P_t phi)(U_e U_j). They agree at t = 0 and differ for t > 0, in value and in the Hessian along e."""
import math
import sys
import numpy as np
from scipy.special import iv
from common import Report


def heat_class(beta, s, th):
    """(P_s e^{2 beta cos})(theta) by the character series, up to a common positive factor.
    chi_n(U_e U_j) is a matrix coefficient of the spin-n/2 representation in each link separately, hence an eigenfunction of
    each link Laplacian with eigenvalue n(n+2)/2; smoothing both links therefore multiplies it by exp(-n(n+2)t)."""
    n = np.arange(0, 120)
    c = (iv(n, 2 * beta) - iv(n + 2, 2 * beta))          # coefficient of chi_n / (n+1) up to a common factor
    chi = np.sin((n + 1) * th) / np.sin(th)
    return float(np.sum(c * np.exp(-n * (n + 2) * s / 2) * chi))


def main():
    R = Report("Part 1 (freezing vs smoothing)")
    beta = 1.0
    # check the series at s = 0 reproduces the weight up to the common factor
    ths = np.linspace(0.3, 2.8, 6)
    r0 = [heat_class(beta, 0.0, th) / math.exp(2 * beta * math.cos(th)) for th in ths]
    R.check("character series reproduces e^{2 beta cos theta} at t = 0 (common factor)", np.ptp(r0) < 1e-10 * abs(r0[0]),
            f"factor {r0[0]:.6f}")
    fac = r0[0]
    for t in (0.05, 0.3):
        d = [abs(math.log(heat_class(beta, 2 * t, th) / fac) - math.log(heat_class(beta, t, th) / fac)) for th in ths]
        R.check(f"t = {t}: -log P_t e^W (both links) differs from frozen-boundary V_t^(B,eta)", min(d) > 1e-3,
                f"min |difference| {min(d):.3e}")
    # Hessian along e at the antipode: second theta-derivative of log, with step h
    h = 1e-3
    def q2(s):
        f = lambda th: math.log(heat_class(beta, s, th))
        return (f(math.pi - h) - 2 * f(math.pi - 2 * h) + f(math.pi - 3 * h)) / h ** 2
    a, b = q2(0.6), q2(0.3)
    R.check("Hessian of the smoothed potential along e differs (antipode, t = 0.3)", abs(a - b) > 1e-2,
            f"global {a:.4f}, frozen {b:.4f}")
    return R.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
