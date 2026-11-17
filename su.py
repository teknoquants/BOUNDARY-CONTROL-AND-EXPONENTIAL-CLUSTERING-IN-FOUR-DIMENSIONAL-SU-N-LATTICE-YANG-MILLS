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
"""SU(N) helpers: orthonormal basis for the metric -Tr(XY), Haar sampling, exponentials, SU(2) quaternions."""
import numpy as np
from scipy.linalg import expm


def su_basis(N):
    """Orthonormal basis of su(N) for <X, Y> = -Tr(XY)."""
    B = []
    for i in range(N):
        for j in range(i + 1, N):
            E = np.zeros((N, N), complex); E[i, j] = 1; E[j, i] = -1; B.append(E / np.sqrt(2))
            E = np.zeros((N, N), complex); E[i, j] = 1j; E[j, i] = 1j; B.append(E / np.sqrt(2))
    for k in range(1, N):
        D = np.zeros((N, N), complex)
        D[:k, :k] = np.eye(k) * 1j; D[k, k] = -1j * k
        B.append(D / np.sqrt(k * (k + 1)))
    return B


def ip(X, Y):
    return float(np.real(-np.trace(X @ Y)))


def haar(N, rng):
    Z = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    Q = Q * (np.diag(R) / np.abs(np.diag(R)))
    return Q / np.linalg.det(Q) ** (1 / N)


def quat_to_su2(q):
    a, b, c, d = q
    return np.array([[a + 1j * d, c + 1j * b], [-c + 1j * b, a - 1j * d]])


def su2_to_quat(U):
    """Quaternion of a real multiple of an SU(2) matrix (any element of R * SU(2))."""
    return np.array([U[0, 0].real, U[0, 1].imag, U[0, 1].real, U[0, 0].imag])


def angle(U):
    return float(np.arccos(np.clip(np.real(np.trace(U)) / 2, -1, 1)))


__all__ = ["su_basis", "ip", "haar", "quat_to_su2", "su2_to_quat", "angle", "expm"]
