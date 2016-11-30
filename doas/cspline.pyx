#! /usr/bin/env python
# coding: utf-8

# flexDOAS is a Python library for the development of DOAS
# algorithm.
# Copyright (C) 2016  Gerrit Kuhlmann (gerrit.kuhlmann@empa.ch)
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License a
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


# cython: profile=True

from __future__ import division
import numpy as np
cimport numpy as np


def hermite_spline(x,p,t):
    """\
    Compute hermite spline on x giving points p and knots t.
    """
    delta = np.diff(p) #/ np.diff(t) # FIXME: somewhere diff(t) is already applied.

    m = np.zeros_like(p)
    m[0] = delta[0]
    m[1:-1] = 0.5 * (delta[1:] + delta[:-1])
    m[-1] = delta[-1]

    a = hermite2cardinal(p,m)

    return compute_1d_spline(x, a, t)




def hermite2cardinal(p,m):
    p0 = np.asarray(p[:-1])
    m0 = np.asarray(m[:-1])
    p1 = np.asarray(p[1:])
    m1 = np.asarray(m[1:])

    a = np.zeros((p0.size,4))

    a[:,0] = p0
    a[:,1] = m0
    a[:,2] = -3.0 * p0 - 2.0 * m0 + 3.0 * p1 - m1
    a[:,3] =  2.0 * p0 +       m0 - 2.0 * p1 + m1

    return a



cpdef np.ndarray[np.float64_t] compute_hermite_spline(
        np.ndarray[np.float64_t] x,
        np.ndarray[np.float64_t] p,
        np.ndarray[np.float64_t] t
    ):
    cdef np.ndarray[np.float64_t] y, m, delta
    cdef double s
    cdef int i,j,k,d

    # compute slope
    delta = np.diff(p) #/ np.diff(t) # FIXME: somewhere diff(t) is already applied.
    m = np.zeros_like(p)
    m[0] = delta[0]
    m[1:-1] = 0.5 * (delta[1:] + delta[:-1])
    m[-1] = delta[-1]

    # compute spline
    y = np.zeros_like(x)
    k = 0
    for i in xrange(x.size):
        if x[i] > t[k+1] and k < t.size-2:
            k += 1

        s = (x[i] - t[k]) / (t[k+1] - t[k])

        y[i] += p[k] * (1 + 2*s) * (1 - s)**2
        y[i] += m[k] * s * (1 - s)**2
        y[i] += p[k+1] * s**2 * (3 - 2*s)
        y[i] += m[k+1] * s**2 * (s - 1)

    return y



def compute_1d_spline(y, a, yp):
    """\
    Compute spline on `y` from cardinal coefficents `a` shape(n,degree) and
    knots `yp` shape(n+1).
    """
    yp = np.asarray(yp)

    k = np.searchsorted(yp, y)
    k[k <= 0] = 1
    k[k >= yp.size] = yp.size - 1

    y = (y - yp[k-1]) / (yp[k] - yp[k-1])

    return np.sum([a[k-1][:,i] * y**i for i in xrange(a.shape[1])], axis=0)




if __name__ == '__main__':
    pass
