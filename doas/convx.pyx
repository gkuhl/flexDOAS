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
# cython: boundscheck=False, wraparound=False, nonecheck=False

from __future__ import division
import cython
import numpy as np
from scipy.special import erf
cimport numpy as np

from libc.math cimport sqrt, log, exp

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cpdef np.ndarray[DTYPE_t] sampling_interval(np.ndarray[DTYPE_t] x):
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] c

    c = np.full(x.size+1, np.nan, dtype=float)
    for i in xrange(0,x.size-1):
        c[i+1] = 0.5 * (x[i] + x[i+1])

    c[0] = x[0] - 0.5 * (x[1] - x[0])
    c[c.size-1] = x[x.size-1] + 0.5 * (x[x.size-1] - x[x.size-2])

    return np.diff(c)



cpdef np.ndarray[DTYPE_t] error_function(np.ndarray[DTYPE_t] x, double sigma, double ssi):
    cdef unsigned int i
    cdef double a, delta, dx, f
    cdef np.ndarray[DTYPE_t, ndim=1] x1, x2, g

    dx = x[1] - x[0]

    delta = 0.5 * ssi
    a = sigma * 1.4142135623730951  # sqrt(2)

    x1 = (x + delta) / a
    x2 = (x - delta) / a
    f = dx / (4.0 * delta)

    g = f * (erf(x1) - erf(x2))

    return g


cpdef np.ndarray[DTYPE_t] asym_gauss(np.ndarray[DTYPE_t] x, double mu, double sigma, double asym):
    cdef np.ndarray[np.float64_t] g
    cdef double a, sigma_plus, sigma_minus
    cdef double tau = 2.5066282746310002 # sqrt(2pi)
    cdef unsigned int i

    i = np.argmax(x >= mu)

    a = (x[1] - x[0]) / (tau * sigma)

    sigma_plus = 1.4142135623730951 * (sigma + asym)
    sigma_minus = 1.4142135623730951 * (sigma - asym)

    g = np.empty(x.size)
    g[:i] = a * np.exp( -( (x[:i] - mu) / sigma_minus)**2 )
    g[i:] = a * np.exp( -( (x[i:] - mu) / sigma_plus)**2 )

    return g


def gauss(np.ndarray x, double mu, double sigma):
    cdef np.ndarray[np.float64_t] g
    cdef double tau = 2.5066282746310002 # sqrt(2pi)

    g  = np.exp( -(x - mu)**2 / (2.0 * sigma**2) )
    g *= 1.0 / (sigma * tau)

    return g


cpdef np.ndarray[DTYPE_t] convolve(
        np.ndarray[DTYPE_t] x,
        np.ndarray[DTYPE_t] y,
        str mode,
        np.ndarray[DTYPE_t] cw,
        np.ndarray[DTYPE_t] fwhm,
        np.ndarray[DTYPE_t] asym
    ):
    """\
    Convolve y on grid x with isf (instrument slit function).

    x: equdistant wavelength coords
    y: values

    cw:   center wavelengths
    fwhm: full width at half maximum of Gaussian slit function
    asym: asym factor (ignore for gauss and erf)
    """
    cdef np.ndarray[np.int_t, ndim=1] imin, imax
    cdef unsigned int i,j,k,l, M=cw.size, N=x.size

    cdef np.ndarray[DTYPE_t, ndim=1] b, sigma, g, ssi, values=np.zeros(M)

    #cdef np.ndarray[DTYPE_t, ndim=1] a, dx
    cdef double a, dx

    cdef double tau = 2.5066282746310002 # sqrt(2pi)
    cdef double fwhm2sigma = 1.0 / (2.0 * sqrt(2.0 * log(2.0)))

    cdef double cw0, sigma0

    # this assumes equidistant grid
    dx = x[1] - x[0]
    a = (dx / tau) #/ sigma

    imin = ((cw - 5.0 * fwhm * fwhm2sigma - x[0]) / dx).astype(int)
    imax = ((cw + 5.0 * fwhm * fwhm2sigma - x[0]) / dx + 1).astype(int)


    if mode == 'erf':
        ssi = sampling_interval(cw)

    for i in xrange(M):
        j = 0 if imin[i] < 0 else imin[i]
        k = N if imax[i] > N else imax[i]

        if j <= k:
            if mode == 'gauss':
                g = np.exp(-(x[j:k] - cw[i])**2 / (2.0 * (fwhm[i] * fwhm2sigma)**2))
                g *= a / (fwhm[i] * fwhm2sigma)
            elif mode == 'erf':
                g = error_function(x[j:k] - cw[i], fwhm[i] * fwhm2sigma, ssi[i])
            elif mode == 'asym_gauss':
                g = asym_gauss(x[j:k], cw[i], fwhm[i] * fwhm2sigma, asym[i] * fwhm2sigma)
            else:
                raise ValueError('Unknown slit function "%s"' % mode)

            values[i] = np.dot(y[j:k], g)
        else:
            values[i] = 0.0

    return values






if __name__ == '__main__':
    pass
