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

"""\
Functions for convolving spectra with various slit function. The module
has not been tested yet and might not work, in particular, for
non-equidistant grids.
"""

import numpy as np
from scipy.special import erf

import misc

def error_function(x, sigma, ssi, dx=None):
    """\
    The error function is the convolution
    of Gaussian and box function. The function
    assumes equidistant x.
    """
    if dx is None:
        dx = misc.sampling_interval(x)

    delta = 0.5 * ssi
    a = sigma * np.sqrt(2.0)

    x1 = (x + delta) / a
    x2 = (x - delta) / a
    f = dx / (4.0 * delta)

    g = dx * ( erf( (x + delta) / a ) - erf( (x - delta) / a ) ) / (4.0 * delta)

    return g


def box(x, xmin, xmax):
    """\
    Box functions on interval [xmin,xmax]
    """
    dx = x[1] - x[0]
    di = ((xmax - xmin) / dx + 1).astype(int)
    p = np.full(di, 1.0 / di)

    return p


def gauss(x, mu, sigma):
    """\
    Gauss function.
    """
    dx = misc.sampling_interval(x)
    tau = np.sqrt(2.0 * np.pi)
    g  = np.exp( -(x - mu)**2 / (2.0 * sigma**2) )
    g *= dx / (sigma * tau)

    return g


def box_gauss(x, mu, sigma, ssi):
    """\
    Convolution of box and gauss function.
    """
    p = box(x, -0.5*ssi, +0.5*ssi)
    g = gauss(x, mu, sigma)

    return np.convolve(p,g, mode='same')


def convolve(x, y, cw, fwhm, mode='gauss'):
    """\
    Convolve y(x) with cw and fwhm (not tested).
    """
    tau = np.sqrt(2.0 * np.pi)
    fwhm2sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    dx = misc.sampling_interval(x)
    ssi = misc.sampling_interval(cw)

    values = np.zeros(cw.size)

    for i in xrange(cw.size):
        j, k = 0, x.size

        if j <= k:
            if mode == 'gauss':
                g = gauss(x[j:k], cw[i], fwhm[i] * fwhm2sigma)

            elif mode == 'box_gauss':
                g = box_gauss(x[j:k], cw[i], fwhm[i] * fwhm2sigma, ssi[i])

            elif mode == 'error_function':
                g = error_function(x[j:k] - cw[i], fwhm[i] * fwhm2sigma, ssi[i])

            else:
                raise ValueError

            values[i] = np.dot(y[j:k], g)
        else:
            values[i] = 0.0

    return values



