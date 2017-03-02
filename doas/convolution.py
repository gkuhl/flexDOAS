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

from __future__ import division, print_function
import numpy as np
from scipy.special import erf

from . import misc

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


def asym_gauss(x, mu, sigma, asym):
    """\
    Simple asymetric gauss.

    |asym| needs to be smaller than |sigma|
    """
    pos = x >= mu
    neg = ~pos

    a = (x[1] - x[0]) / (np.sqrt(2.0 * np.pi) * sigma)

    s = np.empty_like(x)
    s[pos] = np.exp( -np.abs( (x[pos] - mu) / (np.sqrt(2) * (sigma + asym)) )**2 )
    s[neg] = np.exp( -np.abs( (x[neg] - mu) / (np.sqrt(2) * (sigma - asym)) )**2 )

    return a * s


def convolve(x, y, cw, spf, mode='gauss'):
    """\
    Convolve y(x) with cw and fwhm (not tested).

    """
    tau = np.sqrt(2.0 * np.pi)
    fwhm2sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    dx = misc.sampling_interval(x)
    ssi = misc.sampling_interval(cw)

    values = np.zeros(cw.size)

    for i in range(cw.size):
        j, k = 0, x.size

        if j <= k:
            if mode == 'gauss':
                g = gauss(x[j:k], cw[i], spf[i] * fwhm2sigma)

            elif mode == 'box_gauss':
                g = box_gauss(x[j:k], cw[i], spf[i] * fwhm2sigma, ssi[i])

            elif mode == 'error_function':
                g = error_function(x[j:k] - cw[i], spf[i] * fwhm2sigma, ssi[i])

            elif mode == 'asym_gauss':
                g = asym_gauss(x[j:k], cw[i], spf[0,i] * fwhm2sigma, spf[1,i] * fwhm2sigma)

            else:
                raise ValueError

            values[i] = np.dot(y[j:k], g)
        else:
            values[i] = 0.0

    return values



