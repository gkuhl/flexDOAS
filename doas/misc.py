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

from __future__ import division

import warnings

import numpy as np
import scipy
import scipy.interpolate

import convx


def slant_column_fmt(x, x_std):
    """\
    Format slant column plus error to be used with matplotlib.
    """
    b = np.floor(np.log10(np.abs(x))).astype(int)
    return r'%.3f$\pm$%.3f$\times$10$^{%d}$' % (x / 10.0**b, x_std / 10.0**b, b)


def compute_rms(values):
    """\
    Compute root-mean-square of values.
    """
    values = values[np.isfinite(values)]
    return np.sqrt(np.mean(values**2))


def create_covariance_matrix(sigma, knots, length):
    """\
    Create covariance matrix for spline.
    """
    n = knots.size
    Sa = np.empty((n,n))

    if np.ndim(sigma) == 0:
        sigma = np.full(n, sigma)

    if np.ndim(length) == 0:
        length = np.full(n, length)

    for i,j in np.ndindex(n,n):
        if length is None:
            if i == j:
                Sa[i,j] = sigma[i] * sigma[j]
            else:
                Sa[i,j] = 0.0
        else:
            l = 0.5 * (length[i] + length[j])
            Sa[i,j] = sigma[i] * sigma[j] * np.exp(- abs(knots[i] - knots[j]) / l)

    return Sa



def interpolate(xp, x, y, out_type='float', k=1, ext=0):
    """\
    Wrapper around ``scipy.interpolate.UnivariateSpline`` useful
    for mapping bands to wavelengths and vice versa. If out_type
    is "index" returns integer.
    """
    valid = np.isfinite(y)
    f = scipy.interpolate.UnivariateSpline(x[valid], y[valid], k=k, s=0, ext=ext)
    yp = f(xp)

    if out_type == 'index':
        return np.round(yp).astype(int)
    else:
        return yp


def centres(x):
    """\
    Convert level to layer by 0.5 * (x[1:] + x[:-1]).
    """
    return 0.5 * (x[1:] + x[:-1])


def sampling_interval(x):
    x = np.asarray(x)
    c = centres(x)

    c0 = x[0] - (c[0] - x[0])
    cn = x[-1] + (x[-1] - c[-1])

    return np.diff( np.concatenate([[c0], c, [cn]]) )



def convolve(x, y, cw, fwhm, mode='gauss'):
    """\
    Convolve x,y with cw,fwhm for slit function (mode):
    - gauss (Gaussian)
    - erf (Error Function)
    """
    x = np.asarray(x, dtype='f8')
    y = np.asarray(y, dtype='f8')
    cw = np.asarray(cw, dtype='f8')
    fwhm = np.asarray(fwhm, dtype='f8')

    if np.std(np.diff(x)) / np.mean(np.diff(x)) > 1e-9:
        print np.std(np.diff(x)), np.mean(np.diff(x))
        raise ValueError('x need to be equidistant!')

    return convx.convolve(x, y, cw, fwhm, mode)



def convolve_gaussian(x, y, cw, fwhm, integrate=False, points=10):
    """\
    Convolve x,y with cw,fwhm.
    """
    x = np.asarray(x, dtype='f8')
    y = np.asarray(y, dtype='f8')
    cw = np.asarray(cw, dtype='f8')
    fwhm = np.asarray(fwhm, dtype='f8')

    if np.std(np.diff(x)) / np.mean(np.diff(x)) > 1e-9:
        print np.std(np.diff(x)), np.mean(np.diff(x))
        print 'x need to be equidistant!'

    if integrate:
        ssi = sampling_interval(cw)

        z = []
        for i in xrange(cw.size):
            wmin = cw[i] - 0.5 * ssi[i]
            wmax = cw[i] + 0.5 * ssi[i]

            w = np.linspace(wmin,wmax,points)
            h = interpolate(w, cw, fwhm, k=3)

            zi = np.mean(convx.convolve(x, y, w, h, 'gauss'))
            z.append(zi)

        return np.array(z)

    else:
        return convx.convolve(x, y, cw, fwhm, 'gauss')


def refractive_index_of_air(w=None, p=None, t=None, e=None):
    """\
    Computes the refractive index of air for given wavelength (w/nm) and,
    if given, includes corrections for atmospheric pressure (p/Pa),
    temperature (t/C) and partial water vapour pressure (e/hPa).

    References:
    Birch and Downs: Updated Edlen Equation for the Refractive Index of Air, 1993
    """
    w = 1e-3 * w # nm -> um

    n = 1 + 1e-8 * (8343.05 + 2406294.0 / (130 - w**-2) + 15999.0 / (38.9 - w**-2))

    if p is not None and t is not None:
        raise NotImplementedError

    if e is not None:
        raise NotImplementedError

    return n


def vac2air_wavelength(wvl):
    return wvl / refractive_index_of_air(wvl)

def air2vac_wavelength(wvl):
    return wvl * refractive_index_of_air(wvl)


def geometric_layer_amfs(z, sza, vza, z_ground, z_instrument):
    """\
    Compute geometric layer AMFs.

    z: vertical levels heights (TOA to ground) [m]

    sza: solar zenith angle [deg]
    vza: viewing zenith angle [deg]

    z_ground: ground elevation [m]
    z_instrument: aircraft or satellite altitude [m]
    """
    def last_index(z, z0):
        try:
            return np.where(z < z0)[0][0]
        except IndexError:
            pass

    mu0 = np.abs(np.full(z.size-1, 1.0 / np.cos(np.deg2rad(sza))))
    mu  = np.abs(np.full(z.size-1, 1.0 / np.cos(np.deg2rad(vza))))

    # set amf below ground zero
    i = last_index(z, z_ground)
    if i is not None:
        f = 1.0 - (z_ground - z[i]) / (z[i-1] - z[i])
        mu[i-1] *= f
        mu[i:] = 0.0

        mu0[i-1] *= f
        mu0[i:] = 0.0

    # set amf above aircraft to zero
    i = last_index(z, z_instrument)
    if i is not None:
        f = (z_instrument - z[i]) / (z[i-1] - z[i])
        mu[:i-1] = 0.0
        mu[i-1] *= f

    return mu + mu0


def fwhm2sigma(fwhm):
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def sigma2fwhm(sigma):
    return sigma * (2.0 * np.sqrt(2.0 * np.log(2.0)))




