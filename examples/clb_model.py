#! /usr/bin/env python
# coding: utf-8

import numpy as np
import doas


class APEXCalibModel(doas.Function):
    def __init__(self, i, wsf, spf, w_hr, E_hr, poly, xss, offset,
        residual=None, ring=None, isf='gauss'):
        """\
        This is the standard forward model for APEX VNIR calibration.

        wsf     wavelength shift function
        spf     slit parameter function
        w_hr    reference wavelengths
        E_hr    (solar) reference spectrum
        poly    polynomial or spline
        xss     cross sections
        offset  offset polynomial or spline
        isf     instrument slit function
        """
        self.name = 'APEXCalibModel'

        dtype = 'f8'
        self.w_hr = np.asarray(w_hr, dtype)
        self.E_hr = np.asarray(E_hr, dtype)

        self.isf = isf
        self.wsf = wsf
        self.spf = spf

        self.poly = poly
        self.xss = xss
        self.offset = offset
        self.ring = ring


    def compute_clb(self, x, b):
        """\
        Compute spectral calibration (CWs, FWHMs)
        """
        w = b['w0'] + self.wsf(x, b)
        sfp = self.spf(x, b)

        if np.any(sfp <= 0.0):
            raise RuntimeError('SFP is smaller than zero.')

        return w, sfp


    def __call__(self, x, b):
        """\
        Forward model for APEX VNIR calibration.

        w = clb + wsf(x,b)
        sfp = spf(x,b)

        y = poly(x,b) * ref * np.exp(-xss(x,b))
        y = doas.misc.convolve(w, y, cw, fwhm) + offset(x)

        return y
        """
        x = np.asarray(x)

        ring = 0.0 if self.ring is None else self.ring(x, b)
        tau = np.sum(self.xss(x, b), axis=0)
        y = self.poly(x, b) * self.E_hr * np.exp(-tau + ring)

        # project to instrument resolution
        cw, fwhm = self.compute_clb(x, b)
        y = doas.misc.convolve(self.w_hr, y, cw, fwhm, 'gauss')
        y += self.offset(x, b)

        return y


