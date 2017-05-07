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

from __future__ import division, print_function

import logging
import os
import sys
import time

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

import numpy as np
import scipy
import xarray as xr

#from . import convolution
from . import cspline
from . import io
from . import misc
from . import solver

try:
    # the qdoas package is not publicly available
    # and will be replaced in future
    import qdoas
    _QDOAS_AVAILABLE = True
except ImportError:
    _QDOAS_AVAILABLE = False


logging.basicConfig(level=logging.WARNING)


class Dataset(xr.Dataset):
    _main_dim = 'main_dim'

    @classmethod
    def from_netcdf(cls, *args, **kwargs):
        d = xr.open_dataset(*args, **kwargs)
        return cls(d.data_vars, d.coords, d.attrs, compat='broadcast_equals')

    def get(self, name, p=None, default=None, cast=np.array):
        """\
        Get dataset with "name" at pointer "p" and cast to "cast".
        """
        if p == 'mean':
            p = Ellipsis
            do_mean = True
        else:
            do_mean = False

        v = super(Dataset, self).get(name, default)

        if p is not None:
            slice_ = tuple(
                p if dim == self._main_dim else Ellipsis
                for dim in v.dims
            )
            v = v[slice_]

            if do_mean:
                v = np.mean(v, axis=0)

        return cast(v)


    def set(self, name, values, p=Ellipsis):
        """\
        Set variable "name" to "values" at "pointer" (default Ellipsis).
        """
        v = super(Dataset, self).get(name)

        if v is None:
            raise ValueError('name "%s" not in dataset' % name)

        slice_ = tuple(
            p if dim == self._main_dim else Ellipsis
            for dim in v.dims
        )
        v[slice_] = values



class RetrievalResults(Dataset):

    @classmethod
    def from_data(cls, n_mains, n_states, n_obs, filename=None):
        """\
        Storing RetrievalResults.
        """
        main_dim = cls._main_dim
        obs_dim = "obs_dim"
        state_dim = "state_dim"

        coords = {
            cls._main_dim: np.arange(n_mains),
            obs_dim: np.arange(n_obs),
            state_dim: np.arange(n_states)
        }
        attrs = {}

        data = {}
        for name, value, dtype, dims in [
            ('x',           np.nan, 'f8', (main_dim, state_dim)),
            ('x_std',       np.nan, 'f8', (main_dim, state_dim)),
            ('S',           np.nan, 'f8', (main_dim, state_dim, state_dim)),
            ('error_code',   -9999, 'i2', (main_dim,)),
            ('n_iter',       -9999, 'i2', (main_dim,)),
            ('rms',         np.nan, 'f8', (main_dim,)),
            ('residual',    np.nan, 'f8', (main_dim, obs_dim)),
            ('K',           np.nan, 'f8', (main_dim, obs_dim, state_dim)),
            ('A',           np.nan, 'f8', (main_dim, state_dim, state_dim)),
        ]:
            shape = tuple(coords[dim].size for dim in dims)
            values = np.full(shape, value, dtype=dtype)
            data[name] = xr.DataArray(values, dims=dims, encoding={'zlib': True})

        return cls(data, coords, attrs)




class Function(object):
    name = 'none'
    type_ = 'Function'

    def __repr__(self):
        return '<%s "%s" at %d>' % (self.type_, self.name, id(self))

    def to_netcdf(self, filename, mode='a', group=''):
        """\
        Write Function to netcdf file.
        """
        # write data
        d = self.to_dataset()

        if hasattr(self, 'functions'):
            function_names = np.array([f.name for f in self.functions if f is not None])
            function_types = np.array([f.type_ for f in self.functions if f is not None])

            f = xr.Dataset({
                'function_names': xr.DataArray(function_names, dims='function_dim'),
                'function_types': xr.DataArray(function_types, dims='function_dim')
            })
            d = d.merge(f)

        encoding = dict((k,{'zlib': True}) for k in d)
        d.to_netcdf(filename, mode, group=group, encoding=encoding)

        # writer sub functions
        for function in getattr(self, 'functions', []):
            if function is not None:
                name = '%s/%s' % (group, function.name)
                function.to_netcdf(filename, mode='a', group=name)


    @classmethod
    def from_netcdf(cls, filename, group=''):
        """\
        Create Function from group in netcdf file.
        """
        data = xr.open_dataset(filename, group)

        kwargs = {}
        kwargs.update(data.variables)
        kwargs.update(data.attrs)

        for name,type in zip(
                np.array(kwargs.get('function_names', [])),
                np.array(kwargs.get('function_types', []))
            ):
            function = _name2function.get(type, Function)

            group_name = r'/'.join([group, name])
            kwargs[name] = function.from_netcdf(filename, group_name)


        return cls(**kwargs)




class FunctionsList(Function):
    type_ = "FunctionsList"

    def __init__(self, name, functions=None, **kwargs):
        self.name = name

        if functions is None:
            self.functions = [kwargs[k] for k in np.array(kwargs.get('function_names', []))]
        else:
            self.functions = functions


    def __call__(self, x, b=None):
        if b is None:
            b = {}
        return [f(x, b) for f in self.functions]

    def to_dataset(self):
        return xr.Dataset(
            attrs = {
                'name':    self.name,
            }
        )

    def get_function_by_name(self, name):
        """\
        Returns first occurance of function in list by provided "name". Raises
        ValueError if "name" not in functions.
        """
        for f in self.functions:
            if f.name == name:
                return f
        else:
            raise ValueError('"%s" not functions' % name)

    @property
    def names(self):
        return [f.name for f in self.functions]

    @property
    def mapping(self):
        return np.concatenate([f.mapping for f in self.functions]).flatten()



class CrossSection(Function):
    type_ = "CrossSection"

    def __init__(self, name, wavelength, values, scaling=1.0, z=np.nan, dz=np.nan, start=0,
        do_caching=True, spline=None, fit_profile=False):
        """\

        fit_profile: if True fit cross section as profile
        """
        self.name = name

        self.values = np.array(values)
        self.wavelength = np.array(wavelength)
        self.z = np.array(z)
        self.dz = np.array(dz)
        self.scaling = scaling

        self.spline = spline
        self.fit_profile = bool(fit_profile)

        if self.fit_profile:
            self.mapping = np.arange(start, start + self.dz.size)
        else:
            if spline is None:
                self.mapping = np.array([start])
            else:
                self.mapping = np.arange(start, start+spline.n_coefficients)

        self.end = self.mapping[-1]

        # use this to cache calculation of tau
        self.do_caching = bool(do_caching)
        self.pointer = None
        self.cached_data = None


    def to_dataset(self):
        if np.ndim(self.values) == 1:
            data = {'values': xr.DataArray(
                self.values, coords={'wavelength': self.wavelength}, dims='wavelength'
            )}
        else:
            data = {
                'values': xr.DataArray(self.values,
                    coords={'z': self.z, 'wavelength': self.wavelength},
                    dims=('z', 'wavelength')
                ),
                'dz': xr.DataArray(self.dz,
                    coords={'z': self.z},
                    dims=('z')
                )
            }

        return xr.Dataset(
            data,
            attrs = {
                    'start':  self.mapping[0],
                    'name':    self.name,
                    'do_caching':  int(self.do_caching),
                    'fit_profile': int(self.fit_profile),
            }
        )


    @classmethod
    def from_xs_file(cls, filename, filetype, name, start=0, wavelength=None, fwhm=None,
        do_caching=True, scaling=1.0, highpass=None, spline=None, **kwargs):
        """\
        Read cross section / optical depth from filename and, if given, convolute
        with FWHM and interpolate to wavelength.
        """
        if filetype == 'arts':

            w, tau, levels = io.load_arts(filename, levels=Ellipsis)
            levels = 1e3 * levels # km -> m

            dz = np.abs(np.diff(levels))
            z  = 0.5 * (levels[1:] + levels[:-1])

            data = np.array([misc.interpolate(wavelength, w, tau[i], k=1)
                for i in range(tau.shape[0])
            ])

            if scaling is not None:
                data *= scaling

        elif filetype in ['ascii', 'npy']:

            z, dz = np.nan, np.nan

            if filetype == 'npy':
                w, tau = np.load(filename)
            else:
                w, tau = np.loadtxt(filename, unpack=True)

            if wavelength is None:
                wavelength = w.copy()

            if fwhm is None:
                data = misc.interpolate(wavelength, w, tau, k=1)
            else:
                try:
                    data = misc.convolve(w, tau, wavelength, fwhm, mode='gauss')
                except ValueError:
                    # TODO: fix convolution with non-equidistant
                    print('Warning: Interpolate Cross Section')
                    wnew = np.arange(
                            wavelength[0]-fwhm[0], wavelength[-1]+fwhm[-1],
                            np.min(np.diff(w))
                    )
                    tau = misc.interpolate(wnew, w, tau)
                    data = misc.convolve(wnew, tau, wavelength, fwhm, mode='gauss')


            if scaling is not None:
                if isinstance(scaling, str):
                    if scaling == 'ptp':
                        scaling = 1.0 / np.ptp(data)
                    else:
                        raise NotImplementedError

                data *= scaling

            if highpass:
                x = np.arange(data.size)
                data -= np.polyval(np.polyfit(x, data, 3), x)

        else:
            raise NotImplementedError

        return cls(name, wavelength, data, scaling, z, dz, start,
            do_caching=True, spline=spline)


    def __call__(self, x, b):
        """\
        Compute optical depth using tau/cross section, amf and
        scaling x.

        If self.do_caching is True, tau is used from cached data and
        thus amf will be only used if p is different from last call.
        """
        if self.spline is None:
            x = x[self.mapping]
        else:
            # TODO
            b['V0'] /= self.scaling
            x = self.spline(x,b)

            if x.size != self.values.size:
                x = np.interp(self.wavelength, b['w0'], x)

        p = b['p']
        amf = b.get('amf', None)

        # pre-calculated 2D field and store for caching
        if self.do_caching:
            if ((p != self.pointer) or self.cached_data is None):
                if np.ndim(self.values) == 2:

                    if amf is None:
                        z = np.concatenate([self.z + 0.5 * self.dz, [self.z[-1] - 0.5 * self.dz[-1]]])
                        amf = misc.geometric_layer_amfs(z, b['sza'], b['vza'], b['zg'], b['zi'])

                    # TODO: avoid multiplying and summing zeros
                    self.cached_data = np.sum(amf[:,np.newaxis] * self.values, axis=0)
                    self.pointer = p

            # calculate trace-gases optical depth (tau)
            # TODO: avoid multiplying and summing zeros
            if np.ndim(self.values) == 1:
                values = self.values * x

            elif np.ndim(self.values) == 2 and self.do_caching:
                values = self.cached_data * x
            else:
                raise ValueError

        else:
            if np.ndim(self.values) == 1:
                values = self.values * x

            elif np.ndim(self.values) == 2:
                if amf is None:
                    z = np.concatenate([self.z + 0.5 * self.dz, [self.z[-1] - 0.5 * self.dz[-1]]])
                    amf = misc.geometric_layer_amfs(z, b['sza'], b['vza'], b['zg'], b['zi'])

                # TODO: allow for x profiles
                data = np.sum(amf[:,np.newaxis] * self.values, axis=0)
                values = data * x
            else:
                raise ValueError


        return values


class Spline(Function):
    type_ = "Spline"

    def __init__(self, size, kind='cardinal', degree=-1, n_subwindows=None,
        start=0, knots=None, knot_distance=10, name=None):
        """\
        A wrapper class around polynomial/spline fitting.
        After initlization, this class can be called (poly(x)) to return
        poly values.

        """
        self.name = str(name)
        self.kind = kind

        self.size = size
        self.i = np.arange(self.size, dtype='f8')

        self.degree = degree

        if self.kind in ['B-spline', 'C-spline']:
            if knots is None:
                if n_subwindows is None:
                    n_subwindows = int(self.size / knot_distance + 0.5)

                self.knots = np.linspace(0, self.size, n_subwindows+1)
            else:
                self.knots = np.array(knots)

        elif self.kind == 'lagrange':
            self.knots = np.linspace(0, self.size, self.degree + 1)

        else: # polynomial: cardinal
            self.knots = knots

        self.n_coefficients = misc.calculate_n_coefficients(kind, degree, n_subwindows, self.knots)

        self.start = start
        self.end = start + self.n_coefficients
        self.mapping = np.arange(self.start, self.end)

        if self.kind == 'B-spline':
            self.knots = np.concatenate([
                self.knots[0].repeat(self.degree),
                self.knots,
                self.knots[-1].repeat(self.degree)
            ])


    def fit(self, values):
        """\
        Returns first estimate of state vector.
        """
        if self.kind == 'B-spline':
            knots = self.knots[self.degree+1:-self.degree-1]
            tck, fp, ier, msg = scipy.interpolate.splrep(
                self.i, values, t=knots, k=self.degree, full_output=1, s=0
            )
            return tck[1][:self.n_coefficients]

        elif self.kind == 'C-spline':
            if self.i.size > 10000:
                return np.interp(self.knots, self.i, values)

            x0 = np.zeros(self.knots.shape, dtype=float)
            x0[:] = values.mean()
            return solver.linear_least_square(x0, values, self, {'use_mapping': False})

        elif self.kind == 'lagrange':
            return np.interp(self.knots, self.i, values)

        elif self.kind == 'cardinal':
            return np.polyfit(self.i, values, self.degree)

        elif self.kind == 'scaling':
            return np.ones(self.n_coefficients)

        else:
            return np.array([])


    def __call__(self, x, b=None):
        """\
        Calculate spline/polynomial for state vector x.
        """
        if b is None:
            b = {}

        if b.get('use_mapping', True):
            x = x[self.mapping]

        if self.kind == 'B-spline':
            tck = (self.knots, x, self.degree)
            values = scipy.interpolate.splev(self.i, tck, ext=2)

        elif self.kind == 'C-spline':
            x = np.array(x)
            values = cspline.compute_hermite_spline(self.i, x, np.array(self.knots))

        elif self.kind == 'lagrange':
            # TODO: this is slow, rewrite using pre-calculated basis with knots and i
            values = scipy.interpolate.lagrange(self.knots, x)(self.i)

        elif self.kind == 'cardinal':
            values = np.polyval(x, self.i)

        elif self.kind == 'zeros':
            values = np.zeros(self.size)

        else:
            raise ValueError('Unknown kind "%s".' % self.kind)

        return values



class Ring(Function):
    type_ = 'Ring'

    def __init__(self, cw=None, fwhm=None, solar_filename=None, start=0,
        ring=None, spline=None):

        self.name = 'Ring'
        self.w = cw

        self.spline = spline
        if spline is None:
            self.mapping = np.array([start])
        else:
            self.mapping = np.arange(start, start+spline.n_coefficients)


        if ring is None:
            if _QDOAS_AVAILABLE:
                self.ring = qdoas.do_ring(
                    misc.vac2air_wavelength(cw), fwhm,
                    'gaussian', solar_filename, work_dir='/tmp'
                )[1]
                self.ring = np.log(self.ring)
            else:
                raise ValueError('Ring cross section `ring` not provided.')
        else:
            self.ring = ring

    @classmethod
    def from_file(cls, filename, start=0):
        cw, ring = np.loadtxt(filename, unpack=True)
        return cls(cw=cw, ring=ring, start=start)


    def __call__(self, x, b):
        if self.spline is None:
            x = x[self.mapping]
        else:
            x = self.spline(x,b)

        w0 = b.get('w0')
        ring = misc.interpolate(w0, self.w, x * self.ring)

        return ring


class Resolution(object):
    def __init__(self, w, E, cw, h, dh, isf='gauss', start=0, mode='log'):
        """\
        Resolution cross section for correcting differences between
        slit function parameters of reference and spectrum.
        """
        self.w = np.squeeze(w)
        self.E = np.squeeze(E)
        self.mode = mode

        n = self.E.shape[0] if np.ndim(self.E) == 2 else 1
        self.mapping = np.arange(start,start+n)

        if self.mapping.size == 1:
            self.resol = (
                misc.convolve(self.w, self.E, cw, h, isf) - misc.convolve(self.w, self.E, cw, h + dh, isf)
            ) / dh
            if self.mode == 'log':
                self.resol = self.resol / misc.convolve(self.w, self.E, cw, h, isf)
        else:
            self.resol = []
            for i in range(n):
                r = (misc.convolve(self.w, self.E[i], cw, h, isf) - misc.convolve(self.w, self.E[i], cw, h + dh, isf)) / dh

                if self.mode[i] == 'log':
                    r = r / misc.convolve(self.w, self.E[i], cw, h, isf)

                self.resol.append(r)


    def __call__(self, x, b):
        if self.mapping.size == 1:
            return x[self.mapping] * self.resol
        else:
            return np.dot(x[self.mapping], self.resol)




class Undersampling(object):
    def __init__(self, w, E, start=0, mode='log'):
        """\
        mode:
           - optical density mode (od)
        """
        self.w = np.squeeze(w)
        self.E = np.squeeze(E)

        self.mode = mode

        n = self.E.shape[0] if np.ndim(self.E) == 2 else 1
        self.mapping = np.arange(start,start+n)

    def __call__(self, x, b):

        # implicit slit correction
        cw0, fwhm0 = b.get('w0'), b.get('h0')
        cw, fwhm = b.get('w'), b.get('h')

        # spectrum cw is interpolated to refernce cw0
        if self.mapping.size == 1:

            over = misc.convolve(self.w, self.E, cw0, fwhm0)
            under = misc.interpolate(
                cw0, cw, misc.convolve(self.w, self.E, cw, fwhm),
                k=3
            )
            if self.mode == 'log':
                usamp = x[self.mapping[0]] * np.log(under / over)
            else:
                usamp = x[self.mapping[0]] * (over - under)

        else:
            usamp = np.zeros_like(cw0)

            for i in range(self.mapping.size):
                over = misc.convolve(self.w, self.E[i], cw0, fwhm0)
                under = misc.interpolate(
                    cw0, cw, misc.convolve(self.w, self.E[i], cw, fwhm),
                    k=3
                )
                if self.mode[i] == 'log':
                    usamp += x[self.mapping[i]] * np.log(under / over)
                else:
                    usamp += x[self.mapping[i]] * (over - under)


        return usamp




class Retrieval(object):

    def __init__(self, forward_model, data, evolution_model=None, mode='int',
        solver_method='ng', results=None, results_filename=None):

        self.forward_model = forward_model
        self.evolution_model = evolution_model

        self.mode = mode
        self.solver_method = solver_method

        self.data = data

        if results is None:
            try:
                self.results = RetrievalResults.from_data(
                    data.main_dim.size,
                    data.state_dim.size,
                    data.obs_dim.size,
                )
            except AttributeError:
                self.results = RetrievalResults.from_data(
                    data.main_dim.size,
                    self.forward_model.n_states,
                    data.obs_dim.size,
                )
        else:
            self.results = results

    @classmethod
    def from_netcdf(cls, f, f_filename, d, d_filename, r_filename=None):
        f = f.from_netcdf(f_filename)
        d = d.from_netcdf(d_filename)

        if r_filename is None:
            r = None
        else:
            r = RetrievalResults.from_netcdf(r_filename)

        return cls(f, d, results=r)


    def calculate_chi2(self, p):
        x = self.results.get('x', p)
        b = self.data.create_b(p)

        y = self.data.get('y', p)
        Se = np.diag(self.data.get('Se', p))
        xa = self.data.get('xa', p)
        Sa = self.data.get('Sa', p)

        return solver.calculate_chi2(x, y, self.forward_model, Se, Sa, xa, b)


    def solve_all(self, quiet=False):
        n_main = self.data.main_dim.size
        print('Solve for %d observation vectors using "%s" method.' % (n_main, self.solver_method))
        start = time.time()
        for p in range(n_main):
            self.solve(p, quiet=quiet)

        t = (time.time() - start)
        print('Finished after %.2f seconds (%.2f seconds per pixel).' % (t, t / n_main))


    def solve(self, p, solver_method=None, quiet=False):
        start = time.time()

        if solver_method is None:
            solver_method = self.solver_method

        if solver_method is None:
            raise ValueError('No solver method given.')

        logging.info('Solve pointer #%d.' % p)

        # get measurement vector
        logging.info('- get observation y and Se')
        y, Se = self.data.get_observation(p)

        if np.any(np.isnan(y)):
            loggin.info('- "nan"s in measurement vector (skip)')
            return

        if hasattr(self.data, "get_prior"):
            logging.info('- get xa and Sa')
            xa, Sa = self.data.get_prior(p)
        else:
            logging.info('- do not use a priori information')
            xa, Sa = np.nan, np.nan

        logging.info('- get starting vector x0')
        x0 = self.data.get_starting_vector(p)

        logging.info('- get parameter vector b')
        b = self.data.get_parameter_vector(p)

        # MAP or ML?
        if np.any(np.isnan(xa)) or np.any(np.isnan(Sa)):
            logging.info('- nan in xa or Sa (do not use a priori)')
            xa, Sa = np.nan, np.nan

        # get (if available) forward model has method for computing jacobian
        if hasattr(self.forward_model, 'compute_jacobian'):
            logging.info('- use "compute jacobian" method of forward model')
            fprime = getattr(self.forward_model, 'compute_jacobian', None)
        else:
            fprime = None

        # use gauss-newton algorithm to find solution
        logging.info('- use gauss-newton solver')
        x, info = solver.gauss_newton(
           x0, y, self.forward_model, Se, Sa, xa,
           b=b, fprime=fprime,
        )

        if info['success']:
            n_iter = info['n_iter']
            error_code = 1 if (n_iter == 10) else 0 # TODO

            logging.info('- found minimum after %.1f seconds' % (time.time() - start))

            # write results
            self.results.set('x', x, p)
            self.results.set('error_code', error_code, p)
            self.results.set('n_iter', n_iter, p)
            self.results.set('S', info['S'], p)

            # compute more 
            residual = self.forward_model(x, b) - y
            self.results.set('residual', residual, p)
            self.results.set('K', info['K'], p)

            self.results.set('rms', misc.compute_rms(residual), p)
            self.results.set('x_std', np.sqrt(info['S'].diagonal()), p)


        else:
            print(p, info['msg'])
            self.results.set('x', x, p)
            self.results.set('rms', np.nan, p)

        return x





if __name__ == '__main__':
    pass
