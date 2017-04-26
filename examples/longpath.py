#! /usr/bin/env python
# coding: utf-8

from __future__ import division
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray

import doas

ROOT_PATH = '../data'


def add_starting_vector(forward_model, data):
    """\
    Add starting vector to lp-doas dataset
    """
    x0 = xarray.DataArray(np.zeros(forward_model.n_states), dims=('state_dim',), name='x0')
    p = -np.log(data.get('spe', 'mean') / data.get('ref'))
    x0[forward_model.xss.mapping] =  1.0
    x0[forward_model.poly.mapping] = forward_model.poly.fit(p)
    x0[forward_model.wsf.mapping] = 0.0
    data.update(xarray.Dataset({'x0': x0}))

    return data


class Dataset(doas.Dataset):
    main_dim_name = "main_dim"
    obs_dim_name = "obs_dim"

    @classmethod
    def from_data_files(cls, ref_filename, spe_filename):
        """\
        Read dataset of LP-DOAS measurements from ref_filename and
        spe_filename.
        """
        w0, I0 = np.loadtxt(ref_filename, unpack=True)
        I = np.loadtxt(spe_filename)

        n_spectra, n_obs = I.shape

        coords = {
            cls.main_dim_name: np.arange(n_spectra),
            cls.obs_dim_name: np.arange(n_obs),
        }
        attrs = dict()

        dataset = {
            "w0":  xarray.DataArray(w0, dims=(cls.obs_dim_name,)),
            "ref": xarray.DataArray(I0, dims=(cls.obs_dim_name,)),
            "spe": xarray.DataArray(I, dims=(cls.main_dim_name, cls.obs_dim_name,))
        }
        return cls(dataset, coords, attrs)

    def get_observation(self, p):
        """\
        Returns measurement vector and uncertainty co-variance matrix.
        """
        return 0.0, np.nan

    def get_parameter_vector(self, p):
        """\
        Returns parameter vector b for p-th measurement. The vector is used
        by the forward model F(x,b).
        """
        return {
            "p": p,
            "spe": self.get("spe", p),
            "ref": self.get("ref", p),
            "w0": self.get("w0", p)
        }

    def get_starting_vector(self, p):
        """\
        Return starting vector x0 for p-th measurement.
        """
        return self.get('x0')



class StandardDOAS:
    def __init__(self, n_bands, poly_kind, poly_degree, wsf_kind, wsf_degree,
            xss_parameter):
        """\
        Create forward model for standard DOAS equation.
        """
        # cross sections
        self.xss = doas.FunctionsList('xss', [
            doas.CrossSection.from_xs_file(start=i, **kwargs)
            for i, kwargs in enumerate(xss_parameter)
        ])
        # polynomial
        self.poly = doas.Spline(
            n_bands, kind=poly_kind, degree=poly_degree,
            start=self.xss.mapping[-1]+1
        )
        # wavelength shift function
        self.wsf = doas.Spline(
            n_bands, kind=wsf_kind, degree=wsf_degree,
            start=self.poly.mapping[-1]+1
        )
        self.n_states = self.wsf.mapping[-1] + 1


    def __call__(self, x, b):
        """\
        Calculate DOAS equation: y =  F(x,b)
        """
        w = self.xss.get_function_by_name('NO2').wavelength
        I = b["spe"]
        I0 = b["ref"]

        # wavelength shift on reference
        cw = b['w0'] + self.wsf(x, b)

        # absorber optical depth
        tau = np.sum(self.xss(x, b), axis=0)
        tau = doas.misc.interpolate(cw, w, tau, k=3)

        return -self.poly(x,b) + np.log(I0) - tau - np.log(I)



class LpDOAS(doas.Retrieval):
    """\
    Inherit from DOAS retrieval class adding custom visulaization of data.
    """

    def show_timeseries(self, name='NO2'):
        """\
        Plot timeseries of trace gas "name".
        """
        xs = self.forward_model.xss.get_function_by_name(name)
        x = self.results.get('x')
        x_std = self.results.get('x_std')

        cn = xs.scaling * x[:,xs.mapping]
        cn_err = xs.scaling * x_std[:,xs.mapping]

        fig, ax = plt.subplots(1,1)
        ax.errorbar(np.arange(cn.size), cn, cn_err, marker='o', ls='-')
        ax.set_ylabel('column density')


    def show(self, p):
        """\
        Show DOAS fit results for spectrum "p".
        """
        b = self.data.get_parameter_vector(p)
        cw = b['w0']

        f = self.forward_model
        xss = f.xss

        x = self.results.get('x', p)
        x_std = self.results.get('x_std', p)

        rms = self.results.get('rms', p)
        res = self.results.get('residual', p)

        fig, axes = plt.subplots(4,2, figsize=(8,12))

        # ref/spe
        axes[0,0].plot(cw, b['spe'], 'b-', label='spe')
        axes[0,0].plot(cw, b['ref'], 'r-', label='ref')
        axes[0,0].legend(loc=0)
        axes[0,0].set_ylim(ymin=0)

        # residual
        axes[0,1].plot(cw, res, 'b-')
        axes[0,1].set_title('RMS = %.3g' % rms)

        # poly (w/o tau)
        axes[1,0].set_title('poly (%s, %d)' % (f.poly.kind, f.poly.degree))
        axes[1,0].plot(cw, f.poly(x), 'b-')
        axes[1,0].plot(cw, f.poly(x) + res, 'r-')

        # cross sections
        for i,xs in zip([(1,1),(2,0),(2,1),(3,0)], f.xss.functions):
            values = doas.misc.interpolate(cw, xs.wavelength, xs(x,b))

            axes[i].plot(cw, values, 'b-')
            axes[i].plot(cw, values + res, 'r-')

            cn = xs.scaling * x[xs.mapping]
            cn_err = xs.scaling * x_std[xs.mapping]

            axes[i].set_title(r'%s: %s' % (xs.name, doas.misc.slant_column_fmt(cn, cn_err)), loc='right')

        # wavelength shift
        axes[3,1].set_title('shifts (%s, %d)' % (f.wsf.kind, f.wsf.degree))
        axes[3,1].plot(cw, f.wsf(x,b), 'b-')

        for ax in axes.flatten():
            ax.grid(True)

        plt.tight_layout()



def main():

    # create dataset
    ref_filename = os.path.join(ROOT_PATH, "reference.dat")
    spe_filename = os.path.join(ROOT_PATH, "spectra.dat")
    data = Dataset.from_data_files(ref_filename, spe_filename)

    # create forward model (window: 435 - 460 nm, fwhm: 0.67025 nm at 435.84, from Lok)
    cw = data.get('w0')
    w = np.arange(430,460,0.1)
    fwhm = np.full(w.shape, 0.67025)

    # Parameters for cross sections
    xss_parameters = [{
            'name': 'NO2',
            'filename': os.path.join(ROOT_PATH, 'aux_data', 'NO2_Vandaele_2002_294K_400-600nm-0.01nm-vac.npy'),
            'filetype': 'npy',
            'wavelength': w, 'fwhm': fwhm,
            'scaling': 'ptp', 'highpass': True
        },{
            'name': 'O4',
            'filename': os.path.join(ROOT_PATH, 'aux_data', 'O4_ThalmanVolkamer_2013_293K_400-600nm-0.01nm-vac.npy'),
            'filetype': 'npy',
            'wavelength': w, 'fwhm': fwhm,
            'scaling': 'ptp', 'highpass': False
        },{
            'name': 'O3',
            'filename': os.path.join(ROOT_PATH, 'aux_data', 'O3_Serdyuchenko_2014_293K_400-600nm-0.01nm-vac.npy'),
            'filetype': 'npy',
            'wavelength': w, 'fwhm': fwhm,
            'scaling': 'ptp', 'highpass': True
        },{
            'name': 'H2O',
            'filename': os.path.join(ROOT_PATH, 'aux_data', 'H2O_HITRAN2012_400-600nm-0.01nm-vac.npy'),
            'filetype': 'npy',
            'wavelength': w, 'fwhm': fwhm,
            'scaling': 'ptp', 'highpass': False
        }
    ]

    # Create a forward model
    forward_model = StandardDOAS(cw.size, poly_kind='cardinal', poly_degree=5,
            wsf_kind='cardinal', wsf_degree=2, xss_parameter=xss_parameters
    )

    # Add starting vector to dataset
    data = add_starting_vector(forward_model, data)

    # Create retrieval class and solve all
    r = LpDOAS(forward_model, data)
    r.solve_all()

    # save dataset and results
    r.data.to_netcdf('../data/results/test_data.nc')
    r.results.to_netcdf('../data/results/test_results.nc')

    # plot/show fit number 0
    r.show(0)
    plt.savefig('../data/results/test_fit0.png')

    # plot NO2 time series
    r.show_timeseries('NO2')
    plt.savefig('../data/results/test_NO2ts.png')

    return r




if __name__ == '__main__':
    main()
