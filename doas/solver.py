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
import numpy as np

from scipy import dot
from scipy.linalg import inv, LinAlgError


def compute_forward_difference(j, x, f0, vecfunc, b, EPS):
    temp = x[j]

    h = EPS * abs(temp)
    if h == 0.0:
        h = EPS

    # Trick to reduce finite precision error (Press et al.).
    x[j] = temp + h 
    h = x[j] - temp

    f = vecfunc(x, b)
    x[j] = temp

    # Forward difference formula.
    return (f - f0) / h



def forward_jacobian(x, vecfunc, b):
    """\
    Computes forward-difference approximation to Jacobian. On input, x[1..n] is the point at
    which the Jacobian is to be evaluated, fvec[1..n] is the vector of function values at the
    point, and vecfunc(n,x,f) is a user-supplied routine that returns the vector of functions at
    x. On output, df[1..n][1..n] is the Jacobian array.
    """
    EPS = 1.0e-4 # approximated square root of the machine precision.

    if b is None:
        b = {}

    f0 = vecfunc(x, b)
    df = np.zeros((f0.size, x.size))
    x = x.copy()

    for j in range(x.size):
        df[:,j] = compute_forward_difference(j, x, f0, vecfunc, b, EPS)

    return df





def gain_matrix(K, S_eps, S_a=None):
    """\
    Calculate gain matrix.
    """
    # m measurements, n state vector elements
    m,n = K.shape

    if np.size(S_eps) == m: # vector
        K = np.asarray(K)
        S_eps = np.asarray(S_eps)

        w = 1.0 / S_eps
        W = w[:,np.newaxis] * K

        #       inv(K.T @ W + inv(S_a)) @ W
        if S_a is None:
            G = dot(inv(dot(K.T, W)), W.T)
        else:
            S_a = np.asarray(S_a)
            G = dot(inv(dot(K.T, W) + inv(S_a)), W.T)

    else:
        S_eps_I = S_eps.I
        G = (K.T * S_eps_I * K + S_a.I).I * K.T * S_eps_I

    return G


def linear_least_square(x0,y,f, b):
    """\
    Linear least square.
    """
    K = forward_jacobian(x0, f, b)
    return dot(dot(inv(dot(K.T, K)), K.T), y)



def calculate_chi2(x, y, f, Se, Sa, xa, b=None):
    try:
        Se = inv(Se)
        Sa = inv(Sa)
    except ValueError:
        return np.nan

    r = y - f(x,b)
    ra = x - xa

    return dot(r, dot(Se, r)) + dot(ra, dot(Sa, ra))




def gauss_newton(x0, y, f,  Se=np.nan, Sa=np.nan, xa=np.nan, fprime=None, b=None, max_iters=100):
    """\
    Find optimal state vector x using Gauss-Newton method.
    """
    no_error = np.any(np.isnan(Se))
    no_prior = np.any(np.isnan(xa)) or np.any(np.isnan(Sa))

    if fprime is None:
        fprime = forward_jacobian

    if no_error:
        Se_inv = 1.0
    else:
        if np.ndim(Se) == 1:
            Se_inv = 1.0 / Se
        elif np.ndim(Se) == 0:
            Se_inv = 1.0 / np.full(np.shape(y), Se)
        else:
            Se_inv = inv(Se)

    if no_prior:
        Sa_inv = 0.0
    else:
        Sa_inv = inv(Sa)

    xi = x0.copy()

    for i in range(max_iters):
        Fi = f(xi, b)

        if np.any(np.isnan(Fi)):
            return np.full(x0.shape, np.nan), {
                'n_iter':  i+1,
                'S':       np.nan,
                'K':       np.nan,
                'success': False,
                'msg':    'forward model returns nan'
            }

        K = fprime(xi, f, b)

        if np.ndim(Se) == 1:
            S = dot(K.T, dot(np.diag(Se_inv), K)) + Sa_inv
            a = dot(K.T, Se_inv * (y - Fi))
        else:
            S = dot(K.T, dot(Se_inv, K)) + Sa_inv
            a = dot(K.T, dot(Se_inv, (y - Fi)))

        if no_prior:
            a2 = 0.0
        else:
            a2 = dot(Sa_inv, xi - xa)

        try:
            S_inv = inv(S)
        except LinAlgError:
            return np.full(x0.shape, np.nan), {
                'n_iter': i+1,
                'S':      np.nan,
                'K':      K,
                'success': False,
                'msg':    'LinAlgError'
            }
        x = xi + dot(S_inv, a - a2) # TODO: SVD option

        if np.any(np.isnan(Se)):
            if dot(x - xi, x - xi) < x.size * 1e-4: # TODO: allow setting thresholds
                break
        else:
            if dot(x - xi, dot(S, x - xi)) < x.size:
                break

        xi = x.copy()

    # calculate error from chi2 = n
    if no_error:
        F = f(x, b)
        Se = np.sum((F - y)**2) / F.size
        S = dot(K.T, K) / Se

    return x, {
        'n_iter': i+1,
        'S':      inv(S),
        'K':      K,
        'success': True,
        'msg':    'everything seems to have worked fine'
    }




if __name__ == '__main__':
    pass
