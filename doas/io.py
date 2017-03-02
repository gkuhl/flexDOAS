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
import netCDF4


def load_arts(filename, levels=Ellipsis):
    """\
    Load mol_tau file from arts.
    """
    with netCDF4.Dataset(filename) as xs_file:
        wvl = xs_file.variables['wvl'][:]
        z = xs_file.variables['z'][:]
        tau = xs_file.variables['tau'][levels,:,0,0]

    return wvl, tau, z



if __name__ == '__main__':
    pass
