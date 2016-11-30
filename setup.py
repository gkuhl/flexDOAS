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


import setuptools
from distutils.core import setup
from Cython.Build import cythonize

from distutils.core import setup
from distutils.extension import Extension

import numpy


extensions = [
    Extension("doas.cspline", ["doas/cspline.pyx"]),
    Extension("doas.convx", ["doas/convx.pyx"])
]
extensions = cythonize(extensions)

setup(
    name                = 'flexDOAS',
    version             = '0.1',
    description         = 'flexible Python library for DOAS retrievals',
    long_description    = """
    """,
    url                 = '',
    download_url        = '',
    author              = 'Gerrit Kuhlmann',
    author_email        = 'gerrit.kuhlmann@empa.ch',
    platforms           = ['any'],
    license             = 'GNU3',
    keywords            = ['python', 'DOAS'],
    classifiers         = ['Development Status :: Beta',
                           'Intended Audiance :: Science/Research',
                           'License :: GNU version 3',
                           'Operating System :: OS Independent'
                          ],
    packages            = ['doas'],
    package_data        = {'doas': ['*.pyx']},
    ext_modules         = extensions,
    include_dirs        = [numpy.get_include()]
)
