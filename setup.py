# -*- coding: utf-8 -*-
#
#  Copyright 2020-2021 Alexander Sizov <murkyrussian@gmail.com>
#  Copyright 2021 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of ThetaSynthesis.
#
#  ThetaSynthesis is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from pathlib import Path
from setuptools import setup


setup(
    name='ThetaSynthesis',
    version='0.1.0',
    packages=['ThetaSynthesis', 'ThetaSynthesis.abc', 'ThetaSynthesis.synthon', 'ThetaSynthesis.synthon.abc',
              'ThetaSynthesis.synthon.rollout'],
    python_requires='>=3.8.1',
    install_requires=['CGRtools>=4.1,<4.2', 'tqdm', 'StructureFingerprint'],
    package_data={'ThetaSynthesis.synthon.rollout': ['data/*']},
    zip_safe=True,
    license='LGPLv3',
    url='https://github.com/dcloudf/ThetaSynthesis',
    author='Alexander Sizov',
    author_email='',
    long_description=(Path(__file__).parent / 'README.rst').read_text(),
    classifiers=['Environment :: Plugins',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3 :: Only',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Scientific/Engineering :: Information Analysis',
                 'Topic :: Software Development',
                 'Topic :: Software Development :: Libraries',
                 'Topic :: Software Development :: Libraries :: Python Modules']
)
