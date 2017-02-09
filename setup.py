#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import sys

if sys.version_info[0] != 3:
    sys.exit("Only Python 3 is supported")

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "tensorflow",
    "glmnet",
]

test_requirements = [
    "pytest",
]

setup(
    name='concise',
    version='0.4.2',
    description="CONCISE (COnvolutional Neural for CIS-regulatory Elements) is a model for predicting PTR features like mRNA half-life from cis-regulatory elements using deep learning. ",
    long_description=readme + '\n\n' + history,
    author="Žiga Avsec",
    author_email='avsec@in.tum.de',
    url='https://github.com/avsecz/concise',
    packages=["concise"],
    package_dir={'concise':
                 'concise'},
    include_package_data=True,
    setup_requires=['numpy'],
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords=["computational biology", "bioinformatics", "genomics",
              "deep learning", "tensorflow", ],
    classifiers=[
        # classifiers
        # default
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
