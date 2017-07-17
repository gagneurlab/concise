#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import sys

# add back later
if sys.version_info[0] != 3:
    # sys.exit("Only Python 3 is supported")
    print("WARNING: Only Python 3 is supported")

with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn>=0.18",
    "matplotlib",
    "tensorflow",
    # "glmnet",
    "keras>=2.0.2",
    'deeplift>=0.4',
    'simdna==0.2',
    'hyperopt',
]

dependency_links = [
    "https://github.com/kundajelab/deeplift/tarball/v0.4.0-alpha#egg=deeplift-0.4",
    "https://github.com/kundajelab/simdna/tarball/0.2#egg=simdna-0.2",
]

test_requirements = [
    "pytest",
]

setup(
    name='concise',
    version='0.6.0',
    description="CONCISE (COnvolutional Neural for CIS-regulatory Elements) is a model for predicting PTR features like mRNA half-life from cis-regulatory elements using deep learning. ",
    long_description=readme,  # + '\n\n' + history,
    author="Žiga Avsec",
    author_email='avsec@in.tum.de',
    url='https://github.com/avsecz/concise',
    packages=find_packages(),
    package_data={'concise.resources': ['attract_metadata.txt', 'attract_pwm.txt'],
                  'concise.resources.RNAplfold': ["H_RNAplfold", "I_RNAplfold", "M_RNAplfold", "E_RNAplfold"]},
    include_package_data=True,
    # setup_requires=['numpy'],
    install_requires=requirements,
    dependency_links=dependency_links,
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
