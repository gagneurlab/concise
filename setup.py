#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import sys

# add back later
if sys.version_info[0] != 3:
    # sys.exit("Only Python 3 is supported")
    print("WARNING: Only Python 3 is supported")

# with open('README.md') as readme_file:
#     readme = readme_file.read()

requirements = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn>=0.18",
    "matplotlib",
    # "tensorflow", # - not per-se required
    # "glmnet",
    "keras>=2.0.4,<=2.2.4",
    'descartes',
    'shapely',
    'gtfparse>=1.0.7'
]

test_requirements = [
    "pytest",
    'hyperopt',
]

setup(
    name='concise',
    version='0.6.9',
    description="CONCISE (COnvolutional Neural for CIS-regulatory Elements)",
    # long_description=readme,
    author="Žiga Avsec",
    author_email='avsec@in.tum.de',
    url='https://github.com/gagneurlab/concise',
    packages=find_packages(),
    package_data={'concise.resources': ['attract_metadata.txt', 'attract_pwm.txt',
                                        'encode_motifs.txt.gz',
                                        'HOCOMOCOv10_pcms_HUMAN_mono.txt'],
                  'concise.resources.RNAplfold': ["H_RNAplfold", "I_RNAplfold", "M_RNAplfold", "E_RNAplfold"]},
    include_package_data=True,
    setup_requires=['numpy'],
    install_requires=requirements,
    # dependency_links=dependency_links,
    license="MIT license",
    zip_safe=False,
    keywords=["computational biology", "bioinformatics", "genomics",
              "deep learning", "tensorflow", ],
    extras_require={
        'tensorflow': ['tensorflow>=1.0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.0']},
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
