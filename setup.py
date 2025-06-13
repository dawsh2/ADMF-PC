#!/usr/bin/env python
"""
Setup script for ADMF-PC
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='admf-pc',
    version='1.0.0',
    description='Adaptive Data Mining Framework - Protocol + Composition',
    author='ADMF-PC Team',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=requirements + [
        'click>=8.0',
        'tabulate>=0.9',
        'pyarrow>=10.0',
    ],
    entry_points={
        'console_scripts': [
            'admf=cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)