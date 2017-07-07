#!/usr/bin/env python

VERSION="0.2.2"

DESCRIPTION = "Implementation of Empirical Mode Decomposition (EMD) and its variations"

def main():
    import io
    import os
    from setuptools import setup

    with io.open('README.rst', encoding="utf8") as fp:
        long_description = fp.read().strip()

    with open('requirements.txt') as f:
        required = f.read().splitlines()

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]

    setup_params = dict(
        name="EMD-signal",
        version=VERSION,
        description=DESCRIPTION,
        long_description=long_description,
        url="https://github.com/laszukdawid/PyEMD",
        author="Dawid Laszuk",
        author_email="laszukdawid@gmail.com",
        classifiers=classifiers,
        keywords="signal decomposition data analysis",
        packages=["PyEMD"],
        install_requires=required,
        test_suite="PyEMD.tests"
    )

    dist = setup(**setup_params)

if __name__=="__main__":
    main()
