#!/usr/bin/env python

VERSION="0.1"

DESCRIPTION = "Implementation of Empirical Mode Decomposition (EMD) and its variaties"

def main():
    import io
    import os
    from setuptools import setup

    with io.open('README.rst', encoding="utf8") as fp:
        long_description = fp.read().strip()

    with open('requirements.txt') as f:
        required = f.read().splitlines()

    setup_params = dict(
        name="EMD-signal",
        version=VERSION,
        description=DESCRIPTION,
        long_description=long_description,
        packages=["PyEMD"],
        author="Dawid Laszuk",
        author_email="laszukdawid@gmail.com",
        url="https://github.com/laszukdawid/PyEMD",
        install_requires=required,
        test_suite="PyEMD.tests"
    )

    dist = setup(**setup_params)

if __name__=="__main__":
    main()
