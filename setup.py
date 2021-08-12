#!/usr/bin/env python

VERSION="1.0.0"

DESCRIPTION = "Implementation of the Empirical Mode Decomposition (EMD) and its variations"

def main():
    import io
    from setuptools import setup

    with io.open('README.md', encoding="utf8") as fp:
        long_description = fp.read().strip()

    with open('requirements.txt') as f:
        required = f.read().splitlines()

    classifiers=[
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]

    setup_params = dict(
        name="EMD-signal",
        version=VERSION,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/laszukdawid/PyEMD",
        author="Dawid Laszuk",
        author_email="pyemd@dawid.lasz.uk",
        license="Apache-2.0",
        classifiers=classifiers,
        keywords="signal decomposition data analysis",
        packages=["PyEMD"],
        install_requires=required,
        python_requires='>=3.6, <4',
        test_suite="PyEMD.tests",
        extras_require={
            "doc": ["sphinx", "sphinx_rtd_theme"],
        },
    )

    dist = setup(**setup_params)

if __name__=="__main__":
    main()
