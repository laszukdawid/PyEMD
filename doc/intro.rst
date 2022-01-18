Intro
=====

General
-------

**PyEMD** is a Python implementation of `Empirical Mode Decomposition (EMD) <https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform#Techniques>`_ and its variations.
One of the most popular expansion is `Ensemble Empirical Mode Decomposition (EEMD) <http://www.worldscientific.com/doi/abs/10.1142/S1793536909000047>`_, which utilises an ensemble of noise-assisted executions.

As the name suggests, methods in this package take data (signal) and decompose it into a set of component.
All these methods theoretically should decompose a signal into the same set of components but in practise
there are plenty of nuances and different ways to handle noise. Regardless of the method, obtained
components are often called *Intrinsic Mode Functions* (IMF) to highlight that they contain an intrinsic (self)
property which is a specific oscillation (mode). These are generic oscillations; their frequency and 
amplitude can change, however, no they are distinct within analyzed signal.

Installation
------------

Simplest (pip)
``````````````

Using `pip` to install is the quickest way to try and play. The package has had plenty of time to mature
and at this point there aren't that many changes, especially nothing breaking. In the end, the basic EMD
is the same as it was published in 1998.

The easiest way is to install `EMD-signal`_ from the PyPi, for example using

    $ pip install EMD-signal

Once the package is installed it should be accessible in your Python as `PyEMD`, e.g. ::

    >>> from PyEMD import EMD

Research (github)
`````````````````

Do you want to see the code by yourself? Update it? Make it better? Or worse (no judgement)?
Then you likely want to check out the package and install it manually. **Don't worry, installation is simple**.

PyEMD is an open source project hosted on the GitHub on the main author's account, i.e. https://github.com/laszukdawid/PyEMD.
This github page is where all changes are done first and where all `issues`_ should be reported.
The page should have clear instructions on how to download the code. Currently that's a (only) green
button and then following options.

In case you like using command line and want a copy-paste line ::

    $ git clone https://github.com/laszukdawid/PyEMD


Once the code is download, enter package's directory and execute ::

    $ python setup.py install

This will download all required dependencies and will install PyEMD in your environment.
Once it's done do a sanity check with quick import and version print: ::

    $ python -c "import PyEMD; print(PyEMD.__version__)"

It should print out some value concluding that you're good to go. In case of troubles, don't hesitate to submit
an issue ticket via the link provided a bit earlier.

.. _EMD-signal: https://pypi.org/project/EMD-signal/
.. _issues: https://github.com/laszukdawid/PyEMD/issues
