Experimental
============

Also known as **not supported**.

Methods discussed and provided here have no guarantee to work or provide any meaningful results.
These are somehow abandoned projects and unfortunately mid-way through. They aren't completely
discarded simply because of hope that maybe someday someone will come and help fix them.
We all know that the best motivation to do something is to be annoyed by the current state.
Seriously though, mode decomposition in 2D and multi-dim is an interesting topic. Please?


BEMD
----

.. warning::

    Important This is an experimental module. Please use it with care as no
    guarantee can be given for obtaining reasonable results, or that they will be
    computed index the most computation optimal way.

Info
****

**BEMD** performed on bidimensional data such as images.
This procedure uses morphological operators to detect regional maxima
which are then used to span surface envelope with a radial basis function.

Class
*****

.. autoclass:: PyEMD.BEMD.BEMD
    :members:
    :special-members:
    
EMD2D
-----

.. warning::

    Important This is an experimental module. Please use it with care as no
    guarantee can be given for obtaining reasonable results, or that they will be
    computed index the most computation optimal way.

Info
****

**EMD** performed on images. This version uses for envelopes 2D splines,
which are span on extrema defined through maximum filter.

Class
*****

.. autoclass:: PyEMD.EMD2d.EMD2D
    :members:
    :special-members:
