OPD Map
=======

.. currentmodule:: OOPAO.OPD_map

Overview
--------

:class:`OPD_map` is a lightweight wrapper around a static 2-D OPD array. It inserts an arbitrary wavefront aberration into the propagation chain using the ``*`` operator, without the overhead of the :class:`~OOPAO.NCPA.NCPA` modal basis computation.

API reference
-------------

.. class:: OPD_map(OPD)

   Static OPD element for the propagation chain.

   :param OPD: 2-D optical path difference array in metres, matching the telescope resolution.
   :type OPD: numpy.ndarray

   .. attribute:: OPD
      :type: numpy.ndarray

      The OPD array. Can be reassigned between propagations.

   .. attribute:: tag
      :type: str

      Always ``'OPD_map'``.

   **Operator summary**

   +------------------------------+---------------------------------------------------+
   | Expression                   | Effect                                            |
   +==============================+===================================================+
   | ``src * tel * opd_map * wfs``| Adds :attr:`OPD` to the cumulative OPD            |
   +------------------------------+---------------------------------------------------+

   **Example**

   .. code-block:: python

      from OOPAO.OPD_map import OPD_map
      import numpy as np

      static_aber = np.zeros([tel.resolution, tel.resolution])
      static_aber[100:120, 100:120] = 50e-9   # 50 nm flat zone

      opd = OPD_map(static_aber)
      src * tel * opd * wfs
