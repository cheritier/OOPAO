Asterism
========

.. currentmodule:: OOPAO.Asterism

Overview
--------

An :class:`Asterism` is a collection of :class:`~OOPAO.Source.Source` objects representing multiple guide stars (NGS and/or LGS). It can be coupled to a :class:`~OOPAO.Telescope.Telescope` in place of a single source, enabling multi-source and tomographic AO simulations.

When an asterism is coupled to a telescope and an atmosphere, each source in the asterism receives an independent OPD slice computed from the atmosphere layers, accounting for each source's altitude and sky coordinates.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.Source import Source
   from OOPAO.Asterism import Asterism

   src_on  = Source('H', magnitude=8, coordinates=[0, 0])
   src_gs1 = Source('R', magnitude=9, coordinates=[30, 0])
   src_gs2 = Source('R', magnitude=9, coordinates=[30, 120])
   src_gs3 = Source('R', magnitude=9, coordinates=[30, 240])

   ast = Asterism([src_gs1, src_gs2, src_gs3])

   # Propagate through the chain — relay() called on each source independently
   ast * tel * wfs

   # Reset all source EM fields and optical paths, then propagate
   ast ** atm * tel * wfs

API reference
-------------

.. class:: Asterism(list_src)

   Collection of :class:`~OOPAO.Source.Source` objects for multi-source simulations.

   :param list_src: List of :class:`~OOPAO.Source.Source` objects. May mix NGS and LGS types.
   :type list_src: list[Source]

   **Key properties**

   .. attribute:: src
      :type: list[Source]

      List of constituent source objects.

   .. attribute:: n_source
      :type: int

      Number of sources in the asterism.

   .. attribute:: coordinates
      :type: list

      Sky coordinates ``[radius, angle]`` of each source.

   .. attribute:: wavelength
      :type: float

      Wavelength of the first source (used for reference).

   .. attribute:: tag
      :type: str

      Always ``'asterism'``.

   **Operator summary**

   +-----------------------------+------------------------------------------------------------------+
   | Expression                  | Effect                                                           |
   +=============================+==================================================================+
   | ``ast * obj``               | Call ``obj.relay(src)`` for each source in the asterism          |
   +-----------------------------+------------------------------------------------------------------+
   | ``ast ** obj``              | Reset EM field and optical path for each source, then relay      |
   +-----------------------------+------------------------------------------------------------------+
   | ``ast * tel * wfs``         | Propagate all sources through telescope then WFS                 |
   +-----------------------------+------------------------------------------------------------------+
   | ``ast ** atm * tel * wfs``  | Full reset + atmosphere + telescope + WFS for all sources        |
   +-----------------------------+------------------------------------------------------------------+
