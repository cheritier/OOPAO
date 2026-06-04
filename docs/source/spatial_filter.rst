Spatial Filter
==============

.. currentmodule:: OOPAO.SpatialFilter

Overview
--------

A :class:`SpatialFilter` applies a focal-plane amplitude mask to the electromagnetic field, filtering out high spatial frequencies before the field reaches a downstream WFS. Shapes available are circular, square, and Foucault knife-edge.

API reference
-------------

.. class:: SpatialFilter(telescope, shape, diameter, zeroPaddingFactor=2)

   Focal-plane spatial filter.

   :param telescope: Associated telescope.
   :type telescope: Telescope
   :param shape: Mask shape. One of ``'circular'``, ``'square'``, ``'foucault'``.
   :type shape: str
   :param diameter: Filter diameter (or half-width for ``'square'``) in pixels of the zero-padded focal plane.
   :type diameter: float
   :param zeroPaddingFactor: Zero-padding factor for the focal-plane FFT. Default ``2``.
   :type zeroPaddingFactor: int

   .. attribute:: mask
      :type: numpy.ndarray

      Complex 2-D focal-plane mask.

   **Operator summary**

   +-------------------------------+----------------------------------------------+
   | Expression                    | Effect                                       |
   +===============================+==============================================+
   | ``src * tel * sf * wfs``      | Filter high frequencies before WFS sensing   |
   +-------------------------------+----------------------------------------------+
