Display Tools
=============

.. currentmodule:: OOPAO.tools.displayTools

Overview
--------

The ``displayTools`` module provides matplotlib-based helpers for visualising phase maps, pupil images, and WFS frames.

Selected API reference
----------------------

.. function:: displayMap(A, norma=False, axis=2, mask=0, returnOutput=False)

   Display a 2-D phase/OPD map or a 3-D image cube.

   :param A: Input array (2-D or 3-D).
   :type A: numpy.ndarray
   :param norma: If ``True``, normalise each frame independently. Default ``False``.
   :type norma: bool
   :param axis: Cube axis to iterate over. Default ``2``.
   :type axis: int
   :param mask: If ``1``, apply a pupil mask before display. Default ``0``.
   :type mask: int
   :param returnOutput: If ``True``, return the display array instead of plotting. Default ``False``.
   :type returnOutput: bool

.. function:: makeSquareAxes(ax)

   Force a matplotlib axes to have equal aspect ratio (square).

   :param ax: Matplotlib axes object.

.. function:: cl_plot(obj, param, ...)

   Interactive closed-loop display that mimics an AO GUI, showing WFS frame, DM shape, PSF, and Strehl history in real time during a closed-loop run.
