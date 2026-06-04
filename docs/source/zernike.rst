Zernike
=======

.. currentmodule:: OOPAO.Zernike

Overview
--------

The :class:`Zernike` class computes Zernike polynomial modes over a telescope pupil (including annular pupils with central obstruction). It is used primarily to build modal bases for wavefront representation and NCPA generation.

Modes are normalised to unit RMS over the pupil (piston-removed by default).

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.Zernike import Zernike

   # Compute first 20 Zernike modes
   Z = Zernike(tel, J=20)
   Z.computeZernike(tel)

   print(Z.modes.shape)        # (n_pupil_pixels, 20)
   print(Z.modesFullRes.shape) # (resolution, resolution, 20)

   # Get mode name
   print(Z.modeName(2))  # 'Defocus'

API reference
-------------

.. class:: Zernike(telObject, J=1)

   Zernike polynomial basis generator.

   :param telObject: Telescope object providing pupil geometry and diameter.
   :type telObject: Telescope
   :param J: Number of Zernike modes to compute (starting from piston or tip, depending on ``remove_piston``). Default ``1``.
   :type J: int

   **Methods**

   .. method:: computeZernike(telObject, remove_piston=1)

      Compute the Zernike modes and store results in :attr:`modes` and :attr:`modesFullRes`.

      :param telObject: Telescope object.
      :type telObject: Telescope
      :param remove_piston: Number of leading modes to skip (default ``1`` skips piston).
      :type remove_piston: int

   .. method:: modeName(index)

      Return the human-readable name of Zernike mode at the given index (0-based).

      :param index: Mode index.
      :type index: int
      :returns: Name string, e.g. ``'Tip'``, ``'Defocus'``, ``'Coma vertical'``.
      :rtype: str

   .. method:: zernikeRadialFunc(n, m, r)

      Compute the Zernike radial polynomial :math:`R_n^m(r)`.

      :param n: Radial order.
      :type n: int
      :param m: Azimuthal order.
      :type m: int
      :param r: 2-D radial coordinate array (normalised 0–1).
      :type r: numpy.ndarray
      :returns: Radial polynomial values.
      :rtype: numpy.ndarray

   **Key attributes**

   .. attribute:: modes
      :type: numpy.ndarray

      Mode matrix of shape ``(n_pupil_pixels, J)``. Each column is a Zernike mode over the valid pupil pixels.

   .. attribute:: modesFullRes
      :type: numpy.ndarray

      Full-resolution mode cube of shape ``(resolution, resolution, J)``.

   .. attribute:: nModes
      :type: int

      Number of modes (equals ``J``).
