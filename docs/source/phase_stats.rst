Phase Statistics
================

.. currentmodule:: OOPAO.phaseStats

Overview
--------

The ``phaseStats`` module provides functions for computing atmospheric phase statistics — covariance matrices, structure functions, and power spectral densities — used by :class:`~OOPAO.Atmosphere.Atmosphere` to generate phase screens.

Selected API reference
----------------------

.. function:: variance(atm)

   Compute the total phase variance from an atmosphere object.

   :param atm: Atmosphere object.
   :type atm: Atmosphere
   :returns: Phase variance in rad² (at 500 nm).
   :rtype: float

.. function:: covariance(rho, atm)

   Compute the Von Kármán phase covariance at baseline ``rho``.

   :param rho: Separation array in metres.
   :type rho: numpy.ndarray
   :param atm: Atmosphere object.
   :type atm: Atmosphere
   :returns: Phase covariance values.
   :rtype: numpy.ndarray

.. function:: ft_phase_screen(r0, N, delta, L0, l0, seed=None)

   Generate a single Fourier-transform phase screen with Von Kármán statistics.

   :param r0: Fried parameter in metres.
   :type r0: float
   :param N: Screen size in pixels.
   :type N: int
   :param delta: Pixel scale in metres per pixel.
   :type delta: float
   :param L0: Outer scale in metres.
   :type L0: float
   :param l0: Inner scale in metres.
   :type l0: float
   :param seed: Random seed. Default ``None``.
   :type seed: int or None
   :returns: 2-D phase screen in metres.
   :rtype: numpy.ndarray

.. function:: makeCovarianceMatrix(dm, atm, tel)

   Build the DM–atmosphere covariance matrix used for KL basis computation.

   :param dm: Deformable mirror object.
   :type dm: DeformableMirror
   :param atm: Atmosphere object.
   :type atm: Atmosphere
   :param tel: Telescope object.
   :type tel: Telescope
   :returns: Covariance matrix.
   :rtype: numpy.ndarray
