Bi-O Edge WFS
=============

.. currentmodule:: OOPAO.BioEdge

Overview
--------

:class:`BioEdge` implements the Bi-O Edge Wavefront Sensor, a variant of Fourier-filtering sensors that uses four separate edge masks (instead of a single four-sided pyramid) to create four independent pupil images. The interface is intentionally similar to :class:`~OOPAO.Pyramid.Pyramid`.

Like the Pyramid, it supports tip-tilt modulation, multiple post-processing modes, and a built-in :class:`~OOPAO.Detector.Detector`. A ``grey_width`` parameter controls the transition zone (grey-edge) width.

API reference
-------------

.. class:: BioEdge(nSubap, telescope, modulation, lightRatio, postProcessing='slopesMaps', n_pix_separation=2.0, n_pix_edge=None, calibModulation=50.0, extraModulationFactor=0, binning=1, nTheta_user_defined=None, userValidSignal=None, grey_width=0, grey_length=False, delta_theta=0.0, user_modulation_path=None, quadrants_numbering=[0, 1, 2, 3])

   Bi-O Edge Wavefront Sensor.

   :param nSubap: Pupil image diameter in pixels.
   :type nSubap: float
   :param telescope: Associated telescope object.
   :type telescope: Telescope
   :param modulation: Tip-tilt modulation radius in λ/D.
   :type modulation: float
   :param lightRatio: Minimum fractional illumination for valid subaperture selection.
   :type lightRatio: float
   :param postProcessing: Signal computation mode (same options as :class:`~OOPAO.Pyramid.Pyramid`). Default ``'slopesMaps'``.
   :type postProcessing: str
   :param n_pix_separation: Pixel gap between pupil images. Default ``2``.
   :type n_pix_separation: float
   :param n_pix_edge: Edge padding in pixels. Default ``None`` (``n_pix_separation / 2``).
   :type n_pix_edge: float or None
   :param calibModulation: Calibration modulation in λ/D. Default ``50``.
   :type calibModulation: float
   :param extraModulationFactor: Additional modulation steps per quadrant. Default ``0``.
   :type extraModulationFactor: int
   :param binning: Detector binning factor. Default ``1``.
   :type binning: int
   :param nTheta_user_defined: Override modulation step count. Default ``None``.
   :type nTheta_user_defined: int or None
   :param userValidSignal: User-defined valid pixel mask. Default ``None``.
   :type userValidSignal: numpy.ndarray or None
   :param grey_width: Width of the grey-edge transition zone in λ/D. Default ``0``.
   :type grey_width: float
   :param grey_length: If ``True``, enables a variable-length grey zone. Default ``False``.
   :type grey_length: bool
   :param delta_theta: Angular offset for modulation points. Default ``0.0``.
   :type delta_theta: float
   :param user_modulation_path: Custom modulation path as ``[[x, y], ...]`` in λ/D. Default ``None``.
   :type user_modulation_path: list or None
   :param quadrants_numbering: Ordering of the four detector quadrants. Default ``[0, 1, 2, 3]``.
   :type quadrants_numbering: list[int]

   **Key properties** (same as :class:`~OOPAO.Pyramid.Pyramid`)

   * :attr:`signal` — 1-D WFS signal vector.
   * :attr:`cam` — built-in :class:`~OOPAO.Detector.Detector`.
   * :attr:`nSignal` — length of signal vector.

   **Operator summary**

   +-------------------+--------------------------------------------+
   | Expression        | Effect                                     |
   +===================+============================================+
   | ``tel * wfs``     | Propagate OPD through the Bi-O Edge sensor |
   +-------------------+--------------------------------------------+
