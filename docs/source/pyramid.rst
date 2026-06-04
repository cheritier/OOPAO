Pyramid WFS
===========

.. currentmodule:: OOPAO.Pyramid

Overview
--------

The :class:`Pyramid` class implements a Pyramid Wavefront Sensor (PWFS). A four-faced glass pyramid prism placed at the telescope focal plane performs **amplitude filtering** of the electromagnetic field, splitting it into four pupil images on a detector. The WFS signals are derived from the intensity distribution across these four pupils.

The filtering is a Fourier-plane operation: the pyramid mask modulates the focal-plane amplitude, and the four pupil images are the result of this filtering. This is fundamentally different from a direct gradient measurement.

**Modulation** — the star is continuously tip-tilt modulated around the pyramid tip in a circle of radius ``modulation`` λ/D. This averages the Fourier filtering over a range of focal-plane positions, extending the linear regime of the sensor and making it effective even when phase aberrations are large.

Rooftop
~~~~~~~

The ``rooftop`` parameter models a **manufacturing defect at the apex of the pyramid prism** — instead of a perfect four-faced pyramid with four triangular faces meeting at a point, the prism has two triangular faces and two trapezoidal faces. The rooftop is oriented diagonally by default. The ``theta_rotation`` parameter rotates the entire mask (and hence the pupil image positions) around the optical axis.

Signal post-processing
~~~~~~~~~~~~~~~~~~~~~~

The raw detector frame is processed into a 1-D signal vector and a 2-D representation of the WFS signals. Several normalisation conventions are available via the ``postProcessing`` parameter.

GPU acceleration
~~~~~~~~~~~~~~~~

The :class:`Pyramid` is the only OOPAO class with existing GPU (CuPy) acceleration. When CuPy is installed, the FFT operations inside the modulation loop run on GPU automatically.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.Pyramid import Pyramid

   wfs = Pyramid(
       nSubap        = 20,
       telescope     = tel,
       modulation    = 3.0,    # lambda/D
       lightRatio    = 0.5,
   )

   # Propagate light (updates wfs.signal)
   ngs * tel * wfs

   # Access WFS signals
   print(wfs.signal)           # 1-D signal vector
   print(wfs.signal_2D)        # 2-D representation of WFS signals
   print(wfs.cam.frame)        # raw detector frame

   # Enable photon noise
   wfs.cam.photonNoise = True

   # Switch to full-frame output
   wfs.postProcessing = 'fullFrame'

API reference
-------------

.. class:: Pyramid(nSubap, telescope, modulation, lightRatio, postProcessing='slopesMaps', psfCentering=True, n_pix_separation=2.0, calibModulation=50.0, n_pix_edge=None, extraModulationFactor=0, binning=1, nTheta_user_defined=None, userValidSignal=None, old_mask=False, rooftop=0, theta_rotation=0, delta_theta=0.0, user_modulation_path=None, pupilSeparationRatio=None, edgePixel=None, zeroPadding=None)

   Pyramid Wavefront Sensor with focal-plane amplitude filtering and optional modulation.

   :param nSubap: Diameter of each pyramid pupil image in detector pixels.
   :type nSubap: float
   :param telescope: Associated telescope object (carries phase, flux, and pupil information).
   :type telescope: Telescope
   :param modulation: Tip-tilt modulation radius in λ/D units.
   :type modulation: float
   :param lightRatio: Minimum fractional illumination threshold to select valid subapertures.
   :type lightRatio: float
   :param postProcessing: Signal computation mode. Options:

      * ``'slopesMaps'`` — normalised WFS signals (default).
      * ``'fullFrame'`` — raw detector frame.
      * ``'slopesMaps_incidence_flux'`` — WFS signals normalised by incident flux.
      * ``'slopesMaps_sum_flux'`` — WFS signals normalised by total flux sum.
      * ``'fullFrame_incidence_flux'``, ``'fullFrame_sum_flux'`` — full-frame variants.

   :type postProcessing: str
   :param psfCentering: If ``True``, pyramid mask is centred on 4 pixels (default). If ``False``, centred on 1 pixel.
   :type psfCentering: bool
   :param n_pix_separation: Pixel gap between the four pyramid pupil images. Default ``2``.
   :type n_pix_separation: float
   :param calibModulation: Large modulation radius (λ/D) used at calibration to identify valid pixels. Default ``50``.
   :type calibModulation: float
   :param n_pix_edge: Edge padding in pixels around each pupil image. Defaults to ``n_pix_separation / 2``.
   :type n_pix_edge: float or None
   :param extraModulationFactor: Extra modulation points per quadrant (1 adds 4 points total). Default ``0``.
   :type extraModulationFactor: int
   :param binning: Detector binning factor. Default ``1``.
   :type binning: int
   :param nTheta_user_defined: Override the number of modulation steps. Default ``None`` (automatic from ``modulation``).
   :type nTheta_user_defined: int or None
   :param userValidSignal: User-defined boolean mask for valid pixels. Default ``None``.
   :type userValidSignal: numpy.ndarray or None
   :param old_mask: Use the legacy pyramid mask (local tip/tilt shifts). Default ``False``.
   :type old_mask: bool
   :param rooftop: Models a manufacturing defect at the apex of the pyramid prism, where two triangular faces are replaced by two trapezoidal faces. The value defines the width of the flat region in λ/D. The rooftop is oriented diagonally by default. Default ``0`` (perfect four-triangular-face pyramid).
   :type rooftop: float
   :param theta_rotation: Rotation angle of the pyramid mask in radians, which rotates the pupil image positions accordingly. Incompatible with ``old_mask=True``. Default ``0``.
   :type theta_rotation: float
   :param delta_theta: Angular offset for modulation points. Default ``0.0``.
   :type delta_theta: float
   :param user_modulation_path: Custom list of ``[x, y]`` modulation coordinates in λ/D. Default ``None``.
   :type user_modulation_path: list or None
   :param pupilSeparationRatio: **Deprecated.** Use ``n_pix_separation`` instead.
   :type pupilSeparationRatio: float or None
   :param edgePixel: **Deprecated.** Use ``n_pix_edge`` instead.
   :type edgePixel: int or None
   :param zeroPadding: Manual override for the zero-padding size. Default ``None`` (automatic).
   :type zeroPadding: int or None

   **Key properties**

   .. attribute:: signal
      :type: numpy.ndarray

      1-D WFS signal vector (valid pixels only) after the current propagation.

   .. attribute:: signal_2D
      :type: numpy.ndarray

      2-D representation of the WFS signals.

   .. attribute:: nSignal
      :type: int

      Length of the :attr:`signal` vector (number of valid measurement pixels).

   .. attribute:: validPixelsMask
      :type: numpy.ndarray

      Boolean 2-D mask identifying valid detector pixels.

   .. attribute:: cam
      :type: Detector

      Built-in detector object. Use to set noise properties (``cam.photonNoise``, ``cam.readoutNoise``, etc.).

   .. attribute:: focal_plane_camera
      :type: Detector

      Focal-plane camera at the pyramid tip. Used with :class:`~OOPAO.GainSensingCamera.GainSensingCamera`.

   .. attribute:: postProcessing
      :type: str

      Signal post-processing mode. Can be changed between propagations.

   .. attribute:: modulation
      :type: float

      Current modulation radius in λ/D. Changing this triggers recomputation of the modulation path.

   .. attribute:: mask
      :type: numpy.ndarray

      Complex 2-D pyramid mask applied in the focal plane.

   .. attribute:: nTheta
      :type: int

      Number of modulation steps used per frame.

   **Methods**

   .. method:: relay(src)

      Core propagation method. Called automatically by ``src * wfs``. Performs the focal-plane amplitude filtering for each modulation step and computes the WFS signals.

      :param src: Source or Asterism being propagated.
      :type src: Source or Asterism

   .. method:: set_binning(binning)

      Apply a new detector binning factor and recompute the valid pixel mask.

      :param binning: New binning factor.
      :type binning: int

   **Operator summary**

   +--------------------------------------+---------------------------------------------------+
   | Expression                           | Effect                                            |
   +======================================+===================================================+
   | ``ngs * tel * wfs``                  | Propagate through telescope then PWFS             |
   +--------------------------------------+---------------------------------------------------+
   | ``ngs ** atm * tel * wfs``           | Full reset + atmosphere + telescope + PWFS        |
   +--------------------------------------+---------------------------------------------------+
   | ``wfs * wfs.focal_plane_camera``     | Forward PSF to focal-plane camera                 |
   +--------------------------------------+---------------------------------------------------+

   .. note::

      GPU acceleration is activated automatically when CuPy is installed. The FFT operations inside the modulation loop run on GPU; all other computations remain on CPU. Full GPU support across all classes is not yet implemented.
