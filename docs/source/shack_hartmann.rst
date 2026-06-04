Shack-Hartmann WFS
==================

.. currentmodule:: OOPAO.ShackHartmann

Overview
--------

The :class:`ShackHartmann` class implements a Shack-Hartmann Wavefront Sensor (SH-WFS). A lenslet array placed at the pupil plane focuses light from each subaperture onto a detector. The WFS signals are derived from the intensity distribution of the spots across the detector.

Two computation modes are available:

* **Diffractive** (default) — each subaperture spot is computed via FFT, correctly modelling diffraction and LGS elongation effects.
* **Geometric** — direct computation from the phase; faster but ignores diffraction.

Key concepts
~~~~~~~~~~~~

* **nSubap** — number of lenslets across the pupil diameter. Defines the measurement grid.
* **lightRatio** — minimum illumination fraction to mark a subaperture as valid.
* **pixel_scale** — angular size of each detector pixel in arcseconds. If not set, defaults to Shannon sampling (2 pixels per FWHM) or 1 pixel per FWHM.
* **Detector** — the SH-WFS has a built-in :class:`~OOPAO.Detector.Detector` (``wfs.cam``) supporting all noise and binning models.
* **em_field_transform** — an optional :class:`~OOPAO.FieldTransformer.FieldTransformer` applied to the EM field before propagation through the lenslet array, used for example to apply sub-pixel shifts for super-resolution.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.ShackHartmann import ShackHartmann

   wfs = ShackHartmann(
       nSubap       = 20,
       telescope    = tel,
       lightRatio   = 0.5,
   )

   # Propagate light
   ngs * tel * wfs

   print(wfs.signal)          # 1-D WFS signal vector
   print(wfs.signal_2D)       # 2-D representation of WFS signals
   print(wfs.cam.frame)       # raw detector frame

   # Geometric mode (faster)
   wfs.is_geometric = True
   ngs * tel * wfs

API reference
-------------

.. class:: ShackHartmann(nSubap, telescope, lightRatio, threshold_cog=0.01, is_geometric=False, binning_factor=1, pixel_scale=None, threshold_convolution=0.05, shannon_sampling=False, n_pixel_per_subaperture=None, half_pixel_shift=False, em_field_transform=None)

   Shack-Hartmann Wavefront Sensor (diffractive or geometric).

   :param nSubap: Number of lenslets across the pupil diameter.
   :type nSubap: int
   :param telescope: Telescope object carrying phase, flux, and pupil information.
   :type telescope: Telescope
   :param lightRatio: Minimum fractional flux threshold for valid subaperture selection.
   :type lightRatio: float
   :param threshold_cog: Relative threshold (fraction of peak) applied before computing the centre of gravity. Default ``0.01``.
   :type threshold_cog: float
   :param is_geometric: If ``True``, use the geometric WFS (direct gradient). If ``False`` (default), use the diffractive model.
   :type is_geometric: bool
   :param binning_factor: Detector binning factor. Default ``1``.
   :type binning_factor: int
   :param pixel_scale: Desired pixel scale in arcseconds. The nearest realisable scale (given FFT sampling) is used. Default ``None`` (Shannon sampling).
   :type pixel_scale: float or None
   :param threshold_convolution: Relative threshold applied to Gaussian LGS spot model to suppress edge effects. Default ``0.05``.
   :type threshold_convolution: float
   :param shannon_sampling: If ``True`` and ``pixel_scale=None``, uses 2 pix/FWHM. If ``False`` (default), uses 1 pix/FWHM.
   :type shannon_sampling: bool
   :param n_pixel_per_subaperture: Number of pixels per subaperture. If smaller than the FFT-native value, subapertures are cropped; if larger, zero-padded (a warning is shown). Default ``None`` (native).
   :type n_pixel_per_subaperture: int or None
   :param half_pixel_shift: If ``True``, applies a half-pixel focal-plane shift to centre spots on 1 pixel. Default ``False`` (spots centred on 4 pixels).
   :type half_pixel_shift: bool
   :param em_field_transform: A :class:`~OOPAO.FieldTransformer.FieldTransformer` instance applied to the source EM field before propagation through the lenslet array. Used for example to introduce sub-pixel shifts for super-resolution. Default ``None``.
   :type em_field_transform: FieldTransformer or None

   **Key properties**

   .. attribute:: signal
      :type: numpy.ndarray

      1-D WFS signal vector for all valid subapertures.

   .. attribute:: signal_2D
      :type: numpy.ndarray

      2-D representation of the WFS signals.

   .. attribute:: valid_signal_2D
      :type: numpy.ndarray

      2-D layout of the valid WFS signal pixels on the detector.

   .. attribute:: nSignal
      :type: int

      Total length of the signal vector.

   .. attribute:: valid_subapertures
      :type: numpy.ndarray

      2-D boolean mask of valid subapertures over the pupil grid.

   .. attribute:: valid_subapertures_1D
      :type: numpy.ndarray

      Flattened 1-D boolean version of :attr:`valid_subapertures`.

   .. attribute:: nValidSubap
      :type: int

      Number of valid (illuminated) subapertures.

   .. attribute:: cam
      :type: Detector

      Built-in detector. Set ``cam.photonNoise``, ``cam.readoutNoise``, etc. to simulate detector effects.

   .. attribute:: is_geometric
      :type: bool

      Switch between diffractive (``False``) and geometric (``True``) computation modes at runtime.

   .. attribute:: pixel_scale
      :type: float

      Effective pixel scale in arcseconds (read-only after initialisation).

   .. attribute:: n_pix_subap
      :type: int

      Number of pixels per subaperture on the detector.

   **Methods**

   .. method:: relay(src)

      Core propagation method. Called automatically by ``src * wfs``. Computes the spot images for each subaperture and derives the WFS signals.

      :param src: Source or Asterism being propagated.
      :type src: Source or Asterism

   .. method:: set_weighted_centroiding_map(weight_map)

      Set a custom 2-D centroiding weight map for the spots. Recalibrates signal units afterwards.

      :param weight_map: 2-D weight array of shape ``(n_pix_subap, n_pix_subap)``.
      :type weight_map: numpy.ndarray

   .. method:: set_slopes_units()

      Recalibrate the signal unit conversion factor (called automatically after :meth:`set_weighted_centroiding_map`).

   **Operator summary**

   +------------------------------+---------------------------------------------------+
   | Expression                   | Effect                                            |
   +==============================+===================================================+
   | ``ngs * tel * wfs``          | Propagate through telescope then SH-WFS           |
   +------------------------------+---------------------------------------------------+
   | ``ngs ** atm * tel * wfs``   | Full reset + atmosphere + telescope + SH-WFS      |
   +------------------------------+---------------------------------------------------+
