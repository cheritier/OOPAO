Detector
========

.. currentmodule:: OOPAO.Detector

Overview
--------

The :class:`Detector` simulates the effects of a real camera — including photon noise, readout noise, dark current, background, quantisation, pixel saturation, and integration time. It is used both as a standalone PSF camera (propagated from ``tel``) and as the built-in camera inside the WFS classes (``wfs.cam``).

Key concepts
~~~~~~~~~~~~

* **Integration time** — if ``integrationTime`` equals the AO sampling time, each frame is independent. If longer, frames are co-added into a buffer and read out once integration is complete.
* **Noise model** — photon noise is Poisson-distributed; readout noise is Gaussian. EMCCD and CMOS excess noise are modelled when ``sensor`` and ``gain`` are set appropriately.
* **Quantisation** — if ``bits`` is set, pixel values are rounded to the nearest digital number.
* **Full Well Capacity** — if ``FWC`` is set, pixels saturate at that electron count.
* **PSF sampling** — when used as a PSF camera (``tel * cam``), the ``psf_sampling`` parameter controls the FFT zero-padding factor.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.Detector import Detector

   # Simple noise-free detector
   cam = Detector()

   # Detector with noise
   cam = Detector(
       readoutNoise = 1.5,     # e-/pixel/frame
       photonNoise  = True,
       darkCurrent  = 0.1,     # e-/pixel/s
       QE           = 0.9,
   )

   # Use as a PSF camera
   src * tel * cam
   import matplotlib.pyplot as plt
   plt.imshow(cam.frame)

   # EMCCD with gain
   emccd = Detector(sensor='EMCCD', gain=100, readoutNoise=50)

API reference
-------------

.. class:: Detector(nRes=None, integrationTime=None, bits=None, output_precision=None, FWC=None, gain=1, sensor='CCD', QE=1.0, binning=1, psf_sampling=2.0, darkCurrent=0.0, readoutNoise=0, photonNoise=False, backgroundNoise=False, backgroundFlux=None, backgroundMap=None, log_scale=False)

   Camera model with configurable noise, quantisation, and integration.

   :param nRes: Detector resolution in pixels. Ignored for PSF computation (controlled by ``psf_sampling`` instead). Default ``None``.
   :type nRes: int or None
   :param integrationTime: Frame integration time in seconds.

      * ``None`` — matches ``tel.samplingTime`` (one AO frame per readout, default).
      * ``>= samplingTime`` — frames are buffered and summed until the integration is complete.
      * ``< samplingTime`` — raises an error.
   :type integrationTime: float or None
   :param bits: Pixel quantisation in bits. ``None`` disables quantisation. Default ``None``.
   :type bits: int or None
   :param output_precision: Output array dtype precision override. Default ``None``.
   :type output_precision: int or None
   :param FWC: Full Well Capacity in electrons. Pixels exceeding this value saturate. ``None`` disables saturation. Default ``None``.
   :type FWC: int or None
   :param gain: Detector gain. For ``sensor='EMCCD'``, sets the EM multiplication gain. Default ``1``.
   :type gain: int
   :param sensor: Sensor type for noise modelling. One of ``'CCD'`` (default), ``'CMOS'``, or ``'EMCCD'``.
   :type sensor: str
   :param QE: Quantum efficiency (0–1). Applied as a multiplicative factor on photon counts. Default ``1.0``.
   :type QE: float
   :param binning: Pixel binning factor. Default ``1``.
   :type binning: int
   :param psf_sampling: Zero-padding factor for FFT when computing PSFs from a Telescope. ``2`` = Shannon sampling (default).
   :type psf_sampling: float
   :param darkCurrent: Dark current in electrons per pixel per second. Default ``0.0``.
   :type darkCurrent: float
   :param readoutNoise: Readout noise standard deviation in electrons per pixel. Default ``0``.
   :type readoutNoise: float
   :param photonNoise: If ``True``, apply Poisson photon noise to each frame. Default ``False``.
   :type photonNoise: bool
   :param backgroundNoise: If ``True``, apply background noise using :attr:`backgroundFlux`. Default ``False``.
   :type backgroundNoise: bool
   :param backgroundFlux: 2-D background photon map to add to each frame when ``backgroundNoise=True``. Default ``None``.
   :type backgroundFlux: numpy.ndarray or None
   :param backgroundMap: 2-D background map subtracted from each frame. Default ``None``.
   :type backgroundMap: numpy.ndarray or None
   :param log_scale: If ``True``, the output ``frame`` is displayed in log10 scale. Default ``False``.
   :type log_scale: bool

   **Key properties**

   .. attribute:: frame
      :type: numpy.ndarray

      Current detector readout frame (in electrons or digital numbers, depending on ``bits``).

   .. attribute:: buffer_frames
      :type: list

      Accumulated frames during long integrations (cleared on readout).

   .. attribute:: photonNoise
      :type: bool

      Toggle Poisson photon noise on or off between frames.

   .. attribute:: readoutNoise
      :type: float

      Readout noise in electrons per pixel. Can be updated at runtime.

   **Operator summary**

   +-------------------+---------------------------------------------------+
   | Expression        | Effect                                            |
   +===================+===================================================+
   | ``tel * cam``     | Compute PSF and store in ``cam.frame``            |
   +-------------------+---------------------------------------------------+
   | ``wfs * cam``     | Forward WFS focal-plane image to camera           |
   +-------------------+---------------------------------------------------+
