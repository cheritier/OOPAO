Gain Sensing Camera
===================

.. currentmodule:: OOPAO.GainSensingCamera

Overview
--------

The :class:`GainSensingCamera` (GSC) estimates the **optical gains** of a Pyramid WFS using focal-plane PSF images and a convolutional analytical model, following the method of Chambouleyron et al. (2021, A&A). Optical gains account for the non-linear response of the PWFS when operating in the presence of residual turbulence, and are essential for accurate reconstruction in modal control.

The GSC operates on the PSF recorded by the ``wfs.focal_plane_camera`` (a :class:`~OOPAO.Detector.Detector` attached to the Pyramid). It must first be calibrated around a reference operating point (typically flat wavefront), and can then estimate gains at any subsequent operating point.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.GainSensingCamera import GainSensingCamera

   # 1. Set focal-plane camera resolution
   wfs.focal_plane_camera.resolution = wfs.nRes

   # 2. Create GSC with the pyramid mask and a modal basis
   gsc = GainSensingCamera(wfs.mask, modal_basis, n_jobs=10)

   # 3. Calibrate (flat wavefront)
   tel.resetOPD()
   tel * wfs
   wfs * wfs.focal_plane_camera
   wfs.focal_plane_camera * gsc      # first call = calibration

   # 4. Measure optical gains on turbulence
   tel.OPD = atm.OPD.copy()
   tel * wfs
   wfs * wfs.focal_plane_camera
   wfs.focal_plane_camera * gsc      # subsequent calls = estimation

   print(gsc.og)   # optical gain per mode

API reference
-------------

.. class:: GainSensingCamera(mask, basis, n_jobs=10)

   Optical gain estimator for the Pyramid WFS.

   :param mask: Complex pyramid mask array (available as ``wfs.mask``).
   :type mask: numpy.ndarray
   :param basis: Modal basis as a 3-D array of shape ``(n_pix, n_pix, n_modes)``.
   :type basis: numpy.ndarray
   :param n_jobs: Number of parallel FFT jobs. Default ``10``.
   :type n_jobs: int

   **Methods**

   .. method:: calibration(frame)

      Perform GSC calibration from a reference focal-plane frame (flat wavefront). Must be called before :meth:`get_optical_gains`.

      :param frame: Reference PSF frame from ``wfs.focal_plane_camera``.
      :type frame: numpy.ndarray

   .. method:: get_optical_gains(frame)

      Estimate optical gains from a focal-plane PSF frame.

      :param frame: Current PSF frame.
      :type frame: numpy.ndarray
      :returns: Per-mode optical gain vector.
      :rtype: numpy.ndarray

   .. method:: reset_calibration()

      Clear the calibration state to allow re-calibration around a new working point.

   **Key attributes**

   .. attribute:: og
      :type: numpy.ndarray

      Per-mode optical gain vector (populated after the first post-calibration call).

   .. attribute:: n_modes
      :type: int

      Number of modes in the basis.

   .. attribute:: calibration_ready
      :type: bool

      ``True`` once calibration has been performed.

   .. rubric:: Reference

   V. Chambouleyron et al., *Pyramid wavefront sensor optical gains compensation using a convolutional model*, A&A, 2021.
