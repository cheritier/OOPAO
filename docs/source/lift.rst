LiFT
====

.. currentmodule:: OOPAO.LiFT

Overview
--------

:class:`LiFT` (Linearized Focal-plane Technique) is a focal-plane wavefront sensor that estimates wavefront aberrations from PSF images using phase diversity. A known diversity OPD (typically defocus) is introduced, and the wavefront is reconstructed iteratively by minimising the difference between observed and modelled PSFs.

The interaction matrices can be computed analytically (default) or numerically. GPU acceleration via CuPy is supported when available.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.LiFT import LiFT

   lift = LiFT(
       tel            = tel,
       basis          = Z.modesFullRes,   # 3-D Zernike basis [n,n,J]
       det            = cam,
       diversity_OPD  = diversity_map,    # 2-D OPD in metres
       iterations     = 30,
       img_resolution = 64,
       numerical      = False,
   )

   # Reconstruct wavefront from a PSF frame
   lift.get_modal_coefficients(cam.frame)
   print(lift.modal_coefficients)

API reference
-------------

.. class:: LiFT(tel, basis, det, diversity_OPD, iterations, img_resolution, numerical, ang_pixel_arcsec=None)

   Linearized focal-plane wavefront sensor.

   :param tel: Telescope object coupled with a Source (wavelength information required).
   :type tel: Telescope
   :param basis: Modal basis as a 3-D array ``(n_pix, n_pix, n_modes)``. Last dimension = mode index.
   :type basis: numpy.ndarray
   :param det: Detector object providing sampling and noise weighting.
   :type det: Detector
   :param diversity_OPD: 2-D phase diversity OPD map in metres.
   :type diversity_OPD: numpy.ndarray
   :param iterations: Maximum number of iterations for the LiFT algorithm.
   :type iterations: int
   :param img_resolution: PSF image resolution in pixels. Smaller values improve robustness under noise.
   :type img_resolution: int
   :param numerical: If ``True``, compute interaction matrices numerically. If ``False`` (default), use the analytical model.
   :type numerical: bool
   :param ang_pixel_arcsec: Angular pixel size in arcseconds. Overrides the detector sampling when set. Default ``None``.
   :type ang_pixel_arcsec: float or None

   **Methods**

   .. method:: get_modal_coefficients(frame)

      Estimate modal wavefront coefficients from a PSF image.

      :param frame: PSF detector frame.
      :type frame: numpy.ndarray

   **Key attributes**

   .. attribute:: modal_coefficients
      :type: numpy.ndarray

      Estimated modal coefficients in metres (populated by :meth:`get_modal_coefficients`).

   .. attribute:: gpu
      :type: bool

      ``True`` if CuPy GPU acceleration is active.

   .. note::

      Tutorial: ``tutorials/how_to_LIFT.ipynb``
