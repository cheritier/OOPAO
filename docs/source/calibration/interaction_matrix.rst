Interaction Matrix
==================

.. currentmodule:: OOPAO.calibration.InteractionMatrix

Overview
--------

:func:`InteractionMatrix` builds the AO system interaction matrix (or its pseudo-inverse) by applying push-pull excitations to each DM mode and recording the WFS response. Supports parallel batch measurement and optional noise.

API reference
-------------

.. function:: InteractionMatrix(ngs, tel, dm, wfs, M2C, stroke, atm=None, phaseOffset=0, nMeasurements=50, noise='off', invert=True, nTrunc=0, print_time=False, display=False, single_pass=True)

   Build the interaction matrix of an AO system.

   :param ngs: Guide star source object.
   :type ngs: Source
   :param tel: Telescope object.
   :type tel: Telescope
   :param dm: Deformable mirror object.
   :type dm: DeformableMirror
   :param wfs: Wavefront sensor object.
   :type wfs: Pyramid or ShackHartmann
   :param M2C: Mode-to-command matrix of shape ``(n_actuators, n_modes)``.
   :type M2C: numpy.ndarray
   :param stroke: Amplitude of the push (and pull if ``single_pass=False``) in M2C units (metres for Gaussian IFs).
   :type stroke: float
   :param atm: Atmosphere object. If provided, turbulence is frozen during calibration. Default ``None``.
   :type atm: Atmosphere or None
   :param phaseOffset: Static OPD offset (e.g. NCPA) applied during calibration. Default ``0``.
   :type phaseOffset: numpy.ndarray or float
   :param nMeasurements: Number of DM modes excited per batch to speed up calibration. Default ``50``.
   :type nMeasurements: int
   :param noise: ``'off'`` (default) — noise disabled during calibration; ``'on'`` — noise enabled.
   :type noise: str
   :param invert: If ``True`` (default), return a :class:`~OOPAO.calibration.CalibrationVault.CalibrationVault` with the pseudo-inverse. If ``False``, return the raw interaction matrix.
   :type invert: bool
   :param nTrunc: Number of singular values to truncate in the inversion. Default ``0``.
   :type nTrunc: int
   :param print_time: Print elapsed time per iteration. Default ``False``.
   :type print_time: bool
   :param display: Show a tqdm progress bar. Default ``False``.
   :type display: bool
   :param single_pass: If ``True`` (default), only push. If ``False``, push and pull (average).
   :type single_pass: bool

   :returns: :class:`~OOPAO.calibration.CalibrationVault.CalibrationVault` (if ``invert=True``) or raw interaction matrix ``numpy.ndarray``.

   **Example**

   .. code-block:: python

      from OOPAO.calibration.InteractionMatrix import InteractionMatrix

      calib = InteractionMatrix(
          ngs    = ngs,
          tel    = tel,
          dm     = dm,
          wfs    = wfs,
          M2C    = M2C,
          stroke = 1e-9,
          noise  = 'off',
          display = True,
      )
      # Apply reconstructor in closed loop
      dm.coefs = -gain * M2C @ calib.M @ wfs.signal
