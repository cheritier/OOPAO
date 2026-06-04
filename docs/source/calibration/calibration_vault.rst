CalibrationVault
================

.. currentmodule:: OOPAO.calibration.CalibrationVault

Overview
--------

:class:`CalibrationVault` wraps the interaction matrix inversion, computing the pseudo-inverse via SVD. It stores the full and truncated reconstructors, singular values, and conditioning number.

API reference
-------------

.. class:: CalibrationVault(D, nTrunc=0, display=False, print_details=False, invert=True)

   SVD-based pseudo-inverse of an interaction matrix.

   :param D: Interaction matrix to invert (shape ``[n_measurements, n_modes]``).
   :type D: numpy.ndarray
   :param nTrunc: Number of singular values to truncate. Default ``0`` (no truncation).
   :type nTrunc: int
   :param display: If ``True``, plot the singular value spectrum. Default ``False``.
   :type display: bool
   :param print_details: If ``True``, print progress during SVD. Default ``False``.
   :type print_details: bool
   :param invert: If ``True`` (default), compute the pseudo-inverse. If ``False``, store ``D`` without inverting.
   :type invert: bool

   **Key attributes**

   .. attribute:: M
      :type: numpy.ndarray

      Full pseudo-inverse ``V @ S⁻¹ @ Uᵀ``.

   .. attribute:: Mtrunc
      :type: numpy.ndarray

      Truncated pseudo-inverse (``nTrunc`` smallest singular values removed).

   .. attribute:: s
      :type: numpy.ndarray

      Singular value vector.

   .. attribute:: cond
      :type: float

      Conditioning number (``s[0] / s[-1]``).

   .. attribute:: D
      :type: numpy.ndarray

      Reconstructed interaction matrix ``U @ S @ V``.

   **Example**

   .. code-block:: python

      from OOPAO.calibration.CalibrationVault import CalibrationVault

      calib = CalibrationVault(iMat, nTrunc=10, display=True)
      modal_commands = calib.M @ wfs.signal
