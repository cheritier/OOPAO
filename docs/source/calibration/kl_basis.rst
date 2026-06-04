KL Modal Basis
==============

.. currentmodule:: OOPAO.calibration.compute_KL_modal_basis

Overview
--------

:func:`compute_KL_basis` computes the Karhunen-Loève (KL) modal basis for a DM by diagonalising the atmosphere covariance projected onto the DM influence functions. The resulting basis is ordered from lowest to highest spatial frequency (first modes are piston, tip, tilt, then higher-order modes).

This is the recommended modal basis for closed-loop AO control in OOPAO.

API reference
-------------

.. function:: compute_KL_basis(tel, atm, dm, lim=1e-3, remove_piston=True, n_batch=1)

   Compute the KL modal basis for a DM/atmosphere combination.

   :param tel: Telescope object.
   :type tel: Telescope
   :param atm: Atmosphere object (defines the turbulence statistics for the covariance).
   :type atm: Atmosphere
   :param dm: Deformable mirror object.
   :type dm: DeformableMirror
   :param lim: Inversion threshold for the covariance matrix. Default ``1e-3``.
   :type lim: float
   :param remove_piston: If ``True``, remove the piston mode from the basis. Default ``True``.
   :type remove_piston: bool
   :param n_batch: Number of batches for memory-efficient covariance computation. Default ``1``.
   :type n_batch: int

   :returns: Mode-to-command (M2C) matrix of shape ``(n_actuators, n_modes)``.
   :rtype: numpy.ndarray

   **Example**

   .. code-block:: python

      from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

      M2C = compute_KL_basis(tel, atm, dm)
      dm.coefs = M2C   # visualise all modes as OPD maps

.. currentmodule:: OOPAO.calibration.get_modal_basis

.. function:: get_modal_basis_from_ao_obj(ao_obj, nameFolderBasis=None, nameBasis=None)

   Load a previously saved KL modal basis from disk and populate the AO object.

   :param ao_obj: AO system object.
   :type ao_obj: object
   :param nameFolderBasis: Folder containing the saved basis FITS file. Default ``None``.
   :type nameFolderBasis: str or None
   :param nameBasis: Basis filename without extension. Default ``None`` (auto-generated).
   :type nameBasis: str or None

   :returns: Object with ``M2C``, ``basis``, and optionally ``projector``.
   :rtype: object
