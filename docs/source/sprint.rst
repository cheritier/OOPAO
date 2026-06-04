SPRINT
======

.. currentmodule:: OOPAO.SPRINT

Overview
--------

:class:`SPRINT` (System Parameter Recurrent INvasive Tracking) estimates DM–WFS mis-registration parameters from sensitivity matrices. It computes the gradient of the interaction matrix with respect to each mis-registration degree of freedom, then inverts the problem to identify the actual mis-registration from a measured interaction matrix.

The algorithm supports up to 6 degrees of freedom: shift X, shift Y, rotation, anamorphosis, radial scaling, and tangential scaling.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.SPRINT import SPRINT

   sprint = SPRINT(ao_obj, basis=Z)

   # Estimate mis-registration
   sprint.estimate(ao_obj.calib.D)
   print(sprint.mis_registration_estimated)

API reference
-------------

.. class:: SPRINT(obj, basis, nameFolder=None, nameSystem=None, mis_registration_zero_point=None, wfs_mis_registered=None, fast_algorithm=False, n_mis_reg=3, recompute_sensitivity=False, dm_input=None, ind_mis_reg=None)

   DM/WFS mis-registration identification algorithm.

   :param obj: AO system object containing ``tel``, ``dm``, ``wfs``, and ``calib``.
   :type obj: object
   :param basis: Modal basis object with a ``modes`` attribute.
   :type basis: object
   :param nameFolder: Folder to save/load sensitivity matrices. Default ``None`` (uses ``param['pathInput']``).
   :type nameFolder: str or None
   :param nameSystem: System identifier for file naming. Default ``None``.
   :type nameSystem: str or None
   :param mis_registration_zero_point: Reference :class:`~OOPAO.MisRegistration.MisRegistration`. Default ``None`` (uses ``dm.misReg``).
   :type mis_registration_zero_point: MisRegistration or None
   :param wfs_mis_registered: WFS-side mis-registration object. Default ``None``.
   :type wfs_mis_registered: MisRegistration or None
   :param fast_algorithm: Use the faster (less stable) algorithm variant. Default ``False``.
   :type fast_algorithm: bool
   :param n_mis_reg: Number of mis-registration degrees of freedom to estimate. Default ``3`` (shiftX, shiftY, rotation).
   :type n_mis_reg: int
   :param recompute_sensitivity: Force recomputation of sensitivity matrices even if saved files exist. Default ``False``.
   :type recompute_sensitivity: bool
   :param ind_mis_reg: Indices of the mis-registration DOF to estimate. Default ``None`` (first ``n_mis_reg``).
   :type ind_mis_reg: numpy.ndarray or None

   **Key attributes**

   .. attribute:: mis_registration_estimated
      :type: MisRegistration

      Estimated mis-registration from the last :meth:`estimate` call.

   **Methods**

   .. method:: estimate(interaction_matrix)

      Estimate mis-registration parameters from a measured interaction matrix.

      :param interaction_matrix: Measured interaction matrix.
      :type interaction_matrix: numpy.ndarray

   .. note::

      Tutorial: ``tutorials/how_to_SPRINT.py``
