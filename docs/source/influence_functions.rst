Influence Functions
===================

.. currentmodule:: OOPAO.InfluenceFunctions

Overview
--------

The :class:`InfluenceFunctions` class generates the influence functions of real, user-defined deformable mirrors. It acts as a dispatcher: the heavy, system-specific computation is delegated to a dedicated function in ``OOPAO.tools.user_defined_influence_functions_``, imported lazily so that the optional dependencies (``scipy.io``, ``skimage``, ``joblib``) are only loaded when actually needed. Generic operations — left/right and up/down flips, sign convention, actuator centroids — are then applied uniformly whatever the system.

The resulting cube can be passed to a :class:`~OOPAO.DeformableMirror.DeformableMirror` through its ``modes`` argument to simulate the real mirror geometry instead of the default Gaussian model.

.. note::

   Most systems require **input data that is not shipped with OOPAO**. The RAMA and EKARUS cubes are available from public download links (see the table below); the ELT-M4 FEM data and the LBT-ASM eigen modes are available on request (cedric-taissir.heritier@lam.fr). ``PAPYRUS_DM241`` and ``GHOST_DM492`` are modelled from actuator coordinates and need little or no external data.

Supported systems
~~~~~~~~~~~~~~~~~

+--------------------+------------------------------------------------------------+----------------------------------+
| ``name_system``    | Description                                                | Input data                       |
+====================+============================================================+==================================+
| ``ELT_M4``         | ELT M4 from reduced FEM data (892 actuators per segment)   | On request                       |
+--------------------+------------------------------------------------------------+----------------------------------+
| ``LBT_ASM``        | LBT adaptive secondary from the mirror eigen modes         | On request                       |
+--------------------+------------------------------------------------------------+----------------------------------+
| ``RAMA_DM97``      | RAMA 97-actuator DM (measured cube)                        | Public download link             |
+--------------------+------------------------------------------------------------+----------------------------------+
| ``EKARUS_DM468``   | EKARUS 468-actuator DM (zonal cube)                        | Public download link             |
+--------------------+------------------------------------------------------------+----------------------------------+
| ``GHOST_DM492``    | GHOST 492-actuator DM (modelled from coordinates)          | Coordinates file                 |
+--------------------+------------------------------------------------------------+----------------------------------+
| ``PAPYRUS_DM241``  | PAPYRUS 241-actuator DM (Gaussian model)                   | None                             |
+--------------------+------------------------------------------------------------+----------------------------------+

Key concepts
~~~~~~~~~~~~

* **Influence-function cube** — 3-D array of shape ``[n_act, resolution, resolution]``, one 2-D phase map per actuator, sampled at the telescope pupil resolution.
* **Mis-registration** — a :class:`~OOPAO.MisRegistration.MisRegistration` object (shift, rotation, radial/tangential scaling, anamorphosis) applied during the generation, either analytically on the interpolation coordinates (``ELT_M4``) or as a geometric image transformation (other systems).
* **Sign convention** — global sign applied to the cube (``sign=1`` or ``-1``), to match the push/pull convention of the rest of the simulation.
* **Lazy dispatch** — the system-specific compute function is resolved through a dispatch table and imported only when the class is instantiated; unknown system names raise an :class:`~OOPAO.tools.tools.OopaoError` listing the valid options.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.tools.influence_functions import InfluenceFunctions
   from OOPAO.MisRegistration import MisRegistration

   # Optional mis-registration of the DM
   misReg = MisRegistration()
   misReg.rotationAngle = 1.5     # degrees
   misReg.shiftX = 0.05           # metres

   # Generate the ELT-M4 influence functions at the pupil resolution
   IFs = InfluenceFunctions(name_system='ELT_M4',
                            diameter=39,
                            resolution=480,
                            loc='/path/to/M4_data/',
                            mis_registration=misReg,
                            specific_parameters={'n_segments': 6,
                                                 'new_arrangement': False,
                                                 'parallel': True,
                                                 'n_jobs': 6})

   cube = IFs.influence_function_2D    # [n_act, resolution, resolution]
   coords = IFs.coordinates            # actuator centroids

Recent developments
~~~~~~~~~~~~~~~~~~~

.. versionchanged:: 26.7

   * **ELT_M4** — the interpolation of the FEM data onto the pupil grid is now performed through a precomputed sparse barycentric-interpolation operator (one Delaunay triangulation per segment, then a single sparse matrix product for the 892 influence functions of the segment). This reproduces ``scipy.interpolate.griddata(..., method='linear')`` with out-of-hull points set to 0, at a fraction of the cost. Optional segment-level parallelization is available through ``specific_parameters['parallel']``.
   * **LBT_ASM** — the requested ``resolution`` can now be *larger* than the native resolution of the eigen-mode maps (upsampling). The mode cube is padded symmetrically before the geometric transformation so that the magnified pupil fits in the canvas; physical pixel scales are computed before padding and are therefore unaffected. The crop uses explicit end indices, making the degenerate case ``resolution == native resolution`` safe.
   * **LBT_ASM** — ``skimage.transform.warp`` now uses bilinear interpolation (``order=1``) instead of nearest-neighbour (``order=0``), removing blocky artefacts when upsampling. Downsampled influence functions therefore differ at the sub-percent level from previous versions.
   * **Dispatcher** — :class:`InfluenceFunctions` resolves the compute function through a single dispatch table instead of repeated ``if`` blocks; unknown system names raise an :class:`~OOPAO.tools.tools.OopaoError` listing the valid options.

API reference
-------------

.. class:: InfluenceFunctions(name_system, diameter, resolution, loc=None, mis_registration=None, flip_lr=False, flip_ud=False, specific_parameters=None, sign=1)

   Compute the influence functions of a user-defined system.

   :param name_system: Name of the system. One of ``'ELT_M4'``, ``'EKARUS_DM468'``, ``'LBT_ASM'``, ``'GHOST_DM492'``, ``'PAPYRUS_DM241'``, ``'RAMA_DM97'``.
   :type name_system: str
   :param diameter: Telescope diameter in metres.
   :type diameter: float
   :param resolution: Number of pixels across the pupil diameter.
   :type resolution: int
   :param loc: Path to the system input data. Default ``None``.
   :type loc: str or None
   :param mis_registration: Mis-registration applied during the generation (shift, rotation, radial/tangential scaling, anamorphosis). Default ``None`` (no mis-registration).
   :type mis_registration: MisRegistration or None
   :param flip_lr: Flip the influence functions left/right. Default ``False``.
   :type flip_lr: bool
   :param flip_ud: Flip the influence functions up/down. Default ``False``.
   :type flip_ud: bool
   :param specific_parameters: System-specific parameters; currently used by ``ELT_M4`` only (keys ``'n_segments'``, ``'new_arrangement'``, ``'parallel'``, ``'n_jobs'``). Default ``None``.
   :type specific_parameters: dict or None
   :param sign: Sign convention applied to the influence functions. Default ``1``.
   :type sign: int

   :raises OopaoError: If ``name_system`` is not recognized, or if the input data cannot be found at ``loc``.

   **Key properties**

   .. attribute:: influence_function_2D
      :type: numpy.ndarray

      Influence-function cube of shape ``[n_act, resolution, resolution]``, with flips and sign applied.

   .. attribute:: coordinates
      :type: numpy.ndarray

      Actuator coordinates computed as the centroids of the (flipped) influence functions.

   .. attribute:: sign
      :type: int

      Sign convention that was applied to the cube.

System-specific compute functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All compute functions live in ``OOPAO.tools.user_defined_influence_functions_`` and share the common signature below; they are normally called through :class:`InfluenceFunctions` rather than directly. Each returns the raw influence-function cube ``[n_act, resolution, resolution]`` (no flips or sign applied).

.. function:: compute_ELT_M4_influence_functions(name_system, diameter, resolution, loc=None, mis_registration=None, flip_lr=False, flip_ud=False, specific_parameters=None)

   Generate the ELT-M4 influence functions from the reduced FEM data. The FEM node values are interpolated onto the (distorted, mis-registered) pupil coordinates through a precomputed sparse barycentric operator — one Delaunay triangulation per segment. The mis-registration is applied analytically to the query coordinates, so no image resampling is required.

   :param specific_parameters: ``'n_segments'`` (int, number of M4 petals, default 6), ``'new_arrangement'`` (bool, use the ``IDX_NEW.fits`` actuator ordering, default ``False``), ``'parallel'`` (bool, one joblib process per segment, default ``False``), ``'n_jobs'`` (int, capped at ``n_segments``, default 6).
   :type specific_parameters: dict or None

.. function:: compute_LBT_ASM_influence_functions(name_system, diameter, resolution, loc=None, mis_registration=None, flip_lr=False, flip_ud=False, specific_parameters=None)

   Compute the LBT ASM influence functions from the mirror eigen modes (``phase_matrix.sav``), projected from the modal to the zonal basis through the pseudo-inverse of the modes-to-commands matrix (``m2c.fits``). Down- and up-sampling to the requested resolution are supported; when upsampling, the mode cube is padded symmetrically before the geometric transformation.

.. function:: compute_RAMA_DM97_influence_functions(name_system, diameter, resolution, loc=None, mis_registration=None, flip_lr=False, flip_ud=False, specific_parameters=None)

   Load the measured RAMA DM97 cube (``IF_97.npy``), crop and re-center it, and resample it to the requested resolution through :func:`~OOPAO.tools.interpolateGeometricalTransformation.interpolate_cube`.

.. function:: compute_EKARUS_DM468_influence_functions(name_system, diameter, resolution, loc=None, mis_registration=None, flip_lr=False, flip_ud=False, specific_parameters=None)

   Load the EKARUS DM468 zonal cube (``IF_zonal_cube.npy``) and resample it to the requested resolution through :func:`~OOPAO.tools.interpolateGeometricalTransformation.interpolate_cube`.

.. function:: compute_GHOST_DM492_influence_functions(name_system, diameter, resolution, loc=None, mis_registration=None, flip_lr=False, flip_ud=False, specific_parameters=None)

   Build the GHOST DM492 influence functions from the measured actuator coordinates (``dm_coord.mat``) using the standard OOPAO :class:`~OOPAO.DeformableMirror.DeformableMirror` Gaussian model (mechanical coupling 0.15).

.. function:: compute_PAPYRUS_DM241_influence_functions(name_system, diameter, resolution, loc=None, mis_registration=None, flip_lr=False, flip_ud=False, specific_parameters=None)

   Build the PAPYRUS DM241 influence functions from a cartesian 17x17 actuator grid projected onto the T152 pupil, using the standard OOPAO :class:`~OOPAO.DeformableMirror.DeformableMirror` Gaussian model (mechanical coupling 0.36). Requires no external data.

Adding a new system
~~~~~~~~~~~~~~~~~~~

Three steps are required:

1. Write a ``compute_<NAME>_influence_functions(...)`` function in ``user_defined_influence_functions_.py`` following the common signature above and returning a ``[n_act, resolution, resolution]`` cube.
2. Add one entry to the dispatch dictionary in ``InfluenceFunctions._get_compute_function``.
3. Add one row to the *Supported systems* table of this page.
