Deformable Mirror
=================

.. currentmodule:: OOPAO.DeformableMirror

Overview
--------

The :class:`DeformableMirror` (DM) models a continuous face-sheet mirror whose surface is described by a set of actuator influence functions. By default, influence functions are 2-D Gaussians in a Fried geometry (one actuator per subaperture, aligned with the WFS grid), but fully user-defined influence functions or modal bases can also be supplied.

.. important::

   The DM is modelled as **transmissive**, not reflective, to avoid factor-of-2 confusion in OPD. The OPD produced is the physical surface deformation, not twice that value.

The DM is conjugated to the ground layer by default. An optional ``altitude`` parameter places it at a conjugate height (e.g. for MCAO simulations).

Key concepts
~~~~~~~~~~~~

* **Fried geometry** — actuators are placed at subaperture corners, with ``nSubap + 1`` actuators across the diameter.
* **Mechanical coupling** — Gaussian sigma is set so that the deformation at distance ``pitch`` from a pushed actuator equals ``mechCoupling`` (default 0.35).
* **coefs** — the actuator command vector. Setting ``dm.coefs = vec`` immediately recomputes :attr:`OPD`.
* **Mis-registration** — a :class:`~OOPAO.MisRegistration.MisRegistration` object applies rotation, shift, and scaling to the influence functions at construction time.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.DeformableMirror import DeformableMirror

   dm = DeformableMirror(telescope=tel, nSubap=20, mechCoupling=0.35)

   # Apply a flat mirror (zero commands)
   dm.coefs = 0

   # Push the first actuator by 100 nm
   import numpy as np
   cmd = np.zeros(dm.nValidAct)
   cmd[0] = 100e-9
   dm.coefs = cmd

   # Propagate through the DM
   tel * dm   # tel.OPD += dm.OPD

API reference
-------------

.. class:: DeformableMirror(telescope, nSubap, mechCoupling=0.35, coordinates=None, pitch=None, modes=None, misReg=None, M4_param=None, nJobs=30, nThreads=20, print_dm_properties=True, floating_precision=64, altitude=None, flip=False, flip_lr=False, sign=1, actuator_selection=None)

   Deformable mirror with Gaussian or user-defined influence functions.

   :param telescope: Associated telescope object. Provides pupil, pixel scale, and obstruction geometry for actuator selection.
   :type telescope: Telescope
   :param nSubap: Number of WFS subapertures across the pupil diameter. Defines the Fried geometry actuator pitch when ``coordinates`` and ``pitch`` are ``None``.
   :type nSubap: float
   :param mechCoupling: Mechanical coupling coefficient — deformation at distance ``pitch`` from a unit-pushed actuator. Ignored when ``modes`` is specified. Default ``0.35``.
   :type mechCoupling: float
   :param coordinates: User-defined actuator coordinates as an ``(N, 2)`` array in metres. When supplied, all actuators are treated as valid regardless of pupil geometry. Default ``None``.
   :type coordinates: numpy.ndarray or None
   :param pitch: Inter-actuator distance in metres for the Gaussian influence function width. Defaults to ``diameter / nSubap`` when ``None``.
   :type pitch: float or None
   :param modes: User-defined influence functions as a 2-D matrix of shape ``(resolution², n_modes)``, where each column is a flattened 2-D mode. Default ``None``.
   :type modes: numpy.ndarray or None
   :param misReg: :class:`~OOPAO.MisRegistration.MisRegistration` object specifying geometric transformations to apply. Ignored when ``modes`` is provided. Default ``None``.
   :type misReg: MisRegistration or None
   :param M4_param: Parameter file for ELT M4 mirror model. Default ``None``.
   :type M4_param: object or None
   :param nJobs: Number of parallel jobs for influence function computation. Default ``30``.
   :type nJobs: int
   :param nThreads: Number of threads per job. Default ``20``.
   :type nThreads: int
   :param print_dm_properties: Print DM properties at initialisation. Default ``True``.
   :type print_dm_properties: bool
   :param floating_precision: Array precision: ``64`` (float64, default) or ``32`` (float32, lower memory).
   :type floating_precision: int
   :param altitude: Conjugation altitude in metres. ``None`` = ground conjugation (default).
   :type altitude: float or None
   :param flip: Flip influence functions vertically. Default ``False``.
   :type flip: bool
   :param flip_lr: Flip influence functions horizontally. Default ``False``.
   :type flip_lr: bool
   :param sign: Sign applied to DM commands (``+1`` or ``-1``). Default ``1``.
   :type sign: int
   :param actuator_selection: Controls which actuators are selected as valid.

      * ``None`` — selected by pupil geometry (default).
      * Scalar — selected by standard deviation of influence function within the pupil (threshold = scalar).
      * ``[r_inner, r_outer]`` — selected by radial distance ``r_inner < r < r_outer``.
   :type actuator_selection: None, float, or list[float]

   **Key properties**

   .. attribute:: coefs
      :type: numpy.ndarray or float

      Actuator command vector in metres (units of the influence functions). Setting this property immediately recomputes :attr:`OPD`. Accepts a scalar (broadcasts to all valid actuators) or a 1-D array of length :attr:`nValidAct`.

   .. attribute:: OPD
      :type: numpy.ndarray

      2-D optical path difference map in metres produced by the current commands.

   .. attribute:: nValidAct
      :type: int

      Number of valid (selected) actuators.

   .. attribute:: modes
      :type: numpy.ndarray

      2-D influence function matrix of shape ``(resolution², nValidAct)``.

   .. attribute:: pitch
      :type: float

      Inter-actuator pitch in metres.

   .. attribute:: misReg
      :type: MisRegistration

      Current mis-registration state of the DM.

   **Methods**

   .. method:: set_coefs_value(actuator_index, actuator_coefs)

      Update the commands of one or more individual actuators and recompute the DM surface accordingly.
      Use this when you need to set specific actuators without rebuilding the full command vector.

      :param actuator_index: List of actuator indices to update.
      :type actuator_index: list[int]
      :param actuator_coefs: List of command values in metres, one per index.
      :type actuator_coefs: list[float]

      .. code-block:: python

         # Push actuator 42 by 50 nm
         dm.set_coefs_value([42], [50e-9])

         # Push two actuators simultaneously
         dm.set_coefs_value([10, 42], [50e-9, -50e-9])

      .. note::

         This is distinct from ``dm.coefs = value``, which replaces the **entire** command vector
         and expects a 1-D array of length :attr:`nValidAct`. Use :meth:`set_coefs_value` when
         updating individual actuators; use ``dm.coefs`` when applying a full modal correction.

   .. method:: display()

      Show the current OPD map and actuator grid in a matplotlib figure.

   **Operator summary**

   +-------------------+-----------------------------------------------+
   | Expression        | Effect                                        |
   +===================+===============================================+
   | ``tel * dm``      | Add ``dm.OPD`` to ``tel.OPD``                 |
   +-------------------+-----------------------------------------------+
   | ``src * tel * dm``| Full propagation: source → telescope → DM     |
   +-------------------+-----------------------------------------------+
