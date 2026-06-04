NCPA
====

.. currentmodule:: OOPAO.NCPA

Overview
--------

The :class:`NCPA` (Non-Common Path Aberrations) class creates a static OPD offset to simulate aberrations that are seen by the science camera but not by the wavefront sensor. It generates the OPD as a linear combination of modal basis functions (KL modes, Zernike polynomials, or a user-supplied M2C matrix), optionally following a 1/f² power law distribution over modes.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.NCPA import NCPA

   # Blank NCPA (can assign OPD map manually)
   ncpa = NCPA(tel, dm, atm)
   ncpa.OPD = my_opd_map

   # NCPA from specific KL mode coefficients
   ncpa = NCPA(tel, dm, atm, coefficients=[0, 0, 50e-9, 100e-9])

   # 1/f² NCPA: 200 nm RMS, modes 5–25, cutoff frequency 1
   ncpa = NCPA(tel, dm, atm, f2=[200e-9, 5, 25, 1])

   # Propagate
   src * tel * ncpa * wfs

API reference
-------------

.. class:: NCPA(tel, dm, atm, modal_basis='KL', coefficients=None, f2=None, seed=5, M2C=None)

   Static non-common-path aberration map.

   :param tel: Telescope object.
   :type tel: Telescope
   :param dm: Deformable mirror object (used to compute the modal basis).
   :type dm: DeformableMirror
   :param atm: Atmosphere object (used to compute the KL basis).
   :type atm: Atmosphere
   :param modal_basis: Basis used to generate the OPD. ``'KL'`` (default), ``'Zernike'``, or ``'M2C'``.
   :type modal_basis: str
   :param coefficients: List of modal coefficients in metres. Length determines how many modes are used. Default ``None`` (blank OPD).
   :type coefficients: list[float] or None
   :param f2: Four-element list ``[amplitude_m, start_mode, end_mode, cutoff_freq]`` for a 1/f² power-law NCPA. Default ``None``.
   :type f2: list or None
   :param seed: Random seed for reproducible 1/f² generation. Default ``5``.
   :type seed: int
   :param M2C: Mode-to-command matrix for the ``'M2C'`` basis option. Default ``None``.
   :type M2C: numpy.ndarray or None

   **Key properties**

   .. attribute:: OPD
      :type: numpy.ndarray

      2-D OPD map in metres. Can be set directly to an arbitrary map.

   .. attribute:: tag
      :type: str

      Always ``'NCPA'``.

   **Operator summary**

   +----------------------------+---------------------------------------------------+
   | Expression                 | Effect                                            |
   +============================+===================================================+
   | ``src * tel * ncpa * wfs`` | Add NCPA OPD to telescope OPD before WFS sensing  |
   +----------------------------+---------------------------------------------------+
