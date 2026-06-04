Source
======

.. currentmodule:: OOPAO.Source

Overview
--------

A :class:`Source` object represents a point source — either a Natural Guide Star (NGS) at infinite distance or a Laser Guide Star (LGS) at a finite altitude. It carries wavelength, flux, and positional information, and is the **primary carrier of the electromagnetic (EM) field** throughout the OOPAO propagation chain.

Each source maintains its own independent EM field and OPD, which are updated in-place as the source is propagated through optical elements. When multiple sources are grouped in an :class:`~OOPAO.Asterism.Asterism`, each one accumulates its own independent EM field.

Propagation operators
~~~~~~~~~~~~~~~~~~~~~

The :class:`Source` (and :class:`~OOPAO.Asterism.Asterism`) defines two propagation operators:

* **``*`` (``__mul__``)** — propagates the source through an optical element by calling ``obj.relay(self)``. Can be chained across multiple objects:

  .. code-block:: python

     ngs * tel * wfs        # calls tel.relay(ngs), then wfs.relay(ngs)

* **``**`` (``__pow__``)** — **resets** the source EM field and optical path history back to their initial state, then calls ``obj.relay(self)``. Use this to ensure a clean propagation from scratch:

  .. code-block:: python

     ngs ** tel * wfs       # reset EM field and optical path, then propagate

  .. note::

     ``**`` clears both the EM field amplitude/phase **and** the full optical path history (``src.optical_path``). Always use ``**`` at the start of a new propagation sequence to avoid accumulating state from a previous run.

Key concepts
~~~~~~~~~~~~

* **Optical band** — determines the wavelength via the built-in :meth:`photometry` method (bands inherited from OOMAO: ``'V'``, ``'R'``, ``'I'``, ``'J'``, ``'H'``, ``'K'``, etc.).
* **EM field** — the complex electromagnetic field carried by the source, updated by each ``relay()`` call along the chain.
* **OPD** — per-source optical path difference in metres, accumulated as the source propagates through the chain.
* **phase** — 2-D phase map in radians at the source wavelength, derived from the source OPD.
* **nPhoton** — photons per m² per second. Changing this property automatically recomputes the magnitude.
* **fluxMap** — 2-D map of photons per pixel per AO frame (depends on ``tel.samplingTime``).
* **LGS mode** — activated by setting a finite ``altitude``. Enables cone-effect interpolation in the :class:`~OOPAO.Atmosphere.Atmosphere`.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.Source import Source

   # Natural Guide Star in H-band, magnitude 8
   ngs = Source(optBand='H', magnitude=8)

   # Off-axis source at 30 arcsec, 45 degrees
   ngs_offaxis = Source(optBand='H', magnitude=9, coordinates=[30, 45])

   # Laser Guide Star at 90 km altitude
   lgs = Source(optBand='R', magnitude=6, altitude=90e3,
                laser_coordinates=[0, 0], Na_profile=na_array)

   # Propagate through the optical chain
   ngs * tel * wfs
   print(ngs.phase)    # phase map in radians
   print(ngs.nPhoton)  # photons / m^2 / s

   # Reset and re-propagate cleanly
   ngs ** tel * wfs

API reference
-------------

.. class:: Source(optBand, magnitude, coordinates=[0, 0], altitude=np.inf, laser_coordinates=[0, 0], Na_profile=None, fwhm_spot_up=None, display_properties=True, chromatic_shift=None)

   Point source object (NGS or LGS).

   :param optBand: Optical band identifier string (e.g. ``'V'``, ``'R'``, ``'I'``, ``'J'``, ``'H'``, ``'K'``). Determines wavelength and zero-point flux via :meth:`photometry`.
   :type optBand: str
   :param magnitude: Stellar magnitude in the chosen band.
   :type magnitude: float
   :param coordinates: On-sky position ``[radius_arcsec, angle_deg]`` from the telescope axis. Default ``[0, 0]``.
   :type coordinates: list[float]
   :param altitude: Source altitude in metres. ``np.inf`` = NGS (default). A finite value activates LGS mode and the cone effect.
   :type altitude: float
   :param laser_coordinates: Launch telescope position ``[x_m, y_m]`` in metres for LGS. Default ``[0, 0]``.
   :type laser_coordinates: list[float]
   :param Na_profile: Sodium layer profile as a 2-D array ``[altitude_m, profile_value]`` with *n* sampling points. Default ``None``.
   :type Na_profile: numpy.ndarray or None
   :param fwhm_spot_up: LGS uplink spot FWHM in arcseconds. Default ``None``.
   :type fwhm_spot_up: float or None
   :param display_properties: Print source properties on creation. Default ``True``.
   :type display_properties: bool
   :param chromatic_shift: List of shifts in arcseconds applied to the pupil footprint at each atmosphere layer (chromatic dispersion). Default ``None``.
   :type chromatic_shift: list or None

   **Key properties**

   .. attribute:: phase
      :type: numpy.ndarray

      2-D phase map in radians at the source wavelength, derived from the source's own OPD.

   .. attribute:: OPD
      :type: numpy.ndarray

      Per-source 2-D optical path difference in metres, accumulated as the source propagates through the chain.

   .. attribute:: optical_path
      :type: list

      Ordered list of optical elements the source has been propagated through. Cleared when ``**`` is used.

   .. attribute:: nPhoton
      :type: float

      Number of photons per m² per second. Setting this value updates :attr:`magnitude` automatically.

   .. attribute:: magnitude
      :type: float

      Stellar magnitude. Setting this value updates :attr:`nPhoton` automatically.

   .. attribute:: fluxMap
      :type: numpy.ndarray

      2-D photon flux map (photons per pixel per frame) over the telescope pupil, accounting for ``tel.samplingTime`` and ``tel.pixelArea``.

   .. attribute:: wavelength
      :type: float

      Central wavelength in metres, derived from :attr:`optBand`.

   .. attribute:: bandwidth
      :type: float

      Optical bandwidth in metres.

   .. attribute:: type
      :type: str

      ``'NGS'`` or ``'LGS'``, determined by the :attr:`altitude` parameter.

   .. attribute:: tag
      :type: str

      Always ``'source'``. Used internally to distinguish from :class:`~OOPAO.Asterism.Asterism` objects.

   .. attribute:: scintillation
      :type: numpy.ndarray

      2-D scintillation amplitude map (populated when the :class:`~OOPAO.Atmosphere.Atmosphere` uses diffractive propagation).

   **Methods**

   .. method:: photometry(optBand)

      Return ``(wavelength_m, bandwidth_m, zero_point_photons)`` for the given optical band string. Supported bands include: ``'U'``, ``'B'``, ``'V'``, ``'R'``, ``'I'``, ``'z'``, ``'Y'``, ``'J'``, ``'H'``, ``'K'``, ``'L'``, ``'M'``, ``'Na'``, ``'EOS'``, and several detector-specific bands.

      :param optBand: Band identifier string.
      :type optBand: str
      :returns: ``(wavelength, bandwidth, zeroPoint)``
      :rtype: tuple[float, float, float]

   .. method:: print_properties()

      Print a formatted summary of source properties (band, wavelength, magnitude, flux).

   **Operator summary**

   +--------------------+---------------------------------------------------------------+
   | Expression         | Effect                                                        |
   +====================+===============================================================+
   | ``ngs * obj``      | Call ``obj.relay(ngs)``; update EM field and OPD              |
   +--------------------+---------------------------------------------------------------+
   | ``ngs ** obj``     | Reset EM field and optical path, then call ``obj.relay(ngs)`` |
   +--------------------+---------------------------------------------------------------+
   | ``ngs * a * b``    | Chain: ``a.relay(ngs)``, then ``b.relay(ngs)``                |
   +--------------------+---------------------------------------------------------------+
   | ``ngs ** a * b``   | Reset, then chain through ``a`` and ``b``                     |
   +--------------------+---------------------------------------------------------------+
