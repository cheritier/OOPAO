Atmosphere
==========

.. currentmodule:: OOPAO.Atmosphere

Overview
--------

The :class:`Atmosphere` object simulates one or more turbulent layers following Von Kármán statistics. Each layer is independent and has its own altitude, wind speed, wind direction, and fractional Cn² contribution. The total turbulence strength is parameterised by the Fried parameter ``r0`` and the outer scale ``L0``, both expressed at 500 nm.

The correct way to include the atmosphere in a simulation is to place it directly in the propagation chain, before the :class:`~OOPAO.Telescope.Telescope`:

.. code-block:: python

   ngs ** atm * tel * wfs

When ``atm.relay(ngs)`` is called, it updates the **source's** OPD and EM field with the atmospheric phase contribution. The pupil is only applied once the source passes through ``tel`` — propagating through ``atm`` alone (without ``tel``) gives an unpupilled wavefront.

.. note::

   The atmosphere does **not** advance its phase screens automatically during propagation. :meth:`update` must be called explicitly at the start of each loop iteration before propagating.

The legacy ``tel + atm`` coupling (which updated ``tel.OPD`` directly) is retained for **backward compatibility only**. New code should always use the chain syntax above.

Three propagation modes are available:

1. **Geometric only** (default) — phase-only, no scintillation.
2. **Diffractive** — angular-spectrum propagation through each layer; produces amplitude fluctuations (scintillation).
3. **Diffractive + geometric backup** — same as mode 2 but also stores the purely geometric phase for comparison.

Key concepts
~~~~~~~~~~~~

* **Phase screens** — each layer stores an infinitely-scrolling phase screen computed from a power-law spectral model. The screen is larger than the telescope aperture to allow continuous scrolling without repetition.
* **relay()** — updates the source OPD and EM field with the atmospheric phase. Called automatically by ``ngs * atm``. Does not advance the phase screens.
* **update()** — advances all phase screens by one AO time step. Must be called explicitly each loop iteration. Also accepts a user-supplied OPD to inject an arbitrary wavefront directly (bypassing the turbulence screens).
* **Cone effect** — when a :class:`~OOPAO.Source.Source` is an LGS (finite altitude), the footprint on each layer is scaled accordingly via bilinear interpolation.
* **Multi-source / asterism** — when an :class:`~OOPAO.Asterism.Asterism` is propagated, each source receives an independent OPD slice computed from the atmosphere layers according to its sky coordinates and altitude.
* **Elevation** — the effective ``r0`` and layer altitudes are scaled by ``sin(elevation)``; default is 90° (zenith).
* **On-the-fly updates** — wind speed, direction, r0, and elevation can be changed at runtime.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.Atmosphere import Atmosphere

   atm = Atmosphere(
       telescope     = tel,
       r0            = 0.15,          # Fried parameter [m] at 500 nm
       L0            = 25.0,          # Outer scale [m]
       windSpeed     = [10.0, 5.0],   # [m/s] per layer
       fractionalR0  = [0.6, 0.4],    # Cn2 weights (must sum to 1)
       windDirection = [0.0, 45.0],   # [deg] per layer
       altitude      = [0.0, 5000.0], # [m] per layer
   )

   # Closed-loop example
   for i in range(nLoop):
       atm.update()                         # advance phase screens
       ngs ** atm * tel * wfs               # reset, then propagate
       dm.coefs -= gain * M2C @ wfs.signal  # apply correction

   # Inject a user-defined OPD instead of turbulence
   atm.update(my_opd_map)       # positional
   atm.update(OPD=my_opd_map)   # keyword — both are equivalent
   ngs ** atm * tel * wfs

   # Change wind speed on the fly
   atm.windSpeed = [12.0, 7.0]

   # Legacy coupling (backward compatibility only)
   tel + atm    # tel.OPD mirrors atm.OPD — avoid in new code
   tel - atm    # decouple

API reference
-------------

.. class:: Atmosphere(telescope, r0, L0, windSpeed, fractionalR0, windDirection, altitude, src=None, param=None, elevation=90.0, mode=2, angular_spectrum_propagation=False, geometric_phase_backup=False, unwrap_diffractive_phase=False, rytov_var=None, rytov_wvl=500e-9)

   Multi-layer turbulent atmosphere with Von Kármán statistics.

   :param telescope: Telescope object. Carries pupil definition, pixel size, and sampling time. Required at construction even when using the chain syntax.
   :type telescope: Telescope
   :param r0: Fried parameter in metres at 500 nm at the specified elevation.
   :type r0: float
   :param L0: Outer scale in metres.
   :type L0: float
   :param windSpeed: Wind speed for each layer in m/s.
   :type windSpeed: list[float]
   :param fractionalR0: Cn² profile weights for each layer. Values must sum to 1.
   :type fractionalR0: list[float]
   :param windDirection: Wind direction for each layer in degrees.
   :type windDirection: list[float]
   :param altitude: Altitude for each layer in metres.
   :type altitude: list[float]
   :param src: Source object. If ``None``, uses ``telescope.src``. Default ``None``.
   :type src: Source or None
   :param param: Parameter file object. When provided, covariance matrices are saved/loaded from disk to avoid recomputation. Default ``None``.
   :type param: object or None
   :param elevation: Telescope elevation in degrees. Scales effective r0 and layer altitudes. Default ``90.0`` (zenith).
   :type elevation: float
   :param mode: Spectral model for phase screen generation. ``1`` uses aotools; ``2`` uses the OOPAO internal model (default).
   :type mode: int
   :param angular_spectrum_propagation: If ``True``, enables diffractive propagation and scintillation computation. Default ``False``.
   :type angular_spectrum_propagation: bool
   :param geometric_phase_backup: If ``True``, stores the geometric phase alongside the diffractive phase. Requires ``angular_spectrum_propagation=True``. Default ``False``.
   :type geometric_phase_backup: bool
   :param unwrap_diffractive_phase: If ``True``, unwraps the diffractive phase after propagation. Default ``False``.
   :type unwrap_diffractive_phase: bool
   :param rytov_var: Override for the Rytov scintillation variance. If ``None``, computed from Cn² and wavelength. Default ``None``.
   :type rytov_var: float or None
   :param rytov_wvl: Wavelength at which the Rytov variance is computed, in metres. Default ``500e-9``.
   :type rytov_wvl: float

   **Key properties**

   .. attribute:: OPD
      :type: numpy.ndarray

      Current 2-D OPD map in metres (sum of all layers projected onto the telescope pupil). Used by the legacy ``tel + atm`` coupling. In the chain syntax, the source OPD is updated directly by :meth:`relay`.

   .. attribute:: r0
      :type: float

      Fried parameter in metres at 500 nm. Can be updated at runtime.

   .. attribute:: windSpeed
      :type: list[float]

      Per-layer wind speeds in m/s. Can be updated at runtime.

   .. attribute:: windDirection
      :type: list[float]

      Per-layer wind directions in degrees. Can be updated at runtime.

   .. attribute:: elevation
      :type: float

      Telescope elevation in degrees. Changing this rescales the effective r0 and layer altitudes.

   .. attribute:: nLayer
      :type: int

      Number of turbulent layers.

   .. attribute:: seeingArcsec
      :type: float

      Seeing in arcseconds at 500 nm (computed from r0).

   **Methods**

   .. method:: relay(src)

      Core propagation method. Called automatically by ``src * atm``. Updates the source OPD and EM field with the atmospheric phase contribution. Does **not** advance the phase screens — call :meth:`update` first.

      :param src: Source or Asterism being propagated.
      :type src: Source or Asterism

   .. method:: update(OPD=None)

      Advance all phase screens by one AO time step (``tel.samplingTime``).

      If ``OPD`` is supplied, it is used directly to set the source OPD and phase, bypassing the turbulence screens entirely. This allows injection of arbitrary wavefronts into the propagation chain.

      :param OPD: User-supplied 2-D OPD map in metres. If ``None`` (default), the phase screens are advanced normally.
      :type OPD: numpy.ndarray or None

      Both calling styles are equivalent:

      .. code-block:: python

         atm.update(my_opd_map)       # positional
         atm.update(OPD=my_opd_map)   # keyword

   .. method:: generateNewPhaseScreen(seed)

      Regenerate all phase screens from a new random seed.

      :param seed: Random seed.
      :type seed: int

   .. method:: display()

      Display the current phase screen for each layer as a matplotlib figure.

   **Operator summary**

   +----------------------+-----------------------------------------------------------+
   | Expression           | Effect                                                    |
   +======================+===========================================================+
   | ``ngs * atm``        | Call ``atm.relay(ngs)``; update source OPD and EM field   |
   +----------------------+-----------------------------------------------------------+
   | ``ngs ** atm``       | Reset source EM field and optical path, then relay        |
   +----------------------+-----------------------------------------------------------+
   | ``ngs ** atm * tel`` | Full reset + atmospheric phase + pupil applied by tel     |
   +----------------------+-----------------------------------------------------------+
   | ``tel + atm``        | Legacy: ``tel.OPD`` mirrors ``atm.OPD`` (avoid in new code)|
   +----------------------+-----------------------------------------------------------+
   | ``tel - atm``        | Legacy: decouple atmosphere from telescope                |
   +----------------------+-----------------------------------------------------------+

   .. note::

      All wavelength-dependent atmosphere parameters (r0, seeing) are expressed at 500 nm by convention, regardless of the source wavelength. The :class:`~OOPAO.Source.Source` wavelength is used only when computing the phase from the OPD (``phase = OPD * 2π / wavelength``).
