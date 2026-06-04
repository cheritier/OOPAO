Telescope
=========

.. currentmodule:: OOPAO.Telescope

Overview
--------

The :class:`Telescope` defines the entrance pupil geometry of the simulated instrument — shape, diameter, central obstruction, and optional spider arms. It is a required argument when constructing most other OOPAO classes (:class:`~OOPAO.Atmosphere.Atmosphere`, :class:`~OOPAO.DeformableMirror.DeformableMirror`, :class:`~OOPAO.Pyramid.Pyramid`, etc.), because it carries the pupil definition, pixel scale, and AO loop sampling time.

.. note::

   In the current version of OOPAO, the :class:`~OOPAO.Source.Source` object is the **primary carrier of the electromagnetic (EM) field and phase**. Each source maintains its own independent OPD and EM field, which is what all ``relay()`` methods operate on. The ``Telescope.OPD`` property is retained for **backward compatibility only** and should not be relied upon in new code.

The propagation chain
~~~~~~~~~~~~~~~~~~~~~

Light propagation in OOPAO is driven by the ``__mul__`` (``*``) operator, which is defined on :class:`~OOPAO.Source.Source` and :class:`~OOPAO.Asterism.Asterism`. Writing ``ngs * obj`` calls ``obj.relay(ngs)`` under the hood. Every core OOPAO class implements a ``relay()`` method that receives the source and operates directly on its EM field.

A full propagation chain is simply a sequence of ``*`` operations:

.. code-block:: python

   ngs * tel * wfs        # relay called on tel, then on wfs

When multiple sources are grouped in an :class:`~OOPAO.Asterism.Asterism`, the same mechanics apply — ``__mul__`` iterates ``relay()`` over each source independently, so every source accumulates its own EM field:

.. code-block:: python

   ast * tel * wfs        # relay called on tel and wfs for each source in ast

The ``**`` operator resets the source EM field to its initial (unaberrated) state before starting the propagation. Use it to ensure a clean propagation from scratch:

.. code-block:: python

   ngs ** tel * wfs       # reset ngs EM field, then propagate through tel and wfs

Key concepts
~~~~~~~~~~~~

* **Pupil mask** — binary (or user-supplied reflectivity) 2-D array defining the aperture geometry.
* **OPD** *(backward compatibility)* — 2-D optical path difference map in metres, still present on the Telescope but no longer the primary OPD carrier. New code should read OPD from the source object directly.
* **relay()** — the method every core class implements to act on a source's EM field. Called automatically by the ``*`` operator; rarely called directly by users.
* **samplingTime** — AO loop period in seconds, used by the :class:`~OOPAO.Atmosphere.Atmosphere` to advance phase screens.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.Telescope import Telescope
   from OOPAO.Source import Source

   # Create a VLT-like 8-m telescope at 1 kHz
   tel = Telescope(resolution=240, diameter=8.0, samplingTime=1e-3,
                   centralObstruction=0.14)

   # Add four spider vanes (angles in degrees, thickness in metres)
   tel.apply_spiders(angle=[0, 90, 180, 270], thickness_spider=0.05)

   # Propagate a source through the telescope and WFS
   ngs = Source('H', magnitude=8)
   ngs * tel * wfs

   # Reset EM field and re-propagate cleanly
   ngs ** tel * wfs

   # Compute a PSF (zero-padding factor = 2)
   tel.computePSF(zeroPaddingFactor=2)

   # Print the current optical path
   tel.print_optical_path()

API reference
-------------

.. class:: Telescope(resolution, diameter, samplingTime=0.001, centralObstruction=0.0, fov=0.0, pupil=None, pupilReflectivity=1.0, display_optical_path=False)

   Defines the entrance pupil of the telescope.

   :param resolution: Number of pixels across the pupil diameter.
   :type resolution: int
   :param diameter: Physical diameter of the telescope in metres.
   :type diameter: float
   :param samplingTime: AO loop period in seconds (``1 / loop_frequency``). Used by the :class:`~OOPAO.Atmosphere.Atmosphere` to advance phase screens. Default ``0.001``.
   :type samplingTime: float
   :param centralObstruction: Fractional diameter of the central obstruction (0–1). Default ``0.0``.
   :type centralObstruction: float
   :param fov: Field of view in arcseconds. Required for off-axis multi-source simulations. Default ``0.0``.
   :type fov: float
   :param pupil: User-defined binary pupil mask (2-D array). Overrides the automatic circular mask. Default ``None``.
   :type pupil: numpy.ndarray or None
   :param pupilReflectivity: Uniform reflectivity scalar or 2-D map matching the pupil. Default ``1.0``.
   :type pupilReflectivity: float or numpy.ndarray
   :param display_optical_path: If ``True``, prints the optical path each time light is propagated to a WFS. Default ``False``.
   :type display_optical_path: bool

   **Key properties**

   .. attribute:: OPD
      :type: numpy.ndarray

      2-D optical path difference map in metres. Retained for **backward compatibility only** — the authoritative OPD is stored on the :class:`~OOPAO.Source.Source` object.

   .. attribute:: pupil
      :type: numpy.ndarray

      Binary 2-D pupil mask (1 inside aperture, 0 outside).

   .. attribute:: pupilLogical
      :type: numpy.ndarray

      1-D boolean index of valid (illuminated) pixels, used to vectorise operations over the pupil.

   .. attribute:: pixelArea
      :type: int

      Number of illuminated pixels (``pupil.sum()``).

   .. attribute:: pixelSize
      :type: float

      Physical size of one pixel in metres (``diameter / resolution``).

   .. attribute:: src
      :type: Source or Asterism

      Reference to the last source (or asterism) propagated through this telescope.

   .. attribute:: PSF
      :type: numpy.ndarray

      Last PSF computed by :meth:`computePSF`.

   **Methods**

   .. method:: relay(src)

      Core propagation method. Called automatically by ``src * tel``. Applies the telescope pupil and OPD to the source EM field.

      :param src: Source or Asterism being propagated.
      :type src: Source or Asterism

   .. method:: apply_spiders(angle, thickness_spider, offset_X=None, offset_Y=None)

      Carve spider vanes into the pupil mask.

      :param angle: List of vane angles in degrees. Length determines the number of spiders.
      :type angle: list[float]
      :param thickness_spider: Vane width in metres.
      :type thickness_spider: float
      :param offset_X: Per-vane X offset in metres. Default ``None`` (no offset).
      :type offset_X: list[float] or None
      :param offset_Y: Per-vane Y offset in metres. Default ``None`` (no offset).
      :type offset_Y: list[float] or None

   .. method:: computePSF(zeroPaddingFactor)

      Compute the PSF as the squared modulus of the Fourier transform of the complex pupil field. Result stored in :attr:`PSF`.

      :param zeroPaddingFactor: Zero-padding factor applied before the FFT. A value of 2 yields Shannon-sampled PSFs.
      :type zeroPaddingFactor: int

   .. method:: resetOPD()

      Set :attr:`OPD` to zero. Retained for backward compatibility.

   .. method:: print_optical_path()

      Print the ordered list of optical elements the source has been propagated through.

   **Operator summary**

   +--------------------+---------------------------------------------------------------+
   | Expression         | Effect                                                        |
   +====================+===============================================================+
   | ``ngs * tel``      | Call ``tel.relay(ngs)``; applies pupil to source EM field     |
   +--------------------+---------------------------------------------------------------+
   | ``ngs ** tel``     | Reset ``ngs`` EM field, then call ``tel.relay(ngs)``          |
   +--------------------+---------------------------------------------------------------+
   | ``ast * tel``      | Call ``tel.relay(src)`` for each source in the asterism       |
   +--------------------+---------------------------------------------------------------+
   | ``tel + atm``      | Couple atmosphere; ``tel.OPD`` follows ``atm.OPD`` (legacy)   |
   +--------------------+---------------------------------------------------------------+
   | ``tel - atm``      | Decouple atmosphere (legacy)                                  |
   +--------------------+---------------------------------------------------------------+
   | ``ngs * tel * wfs``| Full propagation chain: telescope then WFS                    |
   +--------------------+---------------------------------------------------------------+
