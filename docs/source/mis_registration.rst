MisRegistration
===============

.. currentmodule:: OOPAO.MisRegistration

Overview
--------

A :class:`MisRegistration` object encodes the geometric mis-alignment between a DM and a WFS. It can be applied to a :class:`~OOPAO.DeformableMirror.DeformableMirror` at construction time to warp the influence functions, or used by the SPRINT algorithm to estimate and correct registration errors.

Parameters
~~~~~~~~~~

Six independent degrees of freedom are modelled:

* **rotationAngle** — rigid rotation in degrees.
* **shiftX / shiftY** — lateral displacement in metres.
* **anamorphosisAngle** — differential anamorphosis rotation in degrees.
* **radialScaling** — radial magnification as a fractional offset (0 = no scaling).
* **tangentialScaling** — tangential magnification as a fractional offset.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.MisRegistration import MisRegistration

   misReg = MisRegistration()
   misReg.rotationAngle = 0.5      # 0.5 degree rotation
   misReg.shiftX = 0.01            # 1 cm lateral shift

   # Apply to a DM
   dm = DeformableMirror(tel, nSubap=20, misReg=misReg)

   # Combine two mis-registrations
   total = misReg + another_misReg

API reference
-------------

.. class:: MisRegistration(param=None)

   Geometric mis-registration descriptor.

   :param param: Initialisation source. Can be:

      * ``None`` — all parameters set to zero (default).
      * ``dict`` — keys ``'rotationAngle'``, ``'shiftX'``, ``'shiftY'``, ``'anamorphosisAngle'``, ``'tangentialScaling'``, ``'radialScaling'``.
      * Another :class:`MisRegistration` object — copies all parameters.

   :type param: None, dict, or MisRegistration

   **Attributes**

   .. attribute:: rotationAngle
      :type: float

      Rotation angle in degrees.

   .. attribute:: shiftX
      :type: float

      X shift in metres.

   .. attribute:: shiftY
      :type: float

      Y shift in metres.

   .. attribute:: anamorphosisAngle
      :type: float

      Anamorphosis angle in degrees.

   .. attribute:: radialScaling
      :type: float

      Radial magnification offset (0.0 = no scaling).

   .. attribute:: tangentialScaling
      :type: float

      Tangential magnification offset (0.0 = no scaling).

   .. attribute:: misRegName
      :type: str

      Auto-generated string identifier encoding all parameter values (used as a folder/file name in calibration workflows).

   **Operators**

   ``+`` and ``-`` are overloaded to add or subtract two :class:`MisRegistration` objects component-wise.
