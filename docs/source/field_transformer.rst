Field Transformer
=================

.. currentmodule:: OOPAO.FieldTransformer

Overview
--------

:class:`FieldTransformer` applies geometric transformations (shift, rotation, anamorphosis, scaling) to the electromagnetic field (both phase and amplitude) of individual sources in an asterism. This is used to simulate super-resolution effects or model field-dependent aberrations in a Shack-Hartmann WFS.

Quick start
~~~~~~~~~~~

.. code-block:: python

   from OOPAO.FieldTransformer import FieldTransformer

   ft = FieldTransformer(
       src            = ast,
       shift_x        = [0.5, -0.5, 0.0],   # sub-pixel shifts per GS [pixels]
       shift_y        = [0.0,  0.0, 0.5],
       rotation_angle = [0.0,  0.0, 0.0],
   )

   ast * atm * tel * ft * wfs

API reference
-------------

.. class:: FieldTransformer(src, shift_x=None, shift_y=None, rotation_angle=None, anamorphosisAngle=None, tangentialScaling=None, radialScaling=None, remove_edge_effects=True)

   Geometric EM field transformer for individual sources.
   The class acts on both the :attrib:`intensity` and :attrib:`phase` of each source.

   :param src: Source or Asterism to transform.
   :type src: Source or Asterism
   :param shift_x: Per-source x shifts in pixels (sub-pixel shifts supported). Default ``None``.
   :type shift_x: list[float] or None
   :param shift_y: Per-source y shifts in pixels. Default ``None``.
   :type shift_y: list[float] or None
   :param rotation_angle: Per-source rotation angles in degrees. Default ``None``.
   :type rotation_angle: list[float] or None
   :param anamorphosisAngle: Per-source anamorphosis angle in degrees. Default ``None``.
   :type anamorphosisAngle: list[float] or None
   :param tangentialScaling: Per-source tangential scaling factors. Default ``None``.
   :type tangentialScaling: list[float] or None
   :param radialScaling: Per-source radial scaling factors. Default ``None``.
   :type radialScaling: list[float] or None
   :param remove_edge_effects: If ``True``, suppress edge artefacts near pupil borders after sub-pixel shifts. Default ``True``.
   :type remove_edge_effects: bool

   **Operator summary**

   +-------------------------------+-----------------------------------------------+
   | Expression                    | Effect                                        |
   +===============================+===============================================+
   | ``ast * tel * ft * wfs``      | Apply per-source EM field transforms          |
   +-------------------------------+-----------------------------------------------+
