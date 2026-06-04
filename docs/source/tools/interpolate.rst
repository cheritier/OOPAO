Geometrical Interpolation
=========================

.. currentmodule:: OOPAO.tools.interpolateGeometricalTransformation

Overview
--------

This module provides image transformation helpers used internally by :class:`~OOPAO.DeformableMirror.DeformableMirror` and the mis-registration algorithms to warp influence functions and OPD maps.

Selected API reference
----------------------

.. function:: interpolate_image(image, transformation_matrix, output_shape=None)

   Apply a scikit-image geometric transformation matrix to a 2-D image.

   :param image: Input 2-D image.
   :type image: numpy.ndarray
   :param transformation_matrix: scikit-image ``AffineTransform`` or ``ProjectiveTransform`` object.
   :type transformation_matrix: skimage.transform.GeometricTransform
   :param output_shape: Output array shape. Default ``None`` (same as input).
   :type output_shape: tuple or None
   :returns: Transformed image.
   :rtype: numpy.ndarray

.. function:: interpolate_cube(cube, transformation_matrix, output_shape=None, n_jobs=1)

   Apply a transformation to each slice of a 3-D cube (parallelised with joblib).

   :param cube: Input array of shape ``(n_pix, n_pix, n_slices)``.
   :type cube: numpy.ndarray
   :param transformation_matrix: Transformation to apply.
   :type transformation_matrix: skimage.transform.GeometricTransform
   :param output_shape: Output shape per slice. Default ``None``.
   :type output_shape: tuple or None
   :param n_jobs: Number of parallel jobs. Default ``1``.
   :type n_jobs: int

.. function:: globalTransformation(image, misRegistration, pixelSize)

   Apply a full :class:`~OOPAO.MisRegistration.MisRegistration` (rotation + shift + scaling + anamorphosis) to an image.

   :param image: Input 2-D image.
   :type image: numpy.ndarray
   :param misRegistration: Transformation parameters.
   :type misRegistration: MisRegistration
   :param pixelSize: Physical pixel size in metres (used to convert shift from metres to pixels).
   :type pixelSize: float
