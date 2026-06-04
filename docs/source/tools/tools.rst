tools
=====

.. currentmodule:: OOPAO.tools.tools

Overview
--------

The ``tools`` module contains general-purpose utility functions used throughout OOPAO — file I/O, array manipulation, photon statistics, and error handling.

Selected API reference
----------------------

.. function:: createFolder(path)

   Create a directory (and all parent directories) if it does not exist.

   :param path: Directory path.
   :type path: str

.. function:: read_fits(filename)

   Read a FITS file and return the data array.

   :param filename: Path to the FITS file.
   :type filename: str
   :returns: Data array.
   :rtype: numpy.ndarray

.. function:: crop(imageArray, size, axis, maximum=0)

   Extract a centred sub-image from a 2-D or 3-D array.

   :param imageArray: Input array.
   :type imageArray: numpy.ndarray
   :param size: Output size in pixels.
   :type size: int
   :param axis: Axis along which to crop (for 3-D arrays).
   :type axis: int
   :param maximum: If ``1``, centre on the array maximum instead of the geometric centre. Default ``0``.
   :type maximum: int

.. function:: bin_ndarray(ndarray, new_shape, operation='sum')

   Bin a 2-D or 3-D NumPy array to a new shape by summing or averaging.

   :param ndarray: Input array.
   :type ndarray: numpy.ndarray
   :param new_shape: Target shape after binning.
   :type new_shape: tuple
   :param operation: ``'sum'`` (default) or ``'mean'``.
   :type operation: str

.. function:: set_binning(frame, binning)

   Apply integer pixel binning to a 2-D frame.

   :param frame: Input 2-D array.
   :type frame: numpy.ndarray
   :param binning: Binning factor.
   :type binning: int
   :returns: Binned array.
   :rtype: numpy.ndarray

.. function:: gaussian_2D(size, sigma, center=None)

   Generate a 2-D Gaussian kernel.

   :param size: Array side length in pixels.
   :type size: int
   :param sigma: Gaussian sigma in pixels.
   :type sigma: float
   :param center: ``(row, col)`` centre. Default ``None`` (geometric centre).
   :type center: tuple or None

.. exception:: OopaoError

   Custom exception raised for OOPAO-specific configuration and propagation errors.

.. function:: warning(message)

   Print a formatted OOPAO warning to stdout.

   :param message: Warning text.
   :type message: str
