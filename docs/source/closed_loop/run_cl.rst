run_cl
======

.. currentmodule:: OOPAO.closed_loop.run_cl

Overview
--------

:func:`run_cl` executes a standard single-stage AO closed-loop simulation. It applies noise settings from the parameter file, builds the mis-registration-aware reconstructor, then iterates the loop for ``param['nLoop']`` frames, recording residual variance and Strehl ratio.

API reference
-------------

.. function:: run_cl(param, obj)

   Run a single-stage AO closed loop.

   :param param: Parameter dictionary. Expected keys include:

      * ``'nLoop'`` — number of loop iterations.
      * ``'gainCL'`` — integral controller gain.
      * ``'nPhotonPerSubaperture'`` — NGS flux.
      * ``'photonNoise'`` — bool, enable photon noise.
      * ``'readoutNoise'`` — float, readout noise in electrons.
      * ``'nModes'`` — number of controlled modes.
      * ``'nSubaperture'`` — number of WFS subapertures.
   :type param: dict
   :param obj: AO system object with attributes ``tel``, ``dm``, ``wfs``, ``ngs``, ``atm``, ``calib``, ``M2C_cl``, and ``param``.
   :type obj: object

   :returns: Dictionary containing ``'SR'`` (Strehl ratio per frame), ``'var'`` (residual wavefront variance), and optionally saved interaction matrices.
   :rtype: dict

   **Example**

   .. code-block:: python

      from OOPAO.closed_loop.run_cl import run_cl

      out = run_cl(param, ao_obj)

      import matplotlib.pyplot as plt
      plt.plot(out['SR'])
      plt.xlabel('Frame')
      plt.ylabel('Strehl Ratio')
