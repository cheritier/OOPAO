Closed-Loop Variants
====================

OOPAO provides several specialised closed-loop runners for specific simulation scenarios. All follow the same interface as :func:`~OOPAO.closed_loop.run_cl.run_cl` (``run_cl(param, obj)``).

.. list-table:: Available closed-loop runners
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Description
   * - ``run_cl``
     - Standard single-stage integral controller loop.
   * - ``run_cl_two_stages``
     - Two-stage AO loop (e.g. SCEXAO + SAXO). Propagates through a first-stage DM then a second-stage DM/WFS.
   * - ``run_cl_two_stages_atm_change``
     - Two-stage loop with on-the-fly atmospheric parameter changes.
   * - ``run_cl_first_stage``
     - Runs only the first stage of a two-stage system, saving residuals for subsequent use.
   * - ``run_cl_from_phase_screens``
     - Replay a pre-recorded sequence of phase screens through a closed loop.
   * - ``run_cl_long_push_pull``
     - Measures the system non-linearity by applying a long push-pull sequence during closed-loop.
   * - ``run_cl_sinusoidal_modulation``
     - Sinusoidal modulation of selected modes during closed-loop, for transfer function measurements.

All runners are importable from ``OOPAO.closed_loop``:

.. code-block:: python

   from OOPAO.closed_loop.run_cl_two_stages import run_cl_two_stages
   out = run_cl_two_stages(param, ao_obj)
