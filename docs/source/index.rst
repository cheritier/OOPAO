OOPAO Documentation
===================

**Object Oriented Python Adaptive Optics**

OOPAO is a Python end-to-end Adaptive Optics (AO) simulation framework inspired by the MATLAB toolkit OOMAO. It models a full optical path by propagating light from a source through a chain of physical objects — Atmosphere, Telescope, Deformable Mirror, Wavefront Sensor — using Python's ``*`` and ``**`` operators. The framework was developed at ESO and Laboratoire d'Astrophysique de Marseille, and is designed for AO instrument modelling, interaction matrix generation, and closed-loop simulation.

.. code-block:: python

   # Minimal closed-loop snippet
   ngs = Source('H', magnitude=8)
   tel = Telescope(resolution=240, diameter=8.0, samplingTime=1e-3)
   atm = Atmosphere(tel, r0=0.15, L0=25, windSpeed=[10], fractionalR0=[1],
                    windDirection=[0], altitude=[0])
   dm  = DeformableMirror(tel, nSubap=20)
   wfs = Pyramid(20, tel, modulation=3, lightRatio=0.5)

   for i in range(nLoop):
       atm.update()
       ngs ** atm * tel * dm * wfs
       dm.coefs -= gain * M2C @ wfs.signal

.. toctree::
   :maxdepth: 2
   :caption: Core Objects

   telescope
   source
   atmosphere
   deformable_mirror
   asterism

.. toctree::
   :maxdepth: 2
   :caption: Optical Elements

   zernike
   ncpa
   opd_map
   spatial_filter
   field_transformer

.. toctree::
   :maxdepth: 2
   :caption: Wave-front Sensors

   pyramid
   shack_hartmann
   bio_edge

.. toctree::
   :maxdepth: 2
   :caption: Focal-plane Sensors

   gain_sensing_camera
   lift

.. toctree::
   :maxdepth: 2
   :caption: Calibration

   calibration/calibration_vault
   calibration/interaction_matrix
   calibration/kl_basis
   sprint

.. toctree::
   :maxdepth: 2
   :caption: Mis-Registration

   mis_registration

.. toctree::
   :maxdepth: 2
   :caption: Closed-Loop Utilities

   closed_loop/index

.. toctree::
   :maxdepth: 2
   :caption: Tools & Utilities

   tools/index
   phase_stats
