# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:03:30 2025

@author: cheritier
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap

#%%
plt.ion()
# number of subaperture for the WFS
n_subaperture = 20


# %%-----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 6*n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 8,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.1,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 0 )                                     # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets



#% pad to avoid edge effects
tel.pad(tel.resolution//2)

plt.figure()
plt.imshow(tel.pupil)


#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand     = 'I',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
ngs*tel

# create the Scientific Target object located at 10 arcsec from the  ngs
src = Source(optBand     = 'K',           # Optical band (see photometry.py)
             magnitude   = 8,              # Source Magnitude
             coordinates = [0,0])        # Source coordinated [arcsec,deg]

# combine the SRC to the telescope using '*'
src*tel

# check that the ngs and tel.src objects are the same
tel.src.print_properties()


#%% -----------------------     ATMOSPHERE   ----------------------------------
from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.05,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers in [m]


# initialize atmosphere with current Telescope
atm.initializeAtmosphere(tel,compute_covariance=False)
#%% Detector 
from OOPAO.Detector import Detector

# define a detector with its properties (see Detector class for further documentation)
cam = Detector(integrationTime = tel.samplingTime,      # integration time of the detector
               photonNoise     = False,                  # enable photon noise
               readoutNoise    = 0,                     # readout of the detector in [e-/pixel]
               QE              = 1,                   # quantum efficiency
               psf_sampling    = 6,                     # sampling for the PSF computation 2 = Shannon sampling
               binning         = 1)                     # Binning factor of the PSF


#%% Spatial Filter
from OOPAO.SpatialFilter import SpatialFilter
plt.close('all')

sf = SpatialFilter(telescope = tel,
                   shape = "circular",
                   diameter = 20)

# set the spatial filter with the correct sampling (convolution done in Fourier space)
sf.set_spatial_filter(cam.psf_sampling)

# display the mask
plt.figure()
plt.imshow(np.abs(sf.mask))

# PSF obtained with the spatial filter 
src**atm*tel*sf*cam

plt.figure()
plt.subplot(131)
plt.imshow(src.amplitude_filtered)
plt.title('Pupil Plane Intensity')
plt.colorbar()

plt.subplot(132)
plt.imshow(src.phase_filtered)
plt.title('Phase (Wrapped) [rad]')
plt.colorbar()

plt.subplot(133)
plt.imshow(cam.frame)
plt.title('Focal Plane Intensity: ' + str(int(cam.frame.sum())) + ' counts')


sf = SpatialFilter(telescope = tel,
                   shape = "circular",
                   diameter = tel.resolution)

# set the spatial filter with the correct sampling (convolution done in Fourier space)
sf.set_spatial_filter(cam.psf_sampling)

# display the mask
plt.figure()
plt.imshow(np.abs(sf.mask))

# PSF obtained with the spatial filter 
src**atm*tel*sf*cam

plt.figure()
plt.subplot(131)
plt.imshow(src.amplitude_filtered)
plt.title('Pupil Plane Intensity')
plt.colorbar()

plt.subplot(132)
plt.imshow(src.phase_filtered)
plt.title('Phase (Wrapped) [rad]')
plt.colorbar()

plt.subplot(133)
plt.imshow(cam.frame)
plt.title('Focal Plane Intensity: ' + str(int(cam.frame.sum())) + ' counts')


# PSF obtained without the spatial filter 
src**atm*tel*cam
plt.figure()
plt.subplot(131)
plt.imshow(np.sqrt(src.fluxMap))
plt.colorbar()
plt.title('Pupil Plane Intensity')

plt.subplot(132)
plt.imshow(src.phase)
plt.title('Phase [rad]')
plt.colorbar()

plt.subplot(133)
plt.imshow(cam.frame)
plt.title('Focal Plane Intensity: ' + str(int(cam.frame.sum())) + ' counts')

#%% Application with Shack Hartmann

from OOPAO.ShackHartmann import ShackHartmann

# make sure that the ngs is propagated to the wfs
ngs**tel

# shannon/2 sampled Shack-Hartmann spots
wfs = ShackHartmann(nSubap = n_subaperture,
                    telescope = tel,
                    lightRatio = 0.5,
                    shannon_sampling = True)

# create the spatial filter
sf = SpatialFilter(telescope = tel,
                   shape = "circular",
                   diameter = 20)

# In presence of Spatial Filter
ngs**atm*tel*sf*wfs

plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('Spatially Filtered Shack Hartmann')

# No more Spatial Filter
ngs**atm*tel*wfs
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('Regular Shack Hartmann')

#%% Application with Pyramid

from OOPAO.Pyramid import Pyramid

# make sure that the ngs is propagated to the wfs
ngs**tel

wfs = Pyramid(nSubap            = n_subaperture,                # number of subaperture = number of pixel accros the pupil diameter
              telescope         = tel,                          # telescope object
              lightRatio        = 0.,                          # flux threshold to select valid sub-subaperture
              modulation        = 3,                            # Tip tilt modulation radius
              binning           = 1,                            # binning factor (applied only on the )
              n_pix_separation  = 2,                            # number of pixel separating the different pupils
              n_pix_edge        = 1,                            # number of pixel on the edges of the pupils
              psfCentering      = True,
              postProcessing    = 'slopesMaps_incidence_flux')  # slopesMap_incidence_flux, fullFrame_incidence_flux (see documentation)

# create the spatial filter
sf = SpatialFilter(telescope = tel,
                   shape = "circular",
                   diameter = 8)
# adjust the sampling to match the one of the Pyramid WFS
sf.set_spatial_filter(wfs.zeroPaddingFactor)

# assign it to the Pyramid WFS
wfs.spatialFilter = sf.mask
# propagate
ngs**tel*wfs

plt.figure()
plt.imshow(wfs.cam.frame)

# Remove it from the Pyramid WFS
wfs.spatialFilter = None
# propagate
ngs**tel*wfs

plt.figure()
plt.imshow(wfs.cam.frame)

