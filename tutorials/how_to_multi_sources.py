# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:01:52 2024

@author: cheritier


Tutorial Description — Shack-Hartmann Wave-Front Sensing in OOPAO

This tutorial provides a focused walkthrough of Shack-Hartmann (SH) wave-front sensing in OOPAO, 
illustrating how to configure, sample, visualize, and calibrate a SH sensor within a full optical chain.

Starting from a telescope, guide star, and optional atmosphere, the script introduces the SH WFS as a modular optical element, showing how it receives the propagated NGS signal via the * operator.
It compares several SH sampling regimes:
    - Shannon/2
    - Shannon
    - custom pixel scales
    
Users can inspect raw camera frames, slope maps, valid-subaperture masks, and cubes of individual subaperture spots.

The tutorial highlights practical SH utilities:
    - enabling/disabling photon-noise
    - switching between diffractive and geometric (gradient-only) sensing
    - applying Gaussian centroid-weighting maps to optimize slope estimation. 

A KL modal basis is generated and used to build a modal interaction matrix with the SH sensor.

The calibrated interaction matrix is then used to set up a closed-loop environment, where the SH slopes drive the DM via a modal reconstructor.

"""

import OOPAO
import tomoAO
import time
import matplotlib.pyplot as plt
import numpy as np
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap

# %%
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
                fov                  = 40 )                                     # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

# # Apply spiders to the telescope pupil
# thickness_spider    = 0.05                                                       # thickness of the spiders in m
# angle               = [45, 135, 225, 315]                                        # angle in degrees for each spider
# offset_Y            = [-0.2, -0.2, 0.2, 0.2]                                     # shift offsets for each spider
# offset_X            = None

# tel.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

# # display current pupil
# plt.figure()
# plt.imshow(tel.pupil)

#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

from OOPAO.Asterism import Asterism

n_lgs = 4
lgs_zenith = [20]*n_lgs
lgs_azimuth = np.linspace(0,360,n_lgs,endpoint=False)
lgs_asterism = Asterism([Source(optBand='Na', magnitude=0, coordinates=[lgs_zenith[kLgs], lgs_azimuth[kLgs]], altitude=90e3) for kLgs in range(n_lgs)])
lgs_asterism**tel
#%% -----------------------     ATMOSPHERE   ----------------------------------
from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.15,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers in [m]


# initialize atmosphere with current Telescope
atm.initializeAtmosphere(tel)

# The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
atm.update()

# display the atm.OPD = resulting OPD 
plt.figure()
plt.imshow(np.hstack(atm.OPD)*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()
#%%

lgs_asterism**atm

lgs_asterism.print_optical_path()

fig, axes = plt.subplots(2, n_lgs, figsize=(20, 8))


for i in range(n_lgs):
    im = axes[0, i].imshow(lgs_asterism.OPD[i])
    axes[0, i].axis('off')
    axes[0, i].set_title(f'Source {i+1} OPD')
    fig.colorbar(im, ax=axes[0, i])

    im = axes[1, i].imshow(lgs_asterism.OPD_no_pupil[i])
    axes[1, i].axis('off')
    axes[1, i].set_title(f'Source {i+1} OPD_no_pupil')
    fig.colorbar(im, ax=axes[1, i])



lgs_asterism*tel

lgs_asterism.print_optical_path()

fig, axes = plt.subplots(2, n_lgs, figsize=(20, 8))

for i in range(n_lgs):
    im = axes[0, i].imshow(lgs_asterism.OPD[i])
    axes[0, i].axis('off')
    axes[0, i].set_title(f'Source {i+1} OPD')
    fig.colorbar(im, ax=axes[0, i])

    im = axes[1, i].imshow(lgs_asterism.OPD_no_pupil[i])
    axes[1, i].axis('off')
    axes[1, i].set_title(f'Source {i+1} OPD_no_pupil')
    fig.colorbar(im, ax=axes[1, i])




