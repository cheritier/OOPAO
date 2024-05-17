# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:23:49 2023

@author: cheritier
"""

import time
import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Pyramid import Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
# %% -----------------------     read parameter file   ----------------------------------
from parameterFile_ELT_SCAO_K_Band_3000_KL import initializeParameterFile

param = initializeParameterFile()

# %%
plt.ion()

#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.M1_model.make_ELT_pupil import generateEeltPupilReflectivity

# create the pupil of the ELT
tmp = 1+0.*np.random.rand(798)

M1_pupil_reflectivity = generateEeltPupilReflectivity(refl = tmp,\
                                          npt       = param['resolution'],\
                                          dspider   = param['spiderDiameter'],\
                                          i0        = param['m1_center'][0]+param['m1_shiftX'] ,\
                                          j0        = param['m1_center'][1]+param['m1_shiftY'] ,\
                                          pixscale  = param['pixelSize'],\
                                          gap       = param['gapSize'],\
                                          rotdegree = param['m1_rotationAngle'],\
                                          softGap   = True)

# extract the valid pixels
M1_pupil = M1_pupil_reflectivity>0

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                pupil               = M1_pupil,\
                pupilReflectivity   = M1_pupil_reflectivity)

plt.figure()
plt.imshow(tel.pupil)



#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)

param['m4_filename'] = 'D:/ESO_2020_2022/local/diskb/cheritier/psim/data_calibration/cropped_M4IF.fits'
# if no coordonates specified, create a cartesian dm
dm=DeformableMirror(telescope           = tel,\
                    nSubap              = 20,       # only used if default case is considered ( Fried geometry). Not used if using user-defined coordinates / M4 IFs. 
                    mechCoupling        = 0.45,     # only used if default case is considered ( gaussian IFs).
                    misReg              = misReg,   # mis-reg for the IFs
                    M4_param            = param,    # Necessary to read M4 IFs (using param['m4_filename'])
                    floating_precision  = 32)       # using float32 precision for tutorial

plt.figure()
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

tel*dm