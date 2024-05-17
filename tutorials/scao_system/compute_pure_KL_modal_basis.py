"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
# %% -----------------------     read parameter file   ----------------------------------
from parameter_files.parameterFile_VLT_I_Band_SHWFS import initializeParameterFile

param = initializeParameterFile()

# %%
plt.ion()

#%% -----------------------     TELESCOPE   ----------------------------------


tel.resolutionolution = 20

# create the Telescope object
tel = Telescope(tel.resolutionolution          = tel.resolutionolution,\
                diameter            = 4,\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])
#%% REDEFINITION OF PUPIL TO AVOID EDGE EFFECTS
D           = tel.tel.resolutionolution+1
x           = np.linspace(-tel.tel.resolutionolution/2,tel.tel.resolutionolution/2,tel.tel.resolutionolution)
xx,yy       = np.meshgrid(x,x)
circle      = xx**2+yy**2
obs         = circle>=(tel.centralObstruction*D/2)**2
pupil       = circle<(D/2-n_extra_pixel)**2       
tel.pupil = pupil*obs
plt.figure(), plt.imshow(tel.pupil)
#%% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

tel.computePSF(zeroPaddingFactor = 6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,3])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()


src = Source(optBand   = 'K',\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
# src*tel
#%% -----------------------     ATMOSPHERE   ----------------------------------

# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = 0.15,\
               L0            = param['L0'],\
               windSpeed     = [10],\
               fractionalR0  = [1],\
               windDirection = [0],\
               altitude      = [0])
# initialize atmosphere
atm.initializeAtmosphere(tel)

atm.update()

plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()
#%%



tel.pupil[:]=1

tel.pupil = tel.pupil



Z = np.eye((tel.resolution*tel.resolution))
Z = Z.reshape(tel.resolution*tel.resolution,tel.resolution*tel.resolution)



from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis



dmVirtual = DeformableMirror(telescope = tel,
                             nSubap = 21,modes = Z)


#%%


M2C = compute_KL_basis(tel,atm,dmVirtual,lim=1e-7,remove_piston=False)

#%%
dmVirtual.coefs = M2C

displayMap(dmVirtual.OPD)



