# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:01:52 2024

@author: cheritier
"""

import matplotlib.pyplot as plt
import numpy as np
plt.ion()



#%%  -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope
# create the Telescope object
tel = Telescope(resolution           = 200,                          # resolution of the telescope in [pix]
                diameter             = 8,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.1,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 30 )                                     # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

from OOPAO.Source import Source
ngs = Source(optBand   = 'I', magnitude = 0)               



#%% -----------------------     NGS   ----------------------------------
from OOPAO.Asterism import Asterism

conversion_constant = (180/np.pi)*3600
fov = conversion_constant*ngs.wavelength/tel.D *tel.resolution/2

src_obj = []
n_source = 2

x_science = np.linspace(-10,10,n_source,endpoint=True)
for i in range(len(x_science)):
    for j in range(len(x_science)):
        r  = np.sqrt(x_science[i]**2 + x_science[j]**2)
        th = np.rad2deg(np.arctan2(x_science[i],x_science[j]))
        ngs_tmp=Source(optBand   = 'I',\
        magnitude = 0,coordinates=[r,th])               
        src_obj.append(ngs_tmp)
        
ast = Asterism(src_obj)

ast*tel

ast.display_asterism()

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

# %%

atm.display_atm_layers()

#%%

from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Detector import Detector

cam = Detector()
cam.psf_sampling = 4 # 4 pixel / FXHM for the PSF
cam.resolution = 256 # cropping the PSF to the central part

# create a Deformable Mirror to simulate an AO correction (Fitting error)
dm = DeformableMirror(telescope = tel, nSubap = 20)

# update the NGS coordinates (where the correction will be applied)
ngs.coordinates = [0,0]
atm*ngs*tel
# project the OPD in the NGS direction on the DM
dm.coefs = -np.linalg.pinv(dm.modes)@tel.OPD.reshape(tel.resolution**2)

# re-propagate through the DM to get the "AO corrected" PSF
atm*ast*tel*dm*cam

#%%
# Display the PSFs at their exact location in the field using the tel.PSF property
plt.figure()
plt.imshow(np.log10(tel.PSF+1e-100), extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1] ])# adding 1e-100 for display purpose
plt.clim([np.log10(tel.PSF).max()-4,np.log10(tel.PSF).max()])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.grid()

# Display the PSFs in a grid zoomed on the PSF core using the tel.PSF_list property
plt.figure()
for i in range(ast.n_source):
    plt.subplot(n_source,n_source,i+1)
    plt.imshow(np.log10(tel.PSF_list[i]))# adding 1e-12 for display purpose
    plt.clim([np.log10(tel.PSF_list).max()-4,np.log10(tel.PSF_list).max()])
    plt.axis('off')
    plt.title('NGS @ '+str(np.round(ast.src[i].coordinates[0],1)) +','+str(np.round(ast.src[i].coordinates[1],1)))
