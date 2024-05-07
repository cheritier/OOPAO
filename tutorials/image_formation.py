# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:21:18 2023

@author: cheritier
"""

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.tools.displayTools import displayMap, makeSquareAxes


#%% -----------------------     TELESCOPE   ----------------------------------

# create a Telescope object that defines the pupil mask and the resolution of the pupil
tel = Telescope(resolution          = 120,\
                diameter            = 8,\
                samplingTime        = 1/1000,\
                centralObstruction  = 0.1)
    
plt.figure()
plt.imshow(tel.pupil)

#%% -----------------------     NGS   ----------------------------------
# create a Source object that defines the ngs wavelength and flux properties
ngs = Source(optBand   = 'I',\
             magnitude = 10)
#%% 1 -- Compute the PSF of the system using the Telescope object method

# combine the NGS to the telescope using '*' operator:
ngs*tel

# zero padding factor for the FFT to prevent effects of aliasing. 
zeroPaddingFactor = 4

# compute the pixel size (in arcsec) associated to the zero-padding factor

pixel_scale_arcsec = 206265*ngs.wavelength/tel.D / zeroPaddingFactor

# compute the PSF
tel.computePSF(zeroPaddingFactor = zeroPaddingFactor)

# normalize PSF to its maximum value
PSF_normalized = tel.PSF.copy()/tel.PSF.max()

# crop the image to zoom on the core of the PSF
nCrop = int(tel.resolution * zeroPaddingFactor /3)
PSF_cropped = PSF_normalized[nCrop:-nCrop,nCrop:-nCrop]

# compute the field of view of the cropped PSF in arcesc
fov_cropped_arcsec = PSF_cropped.shape[0] * pixel_scale_arcsec

# display
plt.figure()

ax = plt.subplot(1,2,1)
im = ax.imshow(np.log(PSF_cropped),extent =[-fov_cropped_arcsec/2, fov_cropped_arcsec/2,-fov_cropped_arcsec/2,fov_cropped_arcsec/2] )
plt.colorbar(im,fraction=0.046, pad=0.04)
im.set_clim(vmin = -10, vmax = 0)
plt.title('Log-Scale PSF - Normalized')
plt.xlabel('arcsec')
plt.ylabel('arcsec')

ax = plt.subplot(1,2,2)

im = ax.imshow(tel.PSF,extent =[-fov_cropped_arcsec/2, fov_cropped_arcsec/2,-fov_cropped_arcsec/2,fov_cropped_arcsec/2] )
plt.colorbar(im,fraction=0.046, pad=0.04,label = 'Counts')
plt.title('Linear-Scale')
plt.xlabel('arcsec')
plt.ylabel('arcsec')

# make sure the PSF is properly normalized
print('{: ^30s}'.format('# of photons from the source:') + '{: ^30s}'.format(str(tel.src.fluxMap.sum())))
print('{: ^30s}'.format('Integral of the PSF')           + '{: ^30s}'.format(str(tel.src.fluxMap.sum())))

#%% -----------------------     ZERNIKE    ----------------------------------
nModes = 100                # number of zernike to compute

#create a zernike object
Z = Zernike(telObject = tel, J = nModes)

# compute the zernike polynomials associated to the telescope tel
Z.computeZernike(tel)

# access the Zernike modes: 
# 1) only the pixelsin the pupil 
print('{: ^50s}'.format('Z.modes shape: tel.pixelArea * nZernike') + str(Z.modes.shape))

# 2) the 2D maps of the modes 
print('{: ^50s}'.format('Z.modesFullRes shape : nPix * nPix * nZernike ') + str(Z.modesFullRes.shape))


plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(Z.modesFullRes[:,:,i])
# show the 2D map of all the modes
displayMap(Z.modesFullRes,norma = True)

# show the cross product matrix of the modes . Diagonal matrix  =  orthogonal basis of the modes. Non diagonal matrix = Basis non orthogonal (for instance in presence of a central obstruction)
# the cross product matrix has to be normalized by the number of pixel in the pupil (tel.PixelArea)
plt.figure()
plt.imshow(Z.modes.T@Z.modes/tel.pixelArea)
plt.title('Cross Product Matrix')
#%% PSF corresponding to a Zernike

# Amplitude of the mode (in m RMS)
amp = 100e-9 

# choose a number of zernike polynomial
N= 80

# set the Optical Path Difference (OPD) of the telescope using a the Nth zernike
tel.OPD = np.squeeze(Z.modesFullRes[:,:,N])*amp

# compute the corresponding PSF
tel.computePSF(zeroPaddingFactor=8)

# normlize PSF to its maximum value
PSF_normalized = tel.PSF.copy()/tel.PSF.max()

# crop the image to zoom on the core of the PSF
nCrop = int(tel.resolution * zeroPaddingFactor /3)
PSF_cropped = PSF_normalized[nCrop:-nCrop,nCrop:-nCrop]

# compute the field of view of the cropped PSF in arcesc
fov_cropped_arcsec = PSF_cropped.shape[0] * pixel_scale_arcsec

# display
plt.figure()

ax = plt.subplot(1,2,1)
im = ax.imshow(np.log(PSF_cropped),extent =[-fov_cropped_arcsec/2, fov_cropped_arcsec/2,-fov_cropped_arcsec/2,fov_cropped_arcsec/2] )
plt.colorbar(im,fraction=0.046, label='normalized counts')
im.set_clim(vmin = -10, vmax = 0)
plt.title('log-scale PSF - normalized')
plt.xlabel('arcsec')
plt.ylabel('arcsec')

ax = plt.subplot(1,2,2)

im = ax.imshow(tel.PSF,extent =[-fov_cropped_arcsec/2, fov_cropped_arcsec/2,-fov_cropped_arcsec/2,fov_cropped_arcsec/2] )
plt.colorbar(im,fraction=0.046, label = 'Counts')
plt.title('linear-scale')
plt.xlabel('arcsec')
plt.ylabel('arcsec')

# make sure the PSF is properly normalized
print('{: ^30s}'.format('# of photons from the source:') + '{: ^30s}'.format(str(tel.src.fluxMap.sum())))
print('{: ^30s}'.format('Integral of the PSF')           + '{: ^30s}'.format(str(tel.src.fluxMap.sum())))

#%% -----------------------     MULTIPLE MODES AT THE SAME TIME   ----------------------------------

# Amplitude of the mode (in m RMS)
amp = np.zeros(nModes)

# apply an amplitude on a few modes
amp[0]  = 40e-9
amp[10] = 160e-9
amp[50] = 20e-9
amp[80] = 30e-9

# set the Optical Path Difference (OPD) of the telescope using a the combination of zernike (notice the matrix multiplication opertor @)
tel.OPD = np.squeeze(Z.modesFullRes@amp)

# compute the corresponding PSF
tel.computePSF(zeroPaddingFactor=8)

# normlize PSF to its maximum value
PSF_normalized = tel.PSF.copy()/tel.PSF.max()

# crop the image to get the core of the PSF
nCrop = int(tel.resolution * zeroPaddingFactor /3)

PSF_cropped = PSF_normalized[nCrop:-nCrop,nCrop:-nCrop]

# display
plt.figure()
plt.subplot(1,2,1)
plt.imshow(tel.OPD)
plt.title('Aberrated Phase Screen' )
plt.colorbar(fraction=0.046,label = '[m]')

ax1 = plt.subplot(1,2,2)
im1 = ax1.imshow(np.log(PSF_normalized),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.colorbar(im1,fraction=0.046,label='normalized counts')
im1.set_clim(vmin = -10, vmax = 0)
plt.title('log-scale PSF')
plt.xlabel('[arcsec]')
plt.ylabel('[arcsec]')


#%% -----------------------     MODAL DECOMPOSITION OF THE GENERATED PHASE SCREEN   ----------------------------------


# compute the pseudo inverse of the zernike polynomials (least square minimization)
Z_inv = np.linalg.pinv(Z.modes)

# get the modal coefficients corresponding to previous phase scrren in [m]

    # 1)reshape OPD of the telescope in 1D and truncated to the pupil area
OPD_atm =tel.OPD[np.where(tel.pupil==1)]

    # 2) multiply using Z_inv
coef_modal = Z_inv@OPD_atm


plt.figure()
plt.subplot(1,2,1)
plt.imshow(tel.OPD)
plt.title(' Aberrated phase screen [m]')

plt.subplot(1,3,3)
plt.plot(coef_modal)
plt.xlabel('Zernike Modes')
plt.ylabel('[m]')
plt.title('Modal decomposition of the phase screen')
makeSquareAxes(plt.gca())


#%% -----------------------     MODAL DECOMPOSITION OF THE ATMOSPHERE PHASE SCREEN   ----------------------------------

# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = 0.15,\
               L0            = 25,\
               windSpeed     = [10],\
               fractionalR0  = [1],\
               windDirection = [10],\
               altitude      = [0])

# initialize atmosphere
atm.initializeAtmosphere(tel)

plt.figure()
plt.imshow(atm.OPD)
tel+atm
tel.computePSF(zeroPaddingFactor=8)
plt.figure()
plt.imshow((np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.xlabel('Angular Resolution [arcsec]')
plt.ylabel('Angular Resolution [arcsec]')




# compute the pseudo inverse of the zernike polynomials (least square minimization)
Z_inv = np.linalg.pinv(Z.modes)

# get the modal coefficients corresponding to a given phase screen of the atmosphere in [m]

    # 1)reshape OPD of the atmosphere in 1D and truncated to the pupil area
OPD_atm =atm.OPD[np.where(tel.pupil==1)]

    # 2) multiply using Z_inv
coef_atm = Z_inv@OPD_atm


# reconstruct the atm.OPD using the coef_atm
OPD_atm_rec =  tel.OPD = np.squeeze(Z.modesFullRes@coef_atm)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(atm.OPD)
plt.title('Atmosphere phase screen [m]')
plt.subplot(1,3,2)
plt.imshow(OPD_atm_rec)
plt.title('Fitted by modal basis [m]')

plt.subplot(1,3,3)
plt.plot(1e9*coef_atm)
plt.xlabel('Zernike Modes')
plt.ylabel('[nm]')
plt.title('Modal decomposition of the phase screen')
makeSquareAxes(plt.gca())