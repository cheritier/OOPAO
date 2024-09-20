# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:15:14 2020

@author: cheritie
"""

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.tools.displayTools import displayMap, makeSquareAxes
# %% -----------------------     read parameter file   ----------------------------------
from parameter_files.parameterFile_VLT_I_Band_PWFS import initializeParameterFile

param = initializeParameterFile()

#%% -----------------------     TELESCOPE   ----------------------------------

# create a Telescope object that defines the pupil mask and the resolution of the pupil
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = 1,\
                centralObstruction  = param['centralObstruction'])

#%% -----------------------     NGS   ----------------------------------
# create a Source object that defines the ngs wavelength and flux properties
ngs = Source(optBand   = 'I',\
             magnitude = 10)

# combine the NGS to the telescope using '*' operator:
ngs*tel
# compute the associated diffraction limited PSF
zeroPaddingFactor = 8   # zero padding factor for the FFT to prevent the effects of aliasing
tel.computePSF(zeroPaddingFactor = zeroPaddingFactor)

# normlize PSF to its maximum value
PSF_normalized = tel.PSF.copy()/tel.PSF.max()

# crop the image to get the core of the PSF
nCrop = int(tel.resolution * zeroPaddingFactor /3)

PSF_cropped = PSF_normalized[nCrop:-nCrop,nCrop:-nCrop]

# display
plt.figure()
ax = plt.subplot(1,1,1)
im = ax.imshow(np.log(PSF_cropped))
plt.colorbar(im)
im.set_clim(vmin = -10, vmax = 0)
plt.title('Diffraction Limited PSF')

#%%


N       = zeroPaddingFactor * tel.resolution            
center = N//2           
norma   = (tel.src.fluxMap.sum())

# zeroPadded support for the FFT
supportPadded = np.zeros([N,N],dtype='complex')
supportPadded [center-tel.resolution//2:center+tel.resolution//2,center-tel.resolution//2:center+tel.resolution//2] = tel.pupil*tel.pupilReflectivity*np.exp(1j*tel.src.phase)
# phasor to center the FFT on the 4 central pixels
[xx,yy]                         = np.meshgrid(np.linspace(0,N-1,N),np.linspace(0,N-1,N))

tel.phasor                     = np.exp(-(1j*np.pi*(N+1)/N)*(xx+yy))

# axis in arcsec
tel.xPSF_arcsec       = [-206265*(np.fix(N/2))*(tel.src.wavelength/tel.D) * (tel.resolution/N),206265*(np.fix(N/2))*(tel.src.wavelength/tel.D) * (tel.resolution/N)]
tel.yPSF_arcsec       = [-206265*(np.fix(N/2))*(tel.src.wavelength/tel.D) * (tel.resolution/N),206265*(np.fix(N/2))*(tel.src.wavelength/tel.D) * (tel.resolution/N)]

# axis in radians
tel.xPSF_rad   = [-(np.fix(N/2))*(tel.src.wavelength/tel.D) * (tel.resolution/N),(np.fix(N/2))*(tel.src.wavelength/tel.D) * (tel.resolution/N)]
tel.yPSF_rad   = [-(np.fix(N/2))*(tel.src.wavelength/tel.D) * (tel.resolution/N),(np.fix(N/2))*(tel.src.wavelength/tel.D) * (tel.resolution/N)]

# PSF computation
tel.PSF        = (np.abs(np.fft.fft2(supportPadded*tel.phasor))**2)
# PSF normalization
tel.PSF  = tel.PSF * (norma/tel.PSF.sum())

#%% -----------------------     ZERNIKE    ----------------------------------
nModes = 100                # number of zernike to compute

#create a zernike object
Z = Zernike(telObject = tel, J = nModes)

# compute the zernike polynomials associated to the telescope tel
Z.computeZernike(tel)

# access the Zernike modes: 
# 1) only the pixelsin the pupil 
print('Z.modes shape: nPixelsPupil * nZernike' + str(Z.modes.shape))

# 2) the 2D maps of the modes 
print('Z.modesFullRes shape : nPix * nPix * nZernike ' + str(Z.modesFullRes.shape))


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

# crop the image to get the core of the PSF
nCrop = int(tel.resolution * zeroPaddingFactor /3)

PSF_cropped = PSF_normalized[nCrop:-nCrop,nCrop:-nCrop]

plt.figure()
plt.imshow(tel.OPD)
# display
plt.figure()
ax1 = plt.subplot(1,2,1)
im1 = ax1.imshow(np.log(PSF_normalized),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.colorbar(im1)
im1.set_clim(vmin = -10, vmax = 0)

plt.xlabel('Angular Resolution [arcsec]')
plt.ylabel('Angular Resolution [arcsec]')

ax2 = plt.subplot(1,2,2)
im2 = ax2.imshow(np.log(PSF_cropped))
plt.colorbar(im2)
im2.set_clim(vmin = -10, vmax = 0)
plt.title('Zoom')


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
plt.subplot(1,3,1)
plt.imshow(tel.OPD)
plt.title('Aberrated Phase Screen [m]' )
plt.colorbar()
ax1 = plt.subplot(1,3,2)
im1 = ax1.imshow(np.log(PSF_normalized),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.colorbar(im1)
im1.set_clim(vmin = -10, vmax = 0)
plt.title('PSF')
plt.xlabel('Angular Resolution [arcsec]')
plt.ylabel('Angular Resolution [arcsec]')


ax2 = plt.subplot(1,3,3)
im2 = ax2.imshow(np.log(PSF_cropped))
plt.colorbar(im2)
im2.set_clim(vmin = -10, vmax = 0)
plt.title('Zoom')

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
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])

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
