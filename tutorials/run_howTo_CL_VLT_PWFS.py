# -*- coding: utf-8 -*-
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
from OOPAO.Pyramid import Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.tools.displayTools import displayMap
# %% -----------------------     read parameter file   ----------------------------------
from parameter_files.parameterFile_VLT_I_Band_PWFS import initializeParameterFile

param = initializeParameterFile()

# %%
plt.ion()

#%% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])

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

#%% -----------------------     ATMOSPHERE   ----------------------------------

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

atm.update()

plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()


tel+atm
tel.computePSF(8)
plt.figure()
plt.imshow((np.log10(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,3])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)

plt.figure()
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

#%% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel-atm

wfs = Pyramid(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = param['modulation'],\
              lightRatio            = param['lightThreshold'],\
              n_pix_separation      = param['n_pix_separation'],\
              psfCentering          = param['psfCentering'],\
              postProcessing        = param['postProcessing'])

tel*wfs
plt.close('all')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame')




#%% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis
# foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used 
# filename_M2C    = None  # name of the filename, if None a default name is used 
# # KL Modal basis
# M2C_KL = compute_M2C(telescope            = tel,\
#                                  atmosphere         = atm,\
#                                  deformableMirror   = dm,\
#                                  param              = param,\
#                                  nameFolder         = None,\
#                                  nameFile           = None,\
#                                  remove_piston      = True,\
#                                  HHtName            = None,\
#                                  baseName           = None ,\
#                                  mem_available      = 8.1e9,\
#                                  minimF             = False,\
#                                  nmo                = 300,\
#                                  ortho_spm          = True,\
#                                  SZ                 = np.int(2*tel.OPD.shape[0]),\
#                                  nZer               = 3,\
#                                  NDIVL              = 1)
#
# ao_calib =  ao_calibration(param            = param,\
#                            ngs              = ngs,\
#                            tel              = tel,\
#                            atm              = atm,\
#                            dm               = dm,\
#                            wfs              = wfs,\
#                            nameFolderIntMat = None,\
#                            nameIntMat       = None,\
#                            nameFolderBasis  = None,\
#                            nameBasis        = None,\
#                            nMeasurements    = 100)

#%% ZERNIKE Polynomials
from OOPAO.Zernike import Zernike
# create Zernike Object
Z = Zernike(tel,300)
# compute polynomials for given telescope
Z.computeZernike(tel)

# mode to command matrix to project Zernike Polynomials on DM
M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:]))@Z.modes

# show the first 10 zernikes
dm.coefs = M2C_zernike[:,:10]
tel*dm
displayMap(tel.OPD)

#%% to manually measure the interaction matrix

# amplitude of the modes in m
stroke=1e-9
# Modal Interaction Matrix 
from OOPAO.calibration.InteractionMatrix import InteractionMatrix


#%%
M2C_zonal = np.eye(dm.nValidAct)
# zonal interaction matrix
calib_zonal = InteractionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_zonal,\
                            stroke         = stroke,\
                            nMeasurements  = 25,\
                            noise          = 'off')

plt.figure()
plt.plot(np.std(calib_zonal.D,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

#%%

from OOPAO.calibration.CalibrationVault import CalibrationVault


# Modal interaction matrix
calib_zernike = CalibrationVault(calib_zonal.D@M2C_zernike)

plt.figure()
plt.plot(np.std(calib_zernike.D,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

#%%

from OOPAO.tools.displayTools import cl_plot
tel.resetOPD()
# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs
tel+atm

# dm.coefs[100] = -1

tel.computePSF(4)
plt.close('all')
    
# These are the calibration data used to close the loop
calib_CL    = calib_zernike
M2C_CL      = M2C_zernike


# combine telescope with atmosphere
tel+atm

# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs


plt.show()

param['nLoop'] = 200
# allocate memory to save data
SR                      = np.zeros(param['nLoop'])
total                   = np.zeros(param['nLoop'])
residual                = np.zeros(param['nLoop'])
wfsSignal               = np.arange(0,wfs.nSignal)*0
SE_PSF = []
LE_PSF = np.log10(tel.PSF_norma_zoom)

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,np.log10(wfs.get_modulation_frame(radius = 10)),[[0,0],[0,0]],[dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],np.log10(tel.PSF_norma_zoom),np.log10(tel.PSF_norma_zoom)],\
                   type_fig          = ['imshow','imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                   list_title        = ['Turbulence OPD','Residual OPD','WFS Detector','WFS Modulation Camera',None,None,None,None],\
                   list_lim          = [None,None,None,[-3,0],None,None,[-4,0],[-4,0]],\
                   list_label        = [None,None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],['Long Exposure_PSF','']],\
                   n_subplot         = [4,2],\
                   list_display_axis = [None,None,None,None,True,None,None,None],\
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
# loop parameters
gainCL                  = 0.6
wfs.cam.photonNoise     = True
display                 = False

reconstructor = M2C_CL@calib_CL.M

for i in range(param['nLoop']):
    a=time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # save turbulent phase
    turbPhase = tel.src.phase
    # propagate to the WFS with the CL commands applied
    tel*dm*wfs
        
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,wfsSignal)
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal=wfs.signal
    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    # update displays if required
    if display==True:        
        tel.computePSF(4)
        if i>15:
            SE_PSF.append(np.log10(tel.PSF_norma_zoom))
            LE_PSF = np.mean(SE_PSF, axis=0)
        
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,np.log10(wfs.get_modulation_frame(radius=10)),[np.arange(i+1),residual[:i+1]],dm.coefs,np.log10(tel.PSF_norma_zoom), LE_PSF],
                               plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
            break
    
    SR[i]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD=tel.OPD[np.where(tel.pupil>0)]

    print('Loop'+str(i)+'/'+str(param['nLoop'])+' Turbulence: '+str(total[i])+' -- Residual:' +str(residual[i])+ '\n')

#%%
plt.figure()
plt.plot(total)
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')

