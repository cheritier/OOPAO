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

#%% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

tel.computePSF(zeroPaddingFactor = 2)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,5])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

src=Source(optBand   = 'K',\
           magnitude = param['magnitude'])

#%% -----------------------     ATMOSPHERE   ----------------------------------

# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'],\
               param = param)
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
tel.print_optical_path()
tel-atm
tel.print_optical_path()


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
#%% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel-atm


wfs = Pyramid(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = 1,\
              lightRatio            = 0.01,\
              n_pix_separation      = param['n_pix_separation'],\
              psfCentering          = False,\
              postProcessing        = 'slopesMaps')    
    




#%% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis

from OOPAO.tools.tools import read_fits
M2C_KL = read_fits('D:/ESO_2020_2022/local/diskb/cheritier/psim/data_calibration/M2C_M4_576_res_continuous.fits')

#%%

# show the first 10 zernikes
dm.coefs = (M2C_KL[:,:10])
tel*dm
displayMap(tel.OPD,norma = True)

#%% to manually measure the interaction matrix

# amplitude of the modes in m
stroke=1e-9
# Modal Interaction Matrix

# imat = read_fits('D:/ESO_2020_2022/local/diskb/cheritier/psim/data_calibration/ELT_K_band_96x96/zonal_interaction_matrix_576_res_3_mod_slopesMaps_psfCentering_False.fits')

calib = InteractionMatrix(ngs            = ngs,\
                          atm            = atm,\
                          tel            = tel,\
                          dm             = dm,\
                          wfs            = wfs,\
                          M2C            = M2C_KL[:,:1000],\
                          stroke         = stroke,\
                          nMeasurements  = 20,\
                          noise          = 'off',\
                          print_time     = True)



#%%

# Modal interaction matrix
# calib_zernike = CalibrationVault(imat@M2C_KL[:,:1000])

# plt.figure()
# plt.plot(np.sqrt(np.diag(calib_zernike.D.T@calib_zernike.D))/calib_zernike.D.shape[0]/ngs.fluxMap.sum())
# plt.xlabel('Mode Number')
# plt.ylabel('WFS slopes STD')

from convolutive_model import convolutive_model

ind = list(np.arange(50))
ind.append(200)
ind.append(400)
ind.append(800)
ind.append(999)

CM = convolutive_model(tel   = tel,
                       dm    = dm,
                       wfs   = wfs,
                       M2C   = M2C_KL[:,ind])

#%%

tel.resetOPD()
# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs
tel+atm

# dm.coefs[100] = -1

tel.computePSF(4)
plt.close('all')
    
# These are the calibration data used to close the loop
calib_CL    = calib
M2C_CL      = M2C_KL[:,:1000]


# combine telescope with atmosphere
tel+atm

# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs


plt.show()

param['nLoop'] = 500
# allocate memory to save data
SR                      = np.zeros(param['nLoop'])
total                   = np.zeros(param['nLoop'])
residual                = np.zeros(param['nLoop'])
wfsSignal               = np.arange(0,wfs.nSignal)*0
SE_PSF = []
LE_PSF = np.log10(tel.PSF_norma_zoom)
SE_PSF_K = []

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,np.log10(wfs.get_modulation_frame(radius = 20)),[[0,0],[0,0]],[dm.coordinates[:,0],(dm.coordinates[:,1]),dm.coefs],np.log10(tel.PSF_norma_zoom),np.log10(tel.PSF_norma_zoom)],\
                   type_fig          = ['imshow','imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                   list_title        = ['Turbulence OPD [m]','Residual OPD [m]','WFS Detector Plane','WFS Focal Plane',None,None,None,None],\
                   list_lim          = [None,None,None,[-3,0],None,None,[-4,0],[-5,0]],\
                   list_label        = [None,None,None,None,['Time [ms]','WFE [nm]'],['DM Commands',''],['Short Exposure I Band PSF',''],['Long Exposure K Band PSF','']],\
                   n_subplot         = [4,2],\
                   list_display_axis = [None,None,None,None,True,None,None,None],\
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
# loop parameters
gainCL                  = 0.6
wfs.cam.photonNoise     = True
display                 = True

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
    ngs*tel*dm*wfs
    
    
    if i%10 ==0:
        
        G = np.eye(1000)
        G[:100,:100] = np.diag(1/CM.optical_gains(wfs.modulation_camera_frame))
        reconstructor = np.matmul(M2C_CL,np.matmul(G,calib_CL.M))

        
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,wfsSignal)
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal=wfs.signal
    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    # update displays if required
    if display==True:        
        tel.computePSF(4)
        SE_PSF.append(np.log10(tel.PSF_norma_zoom))

        if i>15:
            src*tel
            tel.computePSF(4)
            SE_PSF_K.append(np.log10(tel.PSF_norma_zoom))

            LE_PSF = np.mean(SE_PSF_K, axis=0)
        
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,np.log10(wfs.get_modulation_frame(radius=20)),[np.arange(i+1),residual[:i+1]],dm.coefs,(SE_PSF[-1]), LE_PSF],
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

