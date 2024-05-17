# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:33:01 2020

@author: cheritie
"""

import matplotlib.pyplot as plt
import numpy             as np 
import time


# local modules 
from OOPAO.Telescope         import Telescope
from OOPAO.Source            import Source
from OOPAO.Atmosphere        import Atmosphere
from OOPAO.Pyramid           import Pyramid
from OOPAO.DeformableMirror  import DeformableMirror
from OOPAO.MisRegistration   import MisRegistration
from OOPAO.tools.tools import *
# calibration
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.calibration.ao_calibration         import ao_calibration

# ELT modules
from OOPAO.M1_model.make_ELT_pupil             import generateEeltPupilReflectivity

#%% -----------------------     read parameter file   ----------------------------------

from parameterFile_ELT_SCAO_K_Band_3000_KL   import initializeParameterFile

param = initializeParameterFile()
param['nModes'] =4200


from OOPAO.tools.set_paralleling_setup import set_paralleling_setup
#%% -----------------------     TELESCOPE   ----------------------------------
# create the pupil of the ELT

tmp = 1+0.*np.random.rand(798)

M1_pupil_reflectivity = generateEeltPupilReflectivity(refl = tmp,\
                                          npt       = param['resolution'],\
                                          dspider   = 0*param['spiderDiameter'],\
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
                centralObstruction  = 0, pupil=M1_pupil)

plt.figure()
plt.imshow(tel.pupil)

#%% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel


# combine the NGS to the telescope using '*' operator:

tel.computePSF(zeroPaddingFactor = 6)
PSF_diff = tel.PSF/tel.PSF.max()
N = 1000

fov_pix = tel.xPSF_arcsec[1]/tel.PSF.shape[0]
fov = N*fov_pix
plt.figure()
plt.imshow(np.log10(PSF_diff[N:-N,N:-N]), extent=(-fov,fov,-fov,fov))
plt.clim([-3.5,0])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')


#%% -----------------------     ATMOSPHERE   ----------------------------------
# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = 0.15,\
               L0            = param['L0'],\
               windSpeed     = [10,10],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'],
               param         = param)

# initialize atmosphere
atm.initializeAtmosphere(tel)

for i in range(atm.nLayer):
    atm.display_atm_layers(layer_index=None, fig_index=i+10)
#%%

misReg = MisRegistration(param)

param['m4_filename'] = 'E:/ESO_2020_2022/local/diskb/cheritier/psim/data_calibration/cropped_M4IF.fits'
# if no coordonates specified, create a cartesian dm
dm=DeformableMirror(telescope           = tel,\
                    nSubap              = 20,       # only used if default case is considered ( Fried geometry). Not used if using user-defined coordinates / M4 IFs. 
                    mechCoupling        = 0.45,     # only used if default case is considered ( gaussian IFs).
                    misReg              = misReg,   # mis-reg for the IFs
                    M4_param            = param,    # Necessary to read M4 IFs (using param['m4_filename'])
                    floating_precision  = 32)       # using float32 precision for tutorial


#%%
from OOPAO.tools.tools import read_fits
from OOPAO.tools.displayTools import *

M2C_KL = read_fits('E:/ESO_2020_2022/local/diskb/cheritier/psim/data_calibration/M2C_M4_576_res_continuous.fits')


# # show the first 10 zernikes
# dm.coefs = (M2C_KL[:,:10])
# tel*dm
# displayMap(tel.OPD,norma = True)

dm.coefs = M2C_KL
tel.resetOPD()

ngs*tel*dm

KL_basis = np.reshape(tel.OPD,[tel.resolution**2, tel.OPD.shape[2]])

KL_basis = np.squeeze(KL_basis[tel.pupilLogical,:])


plt.figure()
plt.imshow(KL_basis.T@KL_basis)

#%% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
# tel-atm
tel.resetOPD()

user_theta = []
# create the Pyramid Object
wfs = Pyramid(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = param['modulation'],\
              calibModulation = 0,\
              lightRatio            = param['lightThreshold'],\
              n_pix_separation      = 0,\
              psfCentering          = False,\
              postProcessing        = 'fullFrame')
    

#%% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
# tel-atm
tel.resetOPD()

user_theta = []
from AO_modules.BioEdge import BioEdge
# create the Pyramid Object
wfs = BioEdge(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = param['modulation'],\
              lightRatio            = param['lightThreshold'],\
              n_pix_separation      = 10,\
              psfCentering          = False,\
              postProcessing        = 'fullFrame',nTheta_user_defined = 40, grey_width = 0)
    

theta_ref = wfs.thetaModulation.copy()




#%% -----------------------     TELESCOPE   ----------------------------------
# create the pupil of the ELT

tmp = 1+0.*np.random.rand(798)
center = [param['resolution']+0.5,param['resolution']+0.5] 
M1_pupil_reflectivity_HR = generateEeltPupilReflectivity(refl = tmp,\
                                          npt       = param['resolution']*2,\
                                          dspider   = param['spiderDiameter'],\
                                          i0        = center[0]+param['m1_shiftX'] ,\
                                          j0        = center[1]+param['m1_shiftY'] ,\
                                          pixscale  = param['pixelSize']/2,\
                                          gap       = param['gapSize'],\
                                          rotdegree = param['m1_rotationAngle'],\
                                          softGap   = True)

# extract the valid pixels
M1_pupil_HR = M1_pupil_reflectivity_HR>0

# create the Telescope object
tel_HR = Telescope(resolution          = param['resolution']*2,\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = 0.,pupil = None)

    
    
ngs*tel_HR
plt.figure()
plt.imshow(tel_HR.pupil)

# mis-registrations object reading the param
misReg = MisRegistration(param)

coordinates = np.zeros([2,2])
# M4 is genererated already projected in the M1 space
dm_HR = DeformableMirror(telescope    = tel_HR,\
                    nSubap       = 2,\
                    misReg       = misReg,\
                    M4_param     = None,coordinates=coordinates,pitch=0.5)

plt.figure()
plt.imshow(np.reshape(dm_HR.modes[:,0],[tel_HR.resolution,tel_HR.resolution]),extent=([-tel.D/2,tel.D/2,-tel.D/2,tel.D/2]))
plt.plot(dm_HR.coordinates[:,0],dm_HR.coordinates[:,1],'xr')      

#%
wfs_HR = Pyramid(nSubap                = param['nSubaperture'],\
                telescope             = tel_HR,\
                modulation            = param['modulation'],\
                lightRatio            = param['lightThreshold'],\
                n_pix_separation      = 0,\
                psfCentering          = False,\
                postProcessing        = 'slopesMaps',calibModulation = 3)
    
    

dm_HR.coefs =  np.asarray([0,1])*1e-9
tel_HR*dm_HR*wfs_HR
    
plt.figure()
plt.imshow(wfs_HR.signal_2D)
    

#%% -----------------------    MODAL BASIS   ----------------------------------
# to manually compute the KL modal basis

# # compute the initial modal basis
# foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used 
# filename_M2C    = None  # name of the filename, if None a default name is used 
# #
# #
# M2C = compute_M2C(  telescope          = tel,\
#                   	atmosphere         = atm,\
#                     deformableMirror   = dm,\
#                     param              = param,\
#                     nameFolder         = None,\
#                     nameFile           = None,\
#                     remove_piston      = True,\
#                     HHtName            = None,\
#                     baseName           = None ,\
#                     minimF             = True,\
#                     nmo                = 4300,\
#                     ortho_spm          = True,\
#                     nZer               = 3)
#

filename_M2C = 'M2C_M4_KL_piston_tip_tilt_minimized_forces_576_res.npy'
M2C = np.load('data_calibration/'+filename_M2C)


# %% -----------------------    (General function) INTERACTION MATRIX   ----------------------------------
# this function can also be used but the previous one is recommanded

# from AO_modules.calibration.InteractionMatrix import InteractionMatrix
# # more or less the same function with the differences:
# #   _ it doesn't save the matrices computed
# #   _ it computes the interaction matrix for any modal basis, not only zonal ones
# # general way to compute an interaction matrix
# calib = InteractionMatrix(ngs           = ngs,\
#                           atm           = atm,\
#                           tel           = tel,\
#                           dm            = dm,\
#                           wfs           = wfs,\
#                           M2C           = np.float32(ao_calib.M2C),\
#                           stroke        = 1e-9,\
#                           phaseOffset   = 0,\
#                           nMeasurements = 25)


imat = np.load('data_calibration/ELT_K_band_96x96/zonal_interaction_matrix_384_res_3_mod_fullFrame_psfCentering_False.npy')
#%%
from AO_modules.calibration.CalibrationVault import CalibrationVault

calib = CalibrationVault(imat@M2C[:,:param['nModes']])



#%% -----------------------    CLOSED LOOP   ----------------------------------
plt.close('all')
# ModalGainsMatrix    = ao_calib.gOpt
# reconstructor       = np.matmul(M2C_CL,ao_calib.calib.M)          # reconstructor matrix wfsSignals to dm commands


from AO_modules.tools.displayTools import cl_plot
tel.resetOPD()
# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs
tel+atm

# dm.coefs[100] = -1

tel.computePSF(4)
plt.close('all')
    
# These are the calibration data used to close the loop
# calib_CL    = C
M2C_CL      = M2C[:,:param['nModes']].copy()

reconstructor = np.matmul(M2C_CL,calib.M)

#%%

import matplotlib.pyplot as plt
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

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,[[0,0],[0,0]],[dm.coordinates[:,0],dm.coordinates[:,1],dm.coefs],np.log10(tel.PSF_norma_zoom),np.log10(tel.PSF_norma_zoom)],\
                   type_fig          = ['imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                   list_title        = ['Turbulence OPD','Residual OPD','WFS Detector',None,None,None,None],\
                   list_lim          = [None,None,None,None,None,[-5,0],[-5,0]],\
                   list_label        = [None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],['Long Exposure_PSF','']],\
                   n_subplot         = [4,2],\
                   list_display_axis = [None,None,None,True,None,None,None],\
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=1)
# loop parameters
gainCL                  = 0.45
wfs.cam.photonNoise     = False
display                 = True


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
        
    dm.coefs=dm.coefs-np.float32(gainCL*np.matmul(reconstructor,wfsSignal))
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal=wfs.signal
    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    # update displays if required
    if display==True and i>0:        
        tel.computePSF(2)
        if i>15:
            SE_PSF.append(np.log10(tel.PSF_norma_zoom))
            LE_PSF = np.mean(SE_PSF, axis=0)
        
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,[np.arange(i),residual[:i]],dm.coefs,np.log10(tel.PSF_norma_zoom), LE_PSF],
                               plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
            break
    
    SR[i]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD=tel.OPD[np.where(tel.pupil>0)]

    print('Loop'+str(i)+'/'+str(param['nLoop'])+' Turbulence: '+str(total[i])+' -- Residual:' +str(residual[i])+ '\n')
