# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""
# commom modules
import matplotlib.pyplot as plt
import numpy             as np 

import __load__psim
__load__psim.load_psim()

from AO_modules.Atmosphere       import Atmosphere
from AO_modules.Pyramid          import Pyramid
from AO_modules.DeformableMirror import DeformableMirror
from AO_modules.MisRegistration  import MisRegistration
from AO_modules.Telescope        import Telescope
from AO_modules.Source           import Source
# calibration modules 
from AO_modules.calibration.compute_KL_modal_basis import compute_M2C
from AO_modules.calibration.ao_calibration import ao_calibration
# display modules
from AO_modules.tools.displayTools           import displayMap

#%% -----------------------     read parameter file   ----------------------------------
from parameterFile_VLT_I_Band import initializeParameterFile
param = initializeParameterFile()

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

tel.computePSF(zeroPaddingFactor = 2)
plt.figure()
plt.imshow(np.log(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])

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
plt.imshow((np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')

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

#%% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel-atm
# create the Pyramid Object
wfs = Pyramid(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = param['modulation'],\
              lightRatio            = param['lightThreshold'],\
              pupilSeparationRatio  = param['pupilSeparationRatio'],\
              calibModulation       = param['calibrationModulation'],\
              psfCentering          = param['psfCentering'],\
              edgePixel             = param['edgePixel'],\
              extraModulationFactor = param['extraModulationFactor'],\
              postProcessing        = param['postProcessing'])

#%% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis
foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used 
filename_M2C    = None  # name of the filename, if None a default name is used 

M2C = compute_M2C(  telescope          = tel,\
                  	atmosphere         = atm,\
                    deformableMirror   = dm,\
                    param              = param,\
                    nameFolder         = None,\
                    nameFile           = None,\
                    remove_piston      = True,\
                    HHtName            = 'covariance_matrix_VLT',\
                    baseName           = None ,\
                    nmo                = 300,\
                    ortho_spm          = True,\
                    nZer               = 3)

#%%
ao_calib =  ao_calibration(param            = param,\
                           ngs              = ngs,\
                           tel              = tel,\
                           atm              = atm,\
                           dm               = dm,\
                           wfs              = wfs,\
                           nameFolderIntMat = None,\
                           nameIntMat       = None,\
                           nameFolderBasis  = None,\
                           nameBasis        = None,\
                           nMeasurements    = 100)

#%% ----------------               PHASE OFFSET  ----------------

#compute a residual phase (from fitting error) representative of two successive phase screens


phi_fitting = ao_calib.basis@ao_calib.projector@np.reshape(atm.OPD,tel.resolution*tel.resolution)

phi_fitting_2D = np.reshape(phi_fitting,[tel.resolution,tel.resolution])

phi_residual_push = atm.OPD - phi_fitting_2D

atm.update()

phi_fitting = ao_calib.basis@ao_calib.projector@np.reshape(atm.OPD,tel.resolution*tel.resolution)

phi_fitting_2D = np.reshape(phi_fitting,[tel.resolution,tel.resolution])

phi_residual_pull = atm.OPD - phi_fitting_2D

var_fit = np.std(phi_residual_pull[np.where(tel.pupil>0)])*1e9
    

plt.figure()
plt.imshow(phi_residual_push*1e9)
plt.colorbar()
plt.title('WFE: '+str(var_fit) + ' nm')

#%% ------------------ MODAL BASIS CONSIDERED ------------------
from AO_modules.tools.tools import emptyClass


basis =  emptyClass()
basis.modes         = np.asarray([ao_calib.M2C[:,32],ao_calib.M2C[:,32]]).T
basis.extra         = 'KL'       
basis.indexModes    = [1,basis.modes.shape[1]] 


dm.coefs = basis.modes
tel*dm

plt.figure()
displayMap(tel.OPD)

  


#%%
from AO_modules.mis_registration_identification_algorithm.estimateMisRegistration import estimateMisRegistration,applyMisRegistration
from AO_modules.calibration.CalibrationVault import calibrationVault


misRegistrationZeroPoint = MisRegistration()
epsilonMisRegistration                  = MisRegistration()
epsilonMisRegistration.shiftX           = np.round(dm.pitch /10,4)
epsilonMisRegistration.shiftY           = np.round(dm.pitch /10,4)
epsilonMisRegistration.rotationAngle    = np.round(np.rad2deg(np.arctan(epsilonMisRegistration.shiftX)/(tel.D/2)),4)

nameFolder_sensitivity_matrice = param['pathInput'] +'/'+ param['name']+'/sensitivity_matrices/'


n = 11
shiftX =    [20,40,10,-40,-20,-10,30,25,-17,-28,10]
shiftY =    [-10,10,0,15,20,-10,15,15,30,20,0]
rot =       [1, -1.5, 0.3, 0.5, 0,0.35, 0.2,1.5,2,0.35,-0.2]



amp = [50]

misReg_out = np.zeros([3,n,3])
gamma_out = np.zeros([3,n])

for i_misReg in range(len(shiftX)):
    
    misRegValuesX   = shiftX[i_misReg]
    misRegValuesY   = shiftY[i_misReg]
    misRegValuesRot = rot[i_misReg]
            
    misRegistration_cl = MisRegistration(param)
    misRegistration_cl.shiftX = misRegValuesX*dm.pitch/100
    misRegistration_cl.shiftY = misRegValuesY*dm.pitch/100
    misRegistration_cl.rotationAngle = misRegValuesRot

    
    dm_cl = applyMisRegistration(tel,misRegistration_cl,param)
    
    for i_amp in range(len(amp)):
        wfs.cam.photonNoise = True
        
        im = np.zeros([wfs.nSignal,2])
        for j in range(2):
  #  PUSH
  # set residual OPD for tel.OPD
            tel.OPD = phi_residual_push 
  # apply modes on the DM
            dm_cl.coefs = basis.modes[:,j] * amp[i_amp]*1e-9
  # propagate through DM and WFS
            tel*dm_cl*wfs
  # save wfs signals for the push            
            sp = wfs.pyramidSignal
            
  #  PULL
  # set residual OPD for tel.OPD
            tel.OPD = phi_residual_pull
  # apply modes on the DM
            dm_cl.coefs = - basis.modes[:,j] * amp[i_amp]*1e-9
  # propagate through DM and WFS
            tel*dm_cl*wfs
  # save wfs signals for the pull            
            sm = wfs.pyramidSignal 
  # compute the interaction matrix            
            im[:,j] = 0.5*(sp-sm)/ (amp[i_amp]*1e-9)
  # save it as a calibration object
        calib_misReg = calibrationVault(im)       
  # estimate mis-registrations
        [mis_reg_est,gamma,alpha] = estimateMisRegistration(nameFolder               = nameFolder_sensitivity_matrice,\
                                                 nameSystem                 = '',\
                                                 tel                        = tel,\
                                                 atm                        = atm,\
                                                 ngs                        = ngs,\
                                                 dm_0                       = dm,\
                                                 calib_in                   = calib_misReg,\
                                                 wfs                        = wfs,\
                                                 basis                      = basis,\
                                                 misRegistrationZeroPoint   = misRegistrationZeroPoint,\
                                                 epsilonMisRegistration     = epsilonMisRegistration,\
                                                 param                      = param,\
                                                 precision                  = 5,\
                                                 return_all                 = True)
        gamma = np.asarray(gamma)   
        alpha = np.asarray(alpha)   
        
        # saving the last estimation of the parameters
        gamma_out[i_amp,i_misReg]       = gamma[-1,0]    
        misReg_out[i_amp,i_misReg,:]    = alpha[-1,:]
        
        print(alpha[-1,:])
    



#%%
from AO_modules.tools.displayTools import makeSquareAxes
plt.figure()
plt.subplot(2,2,1)
plt.title('Optical Gains')
plt.plot(gamma_out.T)
plt.ylim([0,1])
plt.legend(['Target', 'Estimation' ])


makeSquareAxes(plt.gca())
plt.xlabel('Input Shift X [ % of a subap. ]')
plt.ylabel('Optical Gain')


plt.figure()
plt.title('Shift X')
plt.plot(shiftX,'--k')
plt.plot(misReg_out[:,:,1].T/(dm.pitch/100))
plt.xlabel('Input Shift X [% of a subap.]')
plt.ylabel('Estimated Shift X [ % of a subap. ]')
plt.legend(['Target', 'Amplitude 10 nm','Amplitude 50 nm','Amplitude 100 nm' ])

makeSquareAxes(plt.gca())

plt.figure()
plt.title('Shift Y')
plt.plot(shiftY,'--k')
plt.plot(misReg_out[:,:,2].T/(dm.pitch/100))
plt.xlabel('Input Shift X [ % of a subap. ]')
plt.ylabel('Estimated Shift Y [ % of a subap. ]')
#plt.ylim([-10,10])
makeSquareAxes(plt.gca())

plt.figure()
plt.title('Rotation')
plt.plot(rot,'--k')
plt.plot(misReg_out[:,:,0].T)
plt.xlabel('Input Shift X [ % of a subap. ]')
plt.ylabel('Estimated Rotation [ deg ]')
makeSquareAxes(plt.gca())

#%%



plt.figure()
plt.subplot(2,2,1)
plt.title('Optical Gains')
plt.plot(gamma_out.T)
plt.ylim([0,1])
plt.legend(['Target', 'Estimation' ])


makeSquareAxes(plt.gca())
plt.xlabel('Input Shift X [ % of a subap. ]')
plt.ylabel('Optical Gain')



