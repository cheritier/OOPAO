# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:29:43 2022

@author: cheritie
"""
# commom modules
import matplotlib.pyplot as plt
import numpy             as np 

import __load__oopao
__load__oopao.load_oopao()

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
from parameter_files.parameterFile_VLT_I_Band_PWFS import initializeParameterFile
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
plt.xlabel('arcsec')
plt.ylabel('arcsec')

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
plt.imshow(tel.pupil,extent=(-tel.D/2,tel.D/2,-tel.D/2,tel.D/2))
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'o')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.legend(['Valid Actuators'])

#%% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel-atm
# create the Pyramid Object
wfs = Pyramid(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = param['modulation'],\
              lightRatio            = param['lightThreshold'],\
              n_pix_separation      = param['n_pix_separation'],\
              psfCentering          = param['psfCentering'],\
              postProcessing        = param['postProcessing'])

plt.figure()
plt.imshow(wfs.cam.frame)
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
                           nMeasurements    = 1)

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

#%% ------------------ SETTING UP SPRINT ALGORITHM ------------------
from AO_modules.tools.tools import emptyClass

"""
Case where the initial mis-registration is quite small (Closed Loop) => USE A MIDDLE ORDER MODE!
"""
# modal basis considered
index_modes = [100]
basis =  emptyClass()
basis.modes         = ao_calib.M2C[:,index_modes]
basis.extra         = 'VLT_KL_'+str(index_modes[0])              # EXTRA NAME TO DISTINGUISH DIFFERENT SENSITIVITY MATRICES, BE CAREFUL WITH THIS!     


dm.coefs = basis.modes
tel*dm

plt.figure()
displayMap(tel.OPD)

obj =  emptyClass()
obj.ngs     = ngs
obj.tel     = tel
obj.atm     = atm
obj.wfs     = wfs
obj.dm      = dm
obj.param   = param


from AO_modules.SPRINT import SPRINT

Sprint = SPRINT(obj,basis, recompute_sensitivity=True)


#%% -------------------------- EXAMPLE WITH SMALL MIS-REGISTRATIONS - NO UPDATE OF MODEL--------------------------------------

from AO_modules.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration


# mis-reg considered
shiftX =    [30,40,10,-40,-20]          # in % of subap
shiftY =    [-40,10,0,15,20]            # in % of subap
rot =       [1, -1.5, 0.3, 0.25, 0]     # in degrees

n = len(shiftX)

amp = [100]

misReg_out = []
scaling_factor_out = []
for i_misReg in range(n):
    
    # APPLY MIS-REGISTRATION ON THE DM
    misRegistration_cl = MisRegistration(param)
    misRegistration_cl.shiftX = shiftX[i_misReg]*dm.pitch/100
    misRegistration_cl.shiftY = shiftY[i_misReg]*dm.pitch/100
    misRegistration_cl.rotationAngle = rot[i_misReg]

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Mis-Registrations Introduced:')
    misRegistration_cl.print_()
    
    dm_cl = applyMisRegistration(tel,misRegistration_cl,param,print_dm_properties=False)
    
    for i_amp in range(len(amp)):
        wfs.cam.photonNoise = True
        
        im = np.zeros([wfs.nSignal])
        #  PUSH
        # set residual OPD for tel.OPD
        tel.OPD = phi_residual_push 
        # apply modes on the DM
        dm_cl.coefs = basis.modes * amp[i_amp]*1e-9
        # propagate through DM and WFS
        tel*dm_cl*wfs
        # save wfs signals for the push            
        sp = wfs.signal
      
        #  PULL
        # set residual OPD for tel.OPD
        tel.OPD = phi_residual_pull
        # apply modes on the DM
        dm_cl.coefs = - basis.modes * amp[i_amp]*1e-9
        # propagate through DM and WFS
        tel*dm_cl*wfs
        # save wfs signals for the pull            
        sm = wfs.signal 
        # compute the interaction matrix            
        im = 0.5*(sp-sm)/ (amp[i_amp]*1e-9)
            
            
        # estimate mis-registrations
        Sprint.estimate(obj                 = obj,\
                        on_sky_slopes       = im,\
                        n_iteration         = 6,\
                        gain_estimation     = 0.8)        
            
        misReg_out.append(Sprint.mis_registration_buffer[-1])
        scaling_factor_out.append(Sprint.scaling_factor[-1])
        
    

misReg_out = np.asarray(misReg_out)

#%%
from AO_modules.tools.displayTools import makeSquareAxes

plt.figure()
plt.subplot(2,2,1)
plt.title('Scaling Factor')
plt.plot(np.asarray(scaling_factor_out)*0+1,'--k')

plt.plot(scaling_factor_out)
plt.xlabel('Mis-Registration Case')

plt.ylabel('Scaling Factor')
plt.legend(['Amplitude 10 nm','Amplitude 50 nm'])
# plt.ylim([-1,1])

makeSquareAxes(plt.gca())
plt.subplot(2,2,2)
plt.title('Shift X')
plt.plot(shiftX,'--k')
plt.plot(misReg_out[:,0].T/(dm.pitch/100))
plt.xlabel('Mis-Registration Case')
plt.ylabel('Estimated Shift X [ % of a subap. ]')
plt.legend(['Target', 'Amplitude 10 nm','Amplitude 50 nm'])
makeSquareAxes(plt.gca())

plt.subplot(2,2,3)
plt.title('Shift Y')
plt.plot(shiftY,'--k')
plt.plot(misReg_out[:,1].T/(dm.pitch/100))
plt.xlabel('Mis-Registration Case')
plt.ylabel('Estimated Shift Y [ % of a subap. ]')
plt.legend(['Target', 'Amplitude 10 nm','Amplitude 50 nm'])

#plt.ylim([-10,10])
makeSquareAxes(plt.gca())

plt.subplot(2,2,4)
plt.title('Rotation')
plt.plot(rot,'--k')
plt.plot(misReg_out[:,2].T)
plt.xlabel('Mis-Registration Case')
plt.ylabel('Estimated Rotation [ deg ]')
plt.legend(['Target', 'Amplitude 10 nm','Amplitude 50 nm'])

makeSquareAxes(plt.gca())


#%% -------------------------- EXAMPLE WITH LARGE MIS-REGISTRATIONS + UPDATE OF MODEL --------------------------------------
from AO_modules.tools.tools import emptyClass

"""
Case where the initial mis-registration is very large (Bootstrapping) => USE A LOW ORDER MODE!
"""
# modal basis considered
index_modes = [10]
basis =  emptyClass()
basis.modes         = ao_calib.M2C[:,index_modes]
basis.extra         = 'KL_'+str(index_modes[0])+'_'+str(index_modes[-1])              # EXTRA NAME TO DISTINGUISH DIFFERENT SENSITIVITY MATRICES, BE CAREFUL WITH THIS!     


dm.coefs = basis.modes
tel*dm

plt.figure()
displayMap(tel.OPD)

obj =  emptyClass()
obj.ngs     = ngs
obj.tel     = tel
obj.atm     = atm
obj.wfs     = wfs
obj.dm      = dm
obj.param   = param


from AO_modules.SPRINT import SPRINT

Sprint = SPRINT(obj,basis,recompute_sensitivity=True)
from AO_modules.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration


# mis-reg considered
shiftX =    [120] # in % of subap
shiftY =    [150] # in % of subap
rot =       [2]   # in degrees

n = len(shiftX)

amp = [50]

misReg_out = []
scaling_factor_out = []
for i_misReg in range(n):
    
    misRegValuesX   = shiftX[i_misReg]
    misRegValuesY   = shiftY[i_misReg]
    misRegValuesRot = rot[i_misReg]
            
    misRegistration_cl = MisRegistration(param)
    misRegistration_cl.shiftX = misRegValuesX*dm.pitch/100
    misRegistration_cl.shiftY = misRegValuesY*dm.pitch/100
    misRegistration_cl.rotationAngle = misRegValuesRot
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Mis-Registrations Introduced:')
    print('Rotation [deg] \t Shift X [m] \t Shift Y [m]')
    print(str(misRegistration_cl.rotationAngle)   + '\t\t' +str(misRegistration_cl.shiftX)+'\t\t' + str(misRegistration_cl.shiftY))
    
    dm_cl = applyMisRegistration(tel,misRegistration_cl,param,print_dm_properties=False)
    
    for i_amp in range(len(amp)):
        wfs.cam.photonNoise = True
        
        im = np.zeros([wfs.nSignal])
        #  PUSH
        # set residual OPD for tel.OPD
        tel.OPD = phi_residual_push 
        # apply modes on the DM
        dm_cl.coefs = basis.modes * amp[i_amp]*1e-9
        # propagate through DM and WFS
        tel*dm_cl*wfs
        # save wfs signals for the push            
        sp = wfs.signal
      
        #  PULL
        # set residual OPD for tel.OPD
        tel.OPD = phi_residual_pull
        # apply modes on the DM
        dm_cl.coefs = - basis.modes * amp[i_amp]*1e-9
        # propagate through DM and WFS
        tel*dm_cl*wfs
        # save wfs signals for the pull            
        sm = wfs.signal 
        # compute the interaction matrix            
        im = 0.5*(sp-sm)/ (amp[i_amp]*1e-9)
            
            
        # estimate mis-registrations
        Sprint.estimate(obj                 = obj,\
                        on_sky_slopes       = im,\
                        n_iteration         = 10,\
                        n_update_zero_point = 2,\
                        gain_estimation     = 0.8)            
            
        misReg_out.append(Sprint.mis_registration_buffer[-1])
        scaling_factor_out.append(Sprint.scaling_factor[-1])
        
    

misReg_out = np.asarray(misReg_out)


