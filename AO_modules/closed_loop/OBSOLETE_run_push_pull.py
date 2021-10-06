# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:30:10 2020

@author: cheritie
"""

# commom modules
import matplotlib.pyplot as plt
import numpy             as np 
import time

# local modules 
from AO_modules.calibration.CalibrationVault  import calibrationVault
from AO_modules.MisRegistration   import MisRegistration
from AO_modules.calibration.InteractionMatrix import interactionMatrixOnePass
from AO_modules.tools.tools                  import createFolder
from AO_modules.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration


#####################################################################################################################################################
#   # # #   #   #   # # #   #   #           # # #   #   #   #       #
#   #   #   #   #   #       #   #           #   #   #   #   #       #
#   # # #   #   #   # # #   # # #   # # #   # # #   #   #   #       #
#   #       #   #       #   #   #           #       #   #   #       #
#   #        # #    # # #   #   #           #        # #    # # #   # # #
#####################################################################################################################################################


def run_cl_push_pull(param,obj):
        
    nLoop = param['nLoop']
    gain_cl = param['gainCL']
    
    #% ------------------------------------ Noise and Turbulence Update ------------------------------------
    
    # set magnitude of the NGS
    obj.ngs.nPhoton = param['nPhotonPerSubaperture'] * (1/obj.tel.samplingTime)/((obj.tel.D/param['nSubaperture'] )**2)
    
    extraName =''
    # photon Noise
    obj.wfs.cam.photonNoise = param['photonNoise']
    if obj.wfs.cam.photonNoise:
        extraName+=('photonNoise_')
    
    # RON
    obj.wfs.cam.readoutNoise = param['readoutNoise']
    
    if obj.wfs.cam.readoutNoise:
        extraName+=('readoutNoise_')
    
    
    #% ------------------------------------ Calibration ------------------------------------
    
    calib_cl    = calibrationVault(obj.calib.D[:,:param['nModes']])
    M2C_cl      = obj.M2C_cl
    
    
    #% ------------------------------------ push-pull properties  ------------------------------------
    
    amplitude_push_pull      = obj.amplitude_push_pull
    M2C_push_pull            = obj.M2C_push_pull
    number_push_pull         = obj.number_push_pull
    
    
    #%% ------------------------------------ mis-registrations  ------------------------------------
    
    misRegistration_cl = MisRegistration(param)
    
    misRegistration_cl.show()
    obj.dm.misReg.show()
    
#    case with no mis-registration
    if misRegistration_cl == obj.dm.misReg:
        dm_cl = obj.dm
    else:
        dm_cl = applyMisRegistration(obj.tel,misRegistration_cl,param)
   
    
    
    #%% ------------------------------------ closed loop data to be saved  ------------------------------------
    
    wfs_signals         = np.zeros([number_push_pull,nLoop,obj.wfs.nSignal])
    dm_commands         = np.zeros([number_push_pull,nLoop,obj.dm.nValidAct])
    ao_residuals        = np.zeros([number_push_pull,nLoop])
    ao_turbulence       = np.zeros([number_push_pull,nLoop])
    push_pull_buffer    = np.zeros([number_push_pull,obj.wfs.nSignal,M2C_push_pull.shape[1]])
    petalling_nm        = np.zeros([number_push_pull,6,nLoop])

    #%% ------------------------------------ destination folder  ------------------------------------
    
    misregistrationFolder = misRegistration_cl.misRegName
    destinationFolder =  param['pathOutput']+ '/'+param['name'] +'/'+ misregistrationFolder + '/' + extraName + '/'
    createFolder(destinationFolder)
    
    
    #%% Loop initialization
    
    gain_cl = param['gainCL']
    
    # combine with atmosphere
    obj.tel+obj.atm
    plt.close('all')
    
    # initialize DM commands
    dm_cl.coefs=0
    
    # propagate through th whole system
    obj.ngs*obj.tel*dm_cl*obj.wfs
    
    # setup the display
    if obj.display:
        plt.figure(79)
        plt.ion()
        
        ax1=plt.subplot(1,2,1)
        im_atm=ax1.imshow(obj.tel.src.phase)
        plt.colorbar(im_atm)
        plt.title('Turbulence phase [rad]')
        
        
        ax2=plt.subplot(1,2,2)
        im_residual=ax2.imshow(obj.tel.src.phase)
        plt.colorbar(im_residual)
        plt.title('Residual Phase [rad] ')
    
#%% ------------------------------------ Modal Gains  ------------------------------------
    ModalGainsMatrix = obj.gOpt

    # Loop initialization
    reconstructor = np.matmul(M2C_cl,np.matmul(ModalGainsMatrix,calib_cl.M))

#%% ------------------------------------ Closed loop  ------------------------------------
    
    for i_push_pull in range(number_push_pull):
        wfsSignal = np.zeros(obj.wfs.nSignal)
        
        for i_loop in range(nLoop):
            a= time.time()
            # update atmospheric phase screen
            obj.atm.update()
            
            # save phase variance
            ao_turbulence[i_push_pull,i_loop]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9
            
            # save turbulent phase
            turbPhase=obj.tel.src.phase
            
            # propagate to the WFS with the CL commands applied
            obj.tel*dm_cl*obj.wfs
            # release the controller
            if i_loop==nLoop-1:
                # save the residual phase over which the push pull will be achieved
                ao_residualPhase = obj.tel.src.phase
                calib_push = interactionMatrixOnePass(ngs               = obj.ngs,\
                                                      atm               = obj.atm,\
                                                      tel               = obj.tel,\
                                                      dm                = dm_cl,\
                                                      wfs               = obj.wfs,\
                                                      M2C               = M2C_push_pull,\
                                                      stroke            = amplitude_push_pull,\
                                                      phaseOffset       = ao_residualPhase,\
                                                      nMeasurements     = 1,\
                                                      noise             ='on')
                push = calib_push.D
                obj.tel+obj.atm
                # update the atmosphere phase screen
                obj.atm.update()
                # re compute the residual phase
                dm_cl.coefs = np.squeeze(dm_commands[i_push_pull,i_loop-1,:])
                obj.tel*dm_cl
                ao_residualPhase = obj.tel.src.phase
                calib_pull = interactionMatrixOnePass(ngs               = obj.ngs,\
                                                      atm               = obj.atm,\
                                                      tel               = obj.tel,\
                                                      dm                = dm_cl,\
                                                      wfs               = obj.wfs,\
                                                      M2C               = -M2C_push_pull,\
                                                      stroke            = amplitude_push_pull,\
                                                      phaseOffset       = ao_residualPhase,\
                                                      nMeasurements     = 1,\
                                                      noise             ='on')                
                pull = calib_pull.D
                
                push_pull_buffer[i_push_pull,:,:] = push-pull
                
                dm_cl.coefs = np.squeeze(dm_commands[i_push_pull,i_loop-1,:])
                obj.tel+obj.atm

                obj.tel*dm_cl*obj.wfs
            b= time.time()

              
                
            dm_commands[i_push_pull,i_loop,:] = dm_cl.coefs
            
            dm_cl.coefs=dm_cl.coefs-gain_cl*np.matmul(reconstructor,wfsSignal)
            if obj.tel.isPetalFree:
                dm_cl.OPD = obj.tel.removePetalling(dm_cl.OPD)
            ao_residuals[i_push_pull,i_loop]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9

            # store the slopes after computing the commands => 2 frames delay
            wfsSignal=obj.wfs.pyramidSignal
            
            wfs_signals[i_push_pull,i_loop,:] = wfsSignal
            if obj.display==True:
                
                # turbulence phase
                im_atm.set_data(turbPhase)
                im_atm.set_clim(vmin=turbPhase.min(),vmax=turbPhase.max())
                
                # residual phase
                tmp = obj.tel.src.phase
                tmp-=np.mean(tmp[obj.tel.pupil])
                im_residual.set_data(tmp)
                im_residual.set_clim(vmin=tmp.min(),vmax=tmp.max()) 
                plt.draw()
                plt.show()
                plt.pause(0.001)
            if obj.printPetals:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                for i_petal in range(6):
                    petal       = obj.tel.getPetalOPD(i_petal)*1e9             
                    petal_nm = np.round(np.mean(petal[np.where(obj.tel.petalMask == 1)]),5)
                    print('Petal '+str(i_petal+1) +': ' + str(petal_nm))
                    petalling_nm[i_push_pull,i_petal,i_loop] = petal_nm
                    
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Loop'+str(i_loop)+'/'+str(nLoop)+'Turbulence: '+str(ao_turbulence[i_push_pull,i_loop])+' -- Residual:' +str(ao_residuals[i_push_pull,i_loop])+ '\n')
            print('Elapsed Time: ' + str(b-a) + ' s ')

    


    dataCL = dict()
    dataCL['ao_residual']          = ao_residuals
    dataCL['ao_turbulence']        = ao_turbulence
#    dataCL['wfs_signals']          = wfs_signals
#    dataCL['dmCommands']           = dm_commands
    dataCL['push_pull_buffer']     = push_pull_buffer
    dataCL['interactionMatrix']    = calibrationVault(np.mean(push_pull_buffer,axis = 0))
    dataCL['parameterFile']        = param
#    dataCL['petalling']        = petalling_nm
    dataCL['misRegistration']      = misRegistration_cl
    dataCL['destinationFolder']    = destinationFolder
    return dataCL
                   
        