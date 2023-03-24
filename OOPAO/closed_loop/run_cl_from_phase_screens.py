# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:18:31 2020

@author: cheritie
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from ..MisRegistration import MisRegistration
from ..calibration.CalibrationVault import CalibrationVault
from ..mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration
from ..tools.tools import createFolder


#####################################################################################################################################################
#     # #   #         #     # # #   # # #           #         #       #     # # #
#   #       #       #   #   #       #               #       #   #   #   #   #    #
#   #       #       #   #   # # #   # #     # # #   #       #   #   #   #   # # #
#   #       #       #   #       #   #               #       #   #   #   #   #
#     # #   # # #     #     # # #   # # #           # # #     #       #     #
#####################################################################################################################################################

def run_cl_from_phase_screens(param,ao_object,phase_screens):
        
    nLoop = len(phase_screens)
    gain_cl = param['gainCL']
    
    #% ------------------------------------ Noise Update ------------------------------------
    # set magnitude of the NGS
    ao_object.ngs.nPhoton = param['nPhotonPerSubaperture'] * (1/ao_object.tel.samplingTime)/((ao_object.tel.D/param['nSubaperture'] )**2)
    
    extraName =''
    # photon Noise
    ao_object.wfs.cam.photonNoise = param['photonNoise']
    if ao_object.wfs.cam.photonNoise:
        extraName+=('photonNoise_')
    
    # RON
    ao_object.wfs.cam.readoutNoise = param['readoutNoise']
    
    if ao_object.wfs.cam.readoutNoise:
        extraName+=('readoutNoise_')
    
    
    #% ------------------------------------ Calibration ------------------------------------
    if ao_object.calib.D.shape[1] != param['nModes']:
        calib_cl    = CalibrationVault(ao_object.calib.D[:,:param['nModes']])
    else:
        calib_cl = ao_object.calib
    M2C_cl      = ao_object.M2C_cl
        
    #%% ------------------------------------ mis-registrations  ------------------------------------
    
    misRegistration_cl = MisRegistration(param)
    
    misRegistration_cl.show()
    ao_object.dm.misReg.show()
    
#    case with no mis-registration
    if misRegistration_cl == ao_object.dm.misReg:
        dm_cl = ao_object.dm
    else:
        dm_cl = applyMisRegistration(ao_object.tel,misRegistration_cl,param)

    #%% ------------------------------------ closed loop data to be saved  ------------------------------------
    
    wfs_signals             = np.zeros([nLoop,ao_object.wfs.nSignal])
    dm_commands             = np.zeros([nLoop,ao_object.dm.nValidAct])
    modal_coefficients_res  = np.zeros([nLoop,ao_object.M2C_cl.shape[1]])
    modal_coefficients_turb = np.zeros([nLoop,ao_object.M2C_cl.shape[1]])
    ao_residuals            = np.zeros(nLoop)
    ao_turbulence           = np.zeros(nLoop)
    petalling_nm            = np.zeros([6,nLoop])

    #%% ------------------------------------ destination folder  ------------------------------------
    
    misregistrationFolder = misRegistration_cl.misRegName
    destinationFolder =  param['pathOutput']+ '/'+param['name'] +'/'+ misregistrationFolder + '/' + extraName + '/'
    createFolder(destinationFolder)
    
    gain_cl = param['gainCL']
    
    # combine with atmosphere
    ao_object.tel+ao_object.atm
    plt.close('all')
    
    # initialize DM commands
    dm_cl.coefs=0
    
    #%% ------------------------------------ Modal Gains  ------------------------------------

    ModalGainsMatrix = ao_object.gOpt
    
    # Loop initialization
    reconstructor = np.matmul(M2C_cl,np.matmul(ModalGainsMatrix,calib_cl.M))
    get_le_psf = True
    PSF_LE              = []


    #%%
    # propagate through th whole system
    ao_object.ngs*ao_object.tel*dm_cl*ao_object.wfs
    
    # setup the display
    if ao_object.display:
        from OOPAO.tools.displayTools import cl_plot
    
        plot_obj = cl_plot(list_fig          = [phase_screens[0],phase_screens[0]],\
                           type_fig          = ['imshow','imshow'],\
                           list_title        = ['Input OPD [m]','Residual OPD [m]'],\
                           list_lim          = [None,None],\
                           list_label        = [None,None],\
                           n_subplot         = [2,1],\
                           list_display_axis = [None,None],\
                           list_ratio        = [[1,1],[1,1]])
    
    
    from OOPAO.OPD_map import OPD_map
    static_opd = OPD_map(telescope=ao_object.tel)
    
    wfsSignal = np.zeros(ao_object.wfs.nSignal)
    
    ao_object.tel - ao_object.atm
    ao_object.tel.isPaired = True
    for i_loop in range(nLoop):
        a= time.time()
        # update atmospheric phase screen using input phase screens
        static_opd.OPD = phase_screens[i_loop]
        # reset OPD 
        ao_object.tel.resetOPD()
        # save phase variance
        ao_object.tel.src * ao_object.tel * static_opd
        ao_turbulence[i_loop]=np.std(ao_object.tel.OPD[np.where(ao_object.tel.pupil>0)])*1e9
    
        # save turbulent phase
        turbPhase=ao_object.tel.src.phase
        turb_OPD = np.reshape(ao_object.tel.OPD,ao_object.tel.resolution**2)
    
        # propagate to the WFS with the CL commands applied
        ao_object.tel*dm_cl*ao_object.wfs
            
        dm_cl.coefs=dm_cl.coefs-gain_cl*np.matmul(reconstructor,wfsSignal)
        
        dm_commands[i_loop,:] = dm_cl.coefs
    
        ao_residuals[i_loop]=np.std(ao_object.tel.OPD[np.where(ao_object.tel.pupil>0)])*1e9
    
        # store the slopes after computing the commands => 2 frames delay
        wfsSignal=ao_object.wfs.signal
        b= time.time()
        if get_le_psf:
            if i_loop >100:
                ao_object.tel.computePSF(zeroPaddingFactor = 4)
                PSF_LE.append(ao_object.tel.PSF)
            
        if ao_object.display==True:
            cl_plot(list_fig   = [static_opd.OPD,ao_object.tel.OPD],plt_obj = plot_obj)
            plt.pause(0.01)
            if plot_obj.keep_going is False:
                break
    
        print('Loop'+str(i_loop)+'/'+str(nLoop)+'Turbulence: '+str(ao_turbulence[i_loop])+' -- Residual:' +str(ao_residuals[i_loop])+ '\n')
        print('Elapsed Time: ' + str(b-a) + ' s ')
    


    dataCL = dict()
    dataCL['ao_residual']          = ao_residuals
    dataCL['ao_turbulence']        = ao_turbulence
    dataCL['long_exposure_psf']    = np.mean(np.asarray(PSF_LE),axis =0)


    return dataCL


